use super::gillespie::{Event, ReactionNetwork, TrajectoryIterator};

fn propensity_of_event(
    components: &[f64],
    reaction_event: u32,
    reactions: &ReactionNetwork,
) -> f64 {
    reactions.k[reaction_event as usize]
        * reactions.reactants[reaction_event as usize]
            .iter()
            .map(|&reactant| components[reactant as usize])
            .product::<f64>()
}

fn sum_of_reaction_propensities(components: &[f64], reactions: &ReactionNetwork) -> f64 {
    (0..reactions.len() as u32)
        .map(|i| propensity_of_event(components, i, reactions))
        .sum()
}

struct ComponentIter<'r, Signal, Response> {
    signal: Signal,
    response: Response,
    reactions: &'r ReactionNetwork,
    remaining_constant_signal_time: f64,
    components: Vec<f64>,
}

impl<'r, Signal, Response> ComponentIter<'r, Signal, Response>
where
    Signal: TrajectoryIterator,
    Response: TrajectoryIterator<Ex = u32>,
{
    fn new(mut signal: Signal, response: Response, reactions: &'r ReactionNetwork) -> Self {
        let remaining_constant_signal_time = signal
            .advance()
            .map(|event| event.time)
            .unwrap_or(std::f64::INFINITY);
        let num_sig_comp = signal.components().len();
        let num_res_comp = response.components().len();
        ComponentIter {
            signal,
            response,
            reactions,
            remaining_constant_signal_time,
            components: vec![0.0; num_sig_comp + num_res_comp],
        }
    }
}

impl<'r, Signal, Response> Iterator for ComponentIter<'r, Signal, Response>
where
    Signal: TrajectoryIterator,
    Response: TrajectoryIterator<Ex = u32>,
{
    type Item = (f64, f64, f64);

    ///
    /// ```ascii
    ///             |                                    |
    ///             |        +---------------------------| <-- signal[k + 1]
    ///             |========|===========================| <== average[i]
    ///             |        |                           |
    ///          ---|--------+                           | <-- signal[k]
    ///             | signal.timestamp[k + 1]            |
    ///             |                                    |
    ///             +------------------------------------+---> time
    ///     response.timestamp[i]                  response.timestamp[i+1]
    /// ```
    ///
    fn next(&mut self) -> Option<(f64, f64, f64)> {
        let num_ext_comp = self.signal.components().len();
        if let Some(Event { time: delta_t }) = self.response.advance() {
            let mut remaining_constant_response_time = delta_t;
            let mut integrated_propensity = 0.0;
            loop {
                self.components[..num_ext_comp].copy_from_slice(self.signal.components());
                self.components[num_ext_comp..].copy_from_slice(self.response.components());
                if remaining_constant_response_time < self.remaining_constant_signal_time {
                    // the response will change before the signal, therefore we have to
                    // - keep track of time
                    // - perform the response update
                    // - calculate the propensity of the event that happened
                    // - calculate the integrated propensity of the remaining time
                    // - yield the result
                    self.remaining_constant_signal_time -= remaining_constant_response_time;
                    let reaction_event = self.response.update();
                    let event_prop =
                        propensity_of_event(&self.components, reaction_event, self.reactions);
                    integrated_propensity += remaining_constant_response_time
                        * sum_of_reaction_propensities(&self.components, self.reactions);
                    break Some((delta_t, event_prop, integrated_propensity));
                } else {
                    // the signal will change before the response, therefore we havet to
                    // - keep track of time
                    // - integrate the propensity for the amount of time that the signal is const
                    // - update the signal
                    // - get the time when the next signal change will occur
                    remaining_constant_response_time -= self.remaining_constant_signal_time;
                    integrated_propensity += self.remaining_constant_signal_time
                        * sum_of_reaction_propensities(&self.components, self.reactions);
                    self.signal.update();
                    if let Some(Event { time }) = self.signal.advance() {
                        self.remaining_constant_signal_time = time;
                    } else {
                        // When no more events happen to the signal we just say that the signal will
                        // remain constant forever.
                        self.remaining_constant_signal_time = std::f64::INFINITY;
                    }
                }
            }
        } else {
            None
        }
    }
}

/// Returns an Iterator over pairs of timestamps and log_likelihoood values.
///
fn log_likelihood_inner<'a>(
    signal: impl 'a + TrajectoryIterator<Ex = u32>,
    response: impl 'a + TrajectoryIterator<Ex = u32>,
    reactions: &'a ReactionNetwork,
) -> impl 'a + Iterator<Item = (f64, f64)> {
    let component_iter = ComponentIter::new(signal, response, reactions);
    component_iter.scan(
        (0.0, 0.0),
        |(current_time, ll), (dt, inst_rate, integrated_rate)| {
            *current_time += dt;
            *ll += inst_rate.ln() - integrated_rate;
            Some((*current_time, *ll))
        },
    )
}

pub fn log_likelihood<'a>(
    traj_lengths: &'a [f64],
    signal: impl 'a + TrajectoryIterator<Ex = u32>,
    response: impl 'a + TrajectoryIterator<Ex = u32>,
    reactions: &'a ReactionNetwork,
) -> impl 'a + Iterator<Item = f64> {
    let mut ll_iter = log_likelihood_inner(signal, response, reactions).peekable();

    let bin_iter = traj_lengths.iter();

    bin_iter.scan(0.0, move |acc, &bin_edge| {
        while let Some(&(t, lh)) = ll_iter.peek() {
            if t < bin_edge {
                *acc = lh;
                ll_iter.next();
            } else {
                return Some(*acc);
            }
        }
        None
    })
}
