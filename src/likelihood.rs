use super::{ReactionNetwork, Trajectory, TrajectoryArray};

fn calculate_sum_of_reaction_propensities(
    components: &[&[f64]],
    reactions: &ReactionNetwork,
    out: &mut [f64],
) {
    for (i, out) in out.iter_mut().enumerate() {
        for j_reaction in 0..reactions.len() {
            let k = reactions.k[j_reaction];
            let reactants = reactions.reactants[j_reaction];

            let mut tmp = k;
            for &reactant in reactants.iter() {
                if let Some(reactant) = reactant {
                    tmp *= components[reactant as usize][i];
                }
            }

            if j_reaction == 0 {
                *out = tmp;
            } else {
                *out += tmp;
            }
        }
    }
}

fn calculate_selected_reaction_propensities(
    components: &[&[f64]],
    reaction_events: &[u32],
    reactions: &ReactionNetwork,
    out: &mut [f64],
) {
    for (i, (&r_index, out)) in reaction_events.iter().zip(out.iter_mut()).enumerate() {
        let k = reactions.k[r_index as usize];
        let reactants = reactions.reactants[r_index as usize];

        *out = k;
        for &reactant in reactants.iter() {
            if let Some(reactant) = reactant {
                *out *= components[reactant as usize][i];
            }
        }
    }
}

struct TrajectoryIter<'a> {
    index: usize,
    timestamps: &'a [f64],
    values: &'a [f64],
}

impl<'a> Iterator for TrajectoryIter<'a> {
    type Item = (f64, (f64, f64));

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.timestamps.len() || self.index >= self.values.len() {
            let last_val = self.values[self.values.len() - 1];
            return Some((std::f64::INFINITY, (last_val, last_val)));
        }
        if self.index == 0 {
            self.index = 1;
        }

        let result = (
            self.timestamps[self.index],
            (self.values[self.index - 1], self.values[self.index]),
        );
        self.index += 1;
        Some(result)
    }
}

/// Average of `trajectory` with `old_timestamp` in the time-intervals specified by
/// `new_timestamps`.
///
/// Note: This function assumes that both `old_timestamps` and `new_timestamps` are
/// ordered.
///
/// # Discussion
///
/// This function is used for the calculation of the mean propensity over a timestep in
/// the response trajectory. Since the propensity for a reaction is variable, the
/// survival probability depends on the time-integrated value of the total propensity.
/// Since the signal trajectory is assumed to be piecewise constant we can calculate the
/// average signal between every pair of response timestamps. If we use this average
/// signal to compute the total propensity, we don't require the calculation of the time
/// integral anymore.
///
/// This function computes the average of a trajectory given by the values `trajectory`
/// at the timestamps `old_timestamps` where the averaging happens for the intervals
/// between pairs of timestamps in `new_timestamps`.
///
/// Returns a list of averages of size `len(new_timestamps) - 1`.
///
/// ```
///             |                                    |
///             |        +---------------------------| <-- trajectory[k + 1]
///             |========|===========================| <== average[i]
///             |        |                           |
///         ---|--------+                           | <-- trajectory[k]
///             | new_timestamps[k]                  |
///             |                                    |
///             +------------------------------------+---> time
///     old_timestamps[i]                  old_timestamps[i+1]
/// ```
///
fn time_average(
    trajectory: &[f64],
    old_timestamps: &[f64],
    new_timestamps: &[f64],
    out: &mut [f64],
    evaluated: &mut [f64],
) {
    let mut traj_iter = TrajectoryIter {
        index: 1,
        timestamps: old_timestamps,
        values: trajectory,
    };

    let nt_high = &new_timestamps[1..];
    let nt_low = &new_timestamps[..new_timestamps.len() - 1];

    let (mut next_trajectory_change, (mut trajectory_value, _)) = traj_iter.next().unwrap();
    for ((mut low, high), (out, evaluated)) in nt_low
        .iter()
        .cloned()
        .zip(nt_high.iter().cloned())
        .zip(out.iter_mut().zip(evaluated.iter_mut()))
    {
        let delta_t = high - low;
        let mut acc = 0.0;
        while low < high {
            while next_trajectory_change <= low {
                let (a, (b, _)) = traj_iter.next().unwrap();
                next_trajectory_change = a;
                trajectory_value = b;
            }

            acc += trajectory_value * (high.min(next_trajectory_change) - low);
            low = next_trajectory_change;
        }

        *out = acc / delta_t;
        *evaluated = trajectory_value;
    }
}

type TrajectoryStd<'a> = Trajectory<&'a [f64], &'a [f64], &'a [u32]>;

fn log_likelihood_inner<'a>(
    signal: TrajectoryStd<'a>,
    response: TrajectoryStd<'a>,
    reactions: &ReactionNetwork,
    out: &mut [f64],
) {
    let mut resampled_sig = Trajectory {
        length: response.len(),
        num_components: signal.num_components(),
        timestamps: response.timestamps,
        components: vec![0.0; signal.num_components() * response.len()],
        reaction_events: Option::<()>::None,
    };
    let mut evaluated_sig = Trajectory {
        length: response.len(),
        num_components: signal.num_components(),
        timestamps: response.timestamps,
        components: vec![0.0; signal.num_components() * response.len()],
        reaction_events: Option::<()>::None,
    };

    for i in 0..signal.num_components() {
        time_average(
            signal.get_component(i),
            signal.timestamps,
            response.timestamps,
            resampled_sig.get_component_mut(i),
            evaluated_sig.get_component_mut(i),
        );
    }

    let mut components = vec![];
    for i in 0..signal.num_components() {
        components.push(resampled_sig.get_component(i));
    }
    for i in 0..response.num_components() {
        components.push(response.get_component(i));
    }

    let mut averaged_rates = vec![0.0; response.length];
    calculate_sum_of_reaction_propensities(&components, reactions, &mut averaged_rates);

    components.clear();
    for i in 0..signal.num_components() {
        components.push(evaluated_sig.get_component(i));
    }
    for i in 0..response.num_components() {
        components.push(response.get_component(i));
    }

    let mut instantaneous_rates = vec![0.0; response.length];
    calculate_selected_reaction_propensities(
        &components,
        &response.reaction_events.unwrap(),
        reactions,
        &mut instantaneous_rates,
    );

    let dt = response.timestamps[1..]
        .iter()
        .zip(response.timestamps[..response.len() - 1].iter())
        .map(|(&high, &low)| high - low);

    let mut acc = 0.0;
    for (((avrg_rate, inst_rate), dt), out) in averaged_rates
        .iter()
        .zip(instantaneous_rates.iter())
        .zip(dt)
        .zip(out.iter_mut())
    {
        acc += inst_rate.ln() - avrg_rate * dt;
        *out = acc;
    }
}

type TrajectoryArrayStd<'a> = TrajectoryArray<&'a [f64], &'a [f64], &'a [u32]>;

pub fn log_likelihood(
    traj_lengths: &[f64],
    signal: TrajectoryArrayStd<'_>,
    response: TrajectoryArrayStd<'_>,
    reactions: &ReactionNetwork,
    out: &mut [f64],
) {
    let num_responses = response.len();
    let num_signals = signal.len();

    assert!(num_responses == num_signals || (num_responses == 1 || num_signals == 1));

    let out_size = num_responses.max(num_signals);

    let mut tmp = vec![0.0; response.num_steps()];
    for (idx, out) in (0..out_size).zip(out.chunks_mut(response.num_steps())) {
        let r_idx = idx % num_responses;
        let s_idx = idx % num_signals;
        let response = response.get(r_idx);
        log_likelihood_inner(signal.get(s_idx), response, reactions, &mut tmp);

        let bin_iter = traj_lengths.iter().zip(out.iter_mut());
        let mut result_iter = response.timestamps.iter().zip(tmp.iter()).peekable();
        for (&bin_edge, out) in bin_iter {
            while let Some(&(&t, &lh)) = result_iter.peek() {
                if t < bin_edge {
                    *out += lh;
                    result_iter.next();
                } else {
                    break;
                }
            }
            *out = std::f64::NAN;
        }
    }
}
