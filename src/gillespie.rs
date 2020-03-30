use std;

use crate::kde::NormalKernelDensityEstimate;

use arrayvec::ArrayVec;
use rand;

type Count = f64;

pub const MAX_NUM_REACTANTS: usize = 2;
pub const MAX_NUM_PRODUCTS: usize = 2;

#[derive(Clone, Debug)]
pub struct ReactionNetwork {
    pub k: Vec<f64>,
    pub reactants: Vec<ArrayVec<[u32; MAX_NUM_REACTANTS]>>,
    pub products: Vec<ArrayVec<[u32; MAX_NUM_PRODUCTS]>>,
}

impl ReactionNetwork {
    #[allow(unused)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        self.k.len()
    }
}

unsafe fn calc_propensities(
    propensities: &mut [f64],
    components: &[Count],
    reactions: &ReactionNetwork,
) {
    assert_eq!(propensities.len(), reactions.len());
    #[allow(clippy::needless_range_loop)]
    for i in 0..reactions.len() {
        *propensities.get_unchecked_mut(i) = *reactions.k.get_unchecked(i);
        for &reactant in reactions.reactants.get_unchecked(i).iter() {
            *propensities.get_unchecked_mut(i) *=
                *components.get_unchecked(reactant as usize) as f64
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Trajectory<T, C, R> {
    pub timestamps: T,
    pub components: Vec<C>,
    pub reaction_events: Option<R>,
}

impl<T: AsRef<[f64]>, C: AsRef<[Count]>, R> Trajectory<T, C, R> {
    pub fn iter(&self) -> TrajectoryIter<'_, T, C, R> {
        let components = self.components.iter().map(|c| c.as_ref()[0]).collect();
        TrajectoryIter {
            trajectory: self,
            components,
            progress: 0,
        }
    }

    pub fn rev_iter(&self) -> RevTrajectoryIter<'_, T, C, R> {
        let len = self.components[0].as_ref().len();
        let components = self
            .components
            .iter()
            .map(|c| c.as_ref()[len - 1])
            .collect();
        RevTrajectoryIter {
            trajectory: self,
            components,
            progress: len - 1,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct Event {
    pub time: f64,
}

/// An iterator yielding a trajectory.
///
/// `advance` and `update` have to be called in lock-step, else the behavior is not defined.
pub trait TrajectoryIterator {
    type Ex;

    fn advance(&mut self) -> Option<Event>;
    fn update(&mut self) -> Self::Ex;

    fn components(&self) -> &[Count];

    fn next(&mut self) -> Option<(Event, Self::Ex)> {
        if let Some(event) = self.advance() {
            let ex = self.update();
            Some((event, ex))
        } else {
            None
        }
    }

    fn exhaust(&mut self) {
        while self.next().is_some() {}
    }

    fn cap(self, cap: f64) -> CappedTrajectory<Self>
    where
        Self: Sized,
    {
        CappedTrajectory {
            inner: self,
            cap,
            accumulated_time: 0.0,
        }
    }

    fn collect(mut self) -> Trajectory<Vec<f64>, Vec<Count>, Vec<Self::Ex>>
    where
        Self: Sized,
    {
        let mut timestamps = vec![0.0];
        let mut component_traj = vec![];
        for &comp in self.components() {
            component_traj.push(vec![comp]);
        }
        let mut reaction_events = vec![];

        let mut current_time = 0.0;
        while let Some((Event { time }, reaction_event)) = self.next() {
            current_time += time;
            timestamps.push(current_time);
            for (comp_array, &comp) in component_traj.iter_mut().zip(self.components()) {
                comp_array.push(comp);
            }
            reaction_events.push(reaction_event)
        }

        Trajectory {
            timestamps,
            components: component_traj,
            reaction_events: Some(reaction_events),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrajectoryIter<'traj, T, C, R> {
    trajectory: &'traj Trajectory<T, C, R>,
    components: Vec<f64>,
    progress: usize,
}

impl<'traj, T: AsRef<[f64]>, C: AsRef<[Count]>, R: AsRef<[u32]>> TrajectoryIterator
    for TrajectoryIter<'traj, T, C, R>
{
    type Ex = u32;

    fn advance(&mut self) -> Option<Event> {
        let timestamps = self.trajectory.timestamps.as_ref();
        let current_time = timestamps[self.progress];
        timestamps.get(self.progress + 1).map(|&time| Event {
            time: time - current_time,
        })
    }

    fn update(&mut self) -> u32 {
        self.progress += 1;
        for (comp_array, comp) in self
            .trajectory
            .components
            .iter()
            .zip(self.components.iter_mut())
        {
            *comp = comp_array.as_ref()[self.progress];
        }
        self.trajectory
            .reaction_events
            .as_ref()
            .map(|r| r.as_ref()[self.progress - 1])
            .unwrap()
    }

    fn components(&self) -> &[Count] {
        &self.components
    }
}

#[derive(Debug, Clone)]
pub struct RevTrajectoryIter<'traj, T, C, R> {
    trajectory: &'traj Trajectory<T, C, R>,
    components: Vec<f64>,
    progress: usize,
}

impl<'traj, T: AsRef<[f64]>, C: AsRef<[Count]>, R: AsRef<[u32]>> TrajectoryIterator
    for RevTrajectoryIter<'traj, T, C, R>
{
    type Ex = u32;

    fn advance(&mut self) -> Option<Event> {
        let timestamps = self.trajectory.timestamps.as_ref();
        let current_time = timestamps[self.progress];
        if self.progress == 0 {
            return None;
        }
        timestamps.get(self.progress - 1).map(|&time| Event {
            time: current_time - time,
        })
    }

    fn update(&mut self) -> u32 {
        self.progress -= 1;
        for (comp_array, comp) in self
            .trajectory
            .components
            .iter()
            .zip(self.components.iter_mut())
        {
            *comp = comp_array.as_ref()[self.progress];
        }
        self.trajectory
            .reaction_events
            .as_ref()
            .map(|r| r.as_ref()[self.progress])
            .unwrap()
    }

    fn components(&self) -> &[Count] {
        &self.components
    }
}

#[derive(Debug)]
struct SimulatedTrajectory<'a, Rng: rand::Rng> {
    rng: &'a mut Rng,
    propensities: Vec<f64>,
    total_propensity: f64,
    components: Vec<f64>,
    reactions: &'a ReactionNetwork,
    log_rand_var: f64,
}

impl<'a, Rng: rand::Rng> SimulatedTrajectory<'a, Rng> {
    pub fn new(
        mut propensities: Vec<f64>,
        reactions: &'a ReactionNetwork,
        components: Vec<Count>,
        rng: &'a mut Rng,
    ) -> Self {
        assert_eq!(propensities.len(), reactions.len());
        unsafe { calc_propensities(&mut propensities, &components, reactions) };
        let total_propensity = propensities.iter().sum();
        SimulatedTrajectory {
            log_rand_var: -rng.gen::<f64>().ln(),
            propensities,
            total_propensity,
            components,
            reactions,
            rng,
        }
    }

    fn select_reaction(&mut self) -> usize {
        let r: f64 = self.rng.gen();
        let total_propensity: f64 = self.propensities.iter().sum();

        let mut selected_reaction = 0;
        let mut acc = self.propensities[0] / total_propensity;

        while r > acc && selected_reaction + 1 < self.propensities.len() {
            selected_reaction += 1;
            acc += self.propensities[selected_reaction] / total_propensity;
        }

        selected_reaction
    }

    #[allow(clippy::unnecessary_cast)]
    fn update_components_with_reaction(&mut self, selected_reaction: usize) {
        for &reactant in &self.reactions.reactants[selected_reaction] {
            self.components[reactant as usize] -= 1 as Count
        }
        for &product in &self.reactions.products[selected_reaction] {
            self.components[product as usize] += 1 as Count
        }
    }

    fn total_propensity(&mut self) -> f64 {
        self.total_propensity
    }

    fn calc_propensities(&mut self) {
        unsafe {
            calc_propensities(&mut self.propensities, &self.components, self.reactions);
        }
        self.total_propensity = self.propensities.iter().sum();
    }

    fn time_step(&mut self) -> f64 {
        self.log_rand_var / self.total_propensity()
    }

    fn update_components(&mut self) -> u32 {
        let selected_reaction = self.select_reaction();
        self.update_components_with_reaction(selected_reaction);
        self.calc_propensities();
        selected_reaction as u32
    }
}

impl<'a, Rng: rand::Rng> TrajectoryIterator for SimulatedTrajectory<'a, Rng> {
    type Ex = u32;

    fn advance(&mut self) -> Option<Event> {
        Some(Event {
            time: self.time_step(),
        })
    }

    fn update(&mut self) -> u32 {
        self.log_rand_var = -self.rng.gen::<f64>().ln();
        self.update_components()
    }

    fn components(&self) -> &[Count] {
        &self.components
    }
}

pub struct DrivenTrajectory<'a, Rng: rand::Rng, ExtTraj: TrajectoryIterator> {
    external_trajectory: ExtTraj,
    sim: SimulatedTrajectory<'a, Rng>,
    remaining_constant_signal_time: f64,
}

impl<'a, Rng: rand::Rng, ExtTraj: TrajectoryIterator> DrivenTrajectory<'a, Rng, ExtTraj> {
    fn new(sim: SimulatedTrajectory<'a, Rng>, mut external_trajectory: ExtTraj) -> Self {
        let remaining_constant_signal_time = external_trajectory
            .advance()
            .map(|event| event.time)
            .unwrap_or(std::f64::INFINITY);
        DrivenTrajectory {
            sim,
            external_trajectory,
            remaining_constant_signal_time,
        }
    }

    fn step(&mut self) -> (f64, bool) {
        let sim_step = self.sim.time_step();
        if self.remaining_constant_signal_time < sim_step {
            let step = self.remaining_constant_signal_time;

            self.sim.log_rand_var -= step * self.sim.total_propensity();

            self.external_trajectory.update();
            let components = self.external_trajectory.components();
            self.sim.components[..components.len()].copy_from_slice(components);
            self.sim.calc_propensities();

            self.remaining_constant_signal_time = self
                .external_trajectory
                .advance()
                .map(|event| event.time)
                .unwrap_or(std::f64::INFINITY);
            (step, false)
        } else {
            self.remaining_constant_signal_time -= sim_step;
            (sim_step, true)
        }
    }
}

impl<'a, Rng: rand::Rng, ExtTraj: TrajectoryIterator> TrajectoryIterator
    for DrivenTrajectory<'a, Rng, ExtTraj>
{
    type Ex = u32;

    fn advance(&mut self) -> Option<Event> {
        let mut current_time = 0.0;
        loop {
            let (step, ready) = self.step();
            current_time += step;
            if ready {
                break Some(Event { time: current_time });
            }
        }
    }

    fn update(&mut self) -> u32 {
        self.sim.update()
    }

    fn components(&self) -> &[Count] {
        let num_ext_components = self.external_trajectory.components().len();
        &self.sim.components[num_ext_components..]
    }
}

pub struct CappedTrajectory<Inner: TrajectoryIterator> {
    inner: Inner,
    cap: f64,
    accumulated_time: f64,
}

impl<Inner: TrajectoryIterator> TrajectoryIterator for CappedTrajectory<Inner> {
    type Ex = Inner::Ex;

    fn advance(&mut self) -> Option<Event> {
        if self.accumulated_time >= self.cap {
            return None;
        }
        if let Some(Event { time }) = self.inner.advance() {
            self.accumulated_time += time;
            Some(Event { time })
        } else {
            None
        }
    }

    fn update(&mut self) -> Self::Ex {
        self.inner.update()
    }

    fn components(&self) -> &[Count] {
        self.inner.components()
    }
}

pub fn simulate<'a>(
    initial_values: &[f64],
    reactions: &'a ReactionNetwork,
    rng: &'a mut impl rand::Rng,
) -> impl 'a + TrajectoryIterator<Ex = u32> {
    let propensities = vec![0.0; reactions.len()];
    let num_components = initial_values.len();
    assert!(num_components > 0);
    let components = initial_values.to_owned();

    SimulatedTrajectory::new(propensities, reactions, components, rng)
}

pub fn simulate_ext<'a>(
    initial_values: &[f64],
    reactions: &'a ReactionNetwork,
    external_trajectory: impl 'a + TrajectoryIterator,
    rng: &'a mut impl rand::Rng,
) -> impl 'a + TrajectoryIterator<Ex = u32> {
    let propensities = vec![0.0; reactions.len()];
    let num_components = initial_values.len();
    assert!(num_components > 0);
    let mut components = external_trajectory.components().to_owned();
    components.extend_from_slice(initial_values);

    let sim = SimulatedTrajectory::new(propensities, reactions, components, rng);
    DrivenTrajectory::new(sim, external_trajectory)
}

#[derive(Debug, Clone)]
pub struct SimulationCoordinator<Rng: rand::Rng> {
    pub trajectory_len: f64,
    pub equilibration_time: f64,

    pub response_mean: f64,
    pub signal_mean: f64,

    pub p0_samples: usize,

    pub sig_network: ReactionNetwork,
    pub res_network: ReactionNetwork,

    pub rng: Rng,
}

impl<Rng: rand::Rng> SimulationCoordinator<Rng> {
    pub fn equilibrate_signal(&mut self) -> f64 {
        let mut traj = simulate(&[self.signal_mean], &self.sig_network, &mut self.rng)
            .cap(self.equilibration_time);
        traj.exhaust();
        traj.components()[0]
    }

    pub fn equilibrate_response(&mut self, sig_initial: &[Count]) -> f64 {
        let sig = simulate(sig_initial, &self.sig_network, &mut self.rng)
            .cap(self.equilibration_time)
            .collect();
        let mut res = simulate_ext(
            &[self.response_mean],
            &self.res_network,
            sig.rev_iter(),
            &mut self.rng,
        )
        .cap(self.equilibration_time);
        res.exhaust();
        res.components()[0]
    }

    pub fn equilibrate_respones_dist(
        &mut self,
        sig_initial: &[Count],
    ) -> NormalKernelDensityEstimate {
        let data = (0..self.p0_samples)
            .map(|_| self.equilibrate_response(sig_initial))
            .collect();
        NormalKernelDensityEstimate::new(data)
    }

    pub fn generate_signal(&mut self) -> impl '_ + TrajectoryIterator<Ex = u32> {
        simulate(
            &[self.equilibrate_signal()],
            &self.sig_network,
            &mut self.rng,
        )
        .cap(self.trajectory_len)
    }

    pub fn generate_response<'a>(
        &'a mut self,
        sig: impl 'a + TrajectoryIterator,
    ) -> impl 'a + TrajectoryIterator<Ex = u32> {
        simulate_ext(
            &[self.equilibrate_response(sig.components())],
            &self.res_network,
            sig,
            &mut self.rng,
        )
        .cap(self.trajectory_len)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn trajectory_iter_works1() {
        let trajectory = Trajectory {
            timestamps: vec![0.0, 1.0, 2.0, 3.0, 4.0],
            components: vec![
                vec![11.0, 10.0, 9.0, 8.0, 7.0],
                vec![110.0, 100.0, 90.0, 80.0, 70.0],
            ],
            reaction_events: Some(vec![1, 0, 1, 2]),
        };

        assert_eq!(trajectory, trajectory.iter().collect());
    }

    #[test]
    fn trajectory_rev_iter_works1() {
        let trajectory = Trajectory {
            timestamps: vec![0.0, 2.0, 3.0, 7.0, 19.0],
            components: vec![
                vec![11.0, 10.0, 9.0, 8.0, 7.0],
                vec![110.0, 100.0, 90.0, 80.0, 70.0],
            ],
            reaction_events: Some(vec![1, 0, 1, 2]),
        };

        let rev_trajectory = Trajectory {
            timestamps: vec![0.0, 12.0, 16.0, 17.0, 19.0],
            components: vec![
                vec![7.0, 8.0, 9.0, 10.0, 11.0],
                vec![70.0, 80.0, 90.0, 100.0, 110.0],
            ],
            reaction_events: Some(vec![2, 1, 0, 1]),
        };

        assert_eq!(rev_trajectory, trajectory.rev_iter().collect());
    }
}
