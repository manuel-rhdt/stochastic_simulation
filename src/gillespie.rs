use std;
use std::borrow::{Borrow, BorrowMut};

use rand;

type Count = f64;

pub const MAX_NUM_REACTANTS: usize = 2;
pub const MAX_NUM_PRODUCTS: usize = 2;

#[derive(Clone, Debug)]
pub struct ReactionNetwork {
    pub k: Vec<f64>,
    pub reactants: Vec<[Option<u32>; MAX_NUM_REACTANTS]>,
    pub products: Vec<[Option<u32>; MAX_NUM_PRODUCTS]>,
}

impl ReactionNetwork {
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        self.k.len()
    }
}

pub fn calc_propensities(
    propensities: &mut [f64],
    components: &[Count],
    reactions: &ReactionNetwork,
) {
    for i in 0..reactions.len() {
        propensities[i] = reactions.k[i];
        for reactant in reactions.reactants[i].iter().filter_map(|&x| x) {
            propensities[i] *= components[reactant as usize] as f64
        }
    }
}

pub fn try_propagate_time(
    random_variate: f64,
    timestamp: f64,
    next_ext_timestamp: f64,
    total_propensity: f64,
) -> (bool, f64) {
    let max_time_step = next_ext_timestamp - timestamp;

    if max_time_step <= 0.0 {
        return (false, 0.0);
    }

    assert!(total_propensity >= 0.0);

    if max_time_step * total_propensity < random_variate {
        (false, max_time_step)
    } else {
        let time_step = random_variate / total_propensity;
        (true, time_step)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Trajectory<T, C, R> {
    pub length: usize,
    pub num_components: usize,
    pub timestamps: T,
    pub components: C,
    pub reaction_events: Option<R>,
}

impl<T, C, R> Trajectory<T, C, R> {
    pub fn len(&self) -> usize {
        self.length
    }

    pub fn num_components(&self) -> usize {
        self.num_components
    }

    pub fn get_component<U>(&self, comp: usize) -> &[U]
    where
        C: Borrow<[U]>,
    {
        self.components
            .borrow()
            .chunks(self.length)
            .nth(comp)
            .expect("Access out of bounds.")
    }

    pub fn get_component_mut<U>(&mut self, comp: usize) -> &mut [U]
    where
        C: BorrowMut<[U]>,
    {
        self.components
            .borrow_mut()
            .chunks_mut(self.length)
            .nth(comp)
            .expect("Access out of bounds.")
    }

    pub fn as_ref<X: ?Sized, Y: ?Sized, Z: ?Sized>(&self) -> Trajectory<&X, &Y, &Z>
    where
        T: AsRef<X>,
        C: AsRef<Y>,
        R: AsRef<Z>,
    {
        Trajectory {
            length: self.length,
            num_components: self.num_components,
            timestamps: self.timestamps.as_ref(),
            components: self.components.as_ref(),
            reaction_events: self.reaction_events.as_ref().map(AsRef::as_ref),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TrajectoryArray<T, C, R> {
    count: usize,
    inner: Trajectory<T, C, R>,
}

impl<T, C, R> TrajectoryArray<T, C, R> {
    pub fn from_trajectory(traj: Trajectory<T, C, R>) -> Self {
        TrajectoryArray {
            count: 1,
            inner: traj,
        }
    }

    pub fn num_steps(&self) -> usize {
        self.inner.len()
    }

    pub fn num_components(&self) -> usize {
        self.inner.num_components()
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn as_ref<X: ?Sized, Y: ?Sized, Z: ?Sized>(&self) -> TrajectoryArray<&X, &Y, &Z>
    where
        T: AsRef<X>,
        C: AsRef<Y>,
        R: AsRef<Z>,
    {
        TrajectoryArray {
            count: self.count,
            inner: self.inner.as_ref(),
        }
    }

    pub fn get<'a, X, Y, Z>(&'a self, index: usize) -> Trajectory<&'a [X], &'a [Y], &'a [Z]>
    where
        T: Borrow<[X]>,
        C: Borrow<[Y]>,
        R: Borrow<[Z]>,
    {
        if index >= self.len() {
            panic!("Index out of bounds")
        }
        let num_steps = self.num_steps();
        let num_components = self.num_components();
        let timestamps = self
            .inner
            .timestamps
            .borrow()
            .chunks(num_steps)
            .nth(index)
            .unwrap();
        let components = self
            .inner
            .components
            .borrow()
            .chunks(num_steps * num_components)
            .nth(index)
            .unwrap();
        let reaction_events = if let Some(re) = &self.inner.reaction_events {
            Some(re.borrow().chunks(num_steps - 1).nth(index).unwrap())
        } else {
            None
        };

        Trajectory {
            length: num_steps,
            num_components,
            timestamps,
            components,
            reaction_events,
        }
    }

    pub fn get_mut<'a, X, Y, Z>(
        &'a mut self,
        index: usize,
    ) -> Trajectory<&'a mut [X], &'a mut [Y], &'a mut [Z]>
    where
        T: BorrowMut<[X]>,
        C: BorrowMut<[Y]>,
        R: BorrowMut<[Z]>,
    {
        if index >= self.len() {
            panic!("Index out of bounds")
        }
        let num_steps = self.num_steps();
        let num_components = self.num_components();
        let timestamps = self
            .inner
            .timestamps
            .borrow_mut()
            .chunks_mut(num_steps)
            .nth(index)
            .unwrap();
        let components = self
            .inner
            .components
            .borrow_mut()
            .chunks_mut(num_steps * num_components)
            .nth(index)
            .unwrap();
        let reaction_events = if let Some(re) = &mut self.inner.reaction_events {
            Some(
                re.borrow_mut()
                    .chunks_mut(num_steps - 1)
                    .nth(index)
                    .unwrap(),
            )
        } else {
            None
        };

        Trajectory {
            length: num_steps,
            num_components,
            timestamps,
            components,
            reaction_events,
        }
    }
}

#[derive(Debug)]
struct Simulation<'ext, 'other, Rng: rand::Rng> {
    rng: &'other mut Rng,
    current_time: f64,
    ext_progress: usize,
    propensities: &'other mut [f64],
    components: &'other mut [Count],
    ext_trajectory: Option<Trajectory<&'ext [f64], &'ext [Count], &'ext [u32]>>,
    reactions: &'other ReactionNetwork,
}

impl<'ext, 'other, Rng: rand::Rng> Simulation<'ext, 'other, Rng> {
    pub fn new(
        components: &'other mut [Count],
        propensities: &'other mut [f64],
        ext_trajectory: Option<Trajectory<&'ext [f64], &'ext [Count], &'ext [u32]>>,
        reactions: &'other ReactionNetwork,
        rng: &'other mut Rng,
    ) -> Self {
        Self {
            current_time: 0.0,
            ext_progress: 0,
            propensities,
            components,
            ext_trajectory,
            reactions,
            rng,
        }
    }

    pub fn ext_len(&self) -> usize {
        self.ext_trajectory.as_ref().map(|t| t.len()).unwrap_or(0)
    }

    pub fn num_ext_components(&self) -> usize {
        self.ext_trajectory
            .as_ref()
            .map(|t| t.num_components())
            .unwrap_or(0)
    }

    pub fn next_ext_timestamp(&self) -> f64 {
        if self.ext_progress >= self.ext_len() {
            std::f64::INFINITY
        } else if let Some(traj) = &self.ext_trajectory {
            traj.timestamps[self.ext_progress]
        } else {
            panic!("no external trajectory")
        }
    }

    pub fn next_ext_component(&self, comp_num: usize) -> Count {
        let traj = self
            .ext_trajectory
            .as_ref()
            .expect("no external trajectory");
        let index = std::cmp::min(self.ext_progress, traj.len() - 1);
        traj.get_component(comp_num)[index]
    }

    pub fn select_reaction(&mut self) -> usize {
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

    pub fn update_components(&mut self, selected_reaction: usize) {
        for &reactant in &self.reactions.reactants[selected_reaction] {
            if let Some(reactant) = reactant {
                self.components[reactant as usize] -= 1 as Count
            }
        }
        for &product in &self.reactions.products[selected_reaction] {
            if let Some(product) = product {
                self.components[product as usize] += 1 as Count
            }
        }
    }

    pub fn propagate_time(&mut self) -> (f64, usize) {
        let mut random_variate = -self.rng.gen::<f64>().ln();

        loop {
            calc_propensities(&mut self.propensities, &self.components, &self.reactions);
            let total_propensity: f64 = self.propensities.iter().sum();

            let (perform_reaction, timestep) = try_propagate_time(
                random_variate,
                self.current_time,
                self.next_ext_timestamp(),
                total_propensity,
            );
            self.current_time += timestep;

            if perform_reaction {
                let selected_reaction = self.select_reaction();
                self.update_components(selected_reaction);
                return (self.current_time, selected_reaction);
            } else {
                random_variate -= timestep * total_propensity;
                // update the external trajectory
                self.ext_progress += 1;
                for i in 0..self.num_ext_components() {
                    self.components[i] = self.next_ext_component(i);
                }
            }
        }
    }
}

type SimulatedTrajectoryArray = TrajectoryArray<Vec<f64>, Vec<f64>, Vec<u32>>;

pub fn simulate(
    count: usize,
    length: usize,
    initial_values: &[f64],
    reactions: &ReactionNetwork,
    ext_trajectory: Option<TrajectoryArray<&[f64], &[f64], &[u32]>>,
    rng: &mut impl rand::Rng,
) -> SimulatedTrajectoryArray {
    let mut propensities = vec![0.0; reactions.len()];
    let mut components = initial_values.to_owned();

    let num_ext_trajectories = ext_trajectory.as_ref().map(|x| x.len()).unwrap_or(0);

    let num_ext_components = ext_trajectory.map(|x| x.num_components()).unwrap_or(0);
    let num_components = initial_values.len() - num_ext_components;
    assert!(num_components > 0);

    let mut ta = TrajectoryArray {
        count,
        inner: Trajectory {
            length,
            num_components,
            timestamps: vec![0.0; length * count],
            components: vec![0.0; length * count * num_components],
            reaction_events: Some(vec![0_u32; (length - 1) * count]),
        },
    };

    for i in 0..count {
        let mut sim = Simulation::new(
            &mut components,
            &mut propensities,
            ext_trajectory
                .as_ref()
                .map(|x| x.get(i % num_ext_trajectories)),
            reactions,
            rng,
        );

        let mut sim_traj = ta.get_mut(i);

        for comp in 0..num_components {
            sim_traj.get_component_mut(comp)[0] = initial_values[comp + num_ext_components];
        }
        for j in 1..length {
            let (t, selected_reaction) = sim.propagate_time();
            sim_traj.timestamps[j] = t;
            sim_traj
                .reaction_events
                .as_mut()
                .map(|re| re[j - 1] = selected_reaction as u32);

            for comp in 0..num_components {
                sim_traj.get_component_mut(comp)[j] = sim.components[comp + num_ext_components];
            }
        }
    }

    ta
}
