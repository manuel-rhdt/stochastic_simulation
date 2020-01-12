use std;
use std::borrow::{Borrow, BorrowMut};

use pyo3;
use pyo3::buffer::{Element, PyBuffer};
use pyo3::exceptions::TypeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::PyNativeType;
use rand;

mod likelihood;

type Count = f64;

#[derive(Clone, Debug)]
pub struct ReactionNetwork {
    k: Vec<f64>,
    reactants: Vec<[Option<u32>; 2]>,
    products: Vec<[Option<u32>; 2]>,
}

impl ReactionNetwork {
    pub fn len(&self) -> usize {
        self.k.len()
    }
}

impl<'source> FromPyObject<'source> for ReactionNetwork {
    fn extract(obj: &'source PyAny) -> PyResult<Self> {
        let reactants = obj.get_item("reactants")?;
        let products = obj.get_item("products")?;

        let reactants: Vec<Vec<u32>> = reactants.extract()?;
        let mut new_reactants = Vec::new();
        for r in reactants {
            new_reactants.push([r.get(0).cloned(), r.get(1).cloned()]);
        }

        let products: Vec<Vec<u32>> = products.extract()?;
        let mut new_products = Vec::new();
        for p in products {
            new_products.push([p.get(0).cloned(), p.get(1).cloned()]);
        }

        Ok(ReactionNetwork {
            k: obj.get_item("k")?.extract()?,
            reactants: new_reactants,
            products: new_products,
        })
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

pub fn select_reaction(propensities: &[f64]) -> usize {
    let r: f64 = rand::random();
    let total_propensity: f64 = propensities.iter().sum();

    let mut selected_reaction = 0;
    let mut acc = propensities[0] / total_propensity;

    while r > acc && selected_reaction + 1 < propensities.len() {
        selected_reaction += 1;
        acc += propensities[selected_reaction] / total_propensity;
    }

    selected_reaction
}

#[derive(Debug, Clone)]
pub struct Trajectory<T, C, R> {
    length: usize,
    num_components: usize,
    timestamps: T,
    components: C,
    reaction_events: Option<R>,
}

impl<T, C, R> Trajectory<T, C, R> {
    fn len(&self) -> usize {
        self.length
    }

    fn num_components(&self) -> usize {
        self.num_components
    }

    fn get_component<U>(&self, comp: usize) -> &[U]
    where
        C: Borrow<[U]>,
    {
        &self.components.borrow()[self.len() * comp..self.len() * (comp + 1)]
    }

    fn get_component_mut<U>(&mut self, comp: usize) -> &mut [U]
    where
        C: BorrowMut<[U]>,
    {
        let length = self.len();
        &mut self.components.borrow_mut()[length * comp..length * (comp + 1)]
    }

    fn as_ref<X: ?Sized, Y: ?Sized, Z: ?Sized>(&self) -> Trajectory<&X, &Y, &Z>
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

pub struct TrajectoryArray<T, C, R> {
    count: usize,
    inner: Trajectory<T, C, R>,
}

impl<T, C, R> TrajectoryArray<T, C, R> {
    fn from_trajectory(traj: Trajectory<T, C, R>) -> Self {
        TrajectoryArray {
            count: 1,
            inner: traj,
        }
    }

    fn num_steps(&self) -> usize {
        self.inner.len()
    }

    fn num_components(&self) -> usize {
        self.inner.num_components()
    }

    fn len(&self) -> usize {
        self.count
    }

    fn as_ref<X: ?Sized, Y: ?Sized, Z: ?Sized>(&self) -> TrajectoryArray<&X, &Y, &Z>
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

    fn get<'a, X, Y, Z>(&'a self, index: usize) -> Trajectory<&'a [X], &'a [Y], &'a [Z]>
    where
        T: Borrow<[X]>,
        C: Borrow<[Y]>,
        R: Borrow<[Z]>,
    {
        if index >= self.len() {
            panic!("Index out of bounds")
        }
        let stride_t = self.num_steps();
        let stride_c = self.num_steps() * self.num_components();
        let stride_re = stride_t - 1;
        let timestamps = &self.inner.timestamps.borrow()[stride_t * index..stride_t * (index + 1)];
        let components = &self.inner.components.borrow()[stride_c * index..stride_c * (index + 1)];
        let reaction_events = if let Some(re) = &self.inner.reaction_events {
            Some(&re.borrow()[stride_re * index..stride_re * (index + 1)])
        } else {
            None
        };

        Trajectory {
            length: self.inner.len(),
            num_components: self.inner.num_components(),
            timestamps,
            components,
            reaction_events,
        }
    }

    fn get_mut<'a, X, Y, Z>(
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
        let stride_t = self.num_steps();
        let stride_c = self.num_steps() * self.num_components();
        let stride_re = stride_t - 1;
        let num_components = self.num_components();
        let timestamps =
            &mut self.inner.timestamps.borrow_mut()[stride_t * index..stride_t * (index + 1)];
        let components =
            &mut self.inner.components.borrow_mut()[stride_c * index..stride_c * (index + 1)];
        let reaction_events = if let Some(re) = &mut self.inner.reaction_events {
            Some(&mut re.borrow_mut()[stride_re * index..stride_re * (index + 1)])
        } else {
            None
        };

        Trajectory {
            length: stride_t,
            num_components,
            timestamps,
            components,
            reaction_events,
        }
    }
}

fn assert_dim(buffer: &PyBuffer, dimensions: usize, name: &str) -> PyResult<()> {
    if buffer.dimensions() != dimensions {
        Err(TypeError::py_err(format!(
            "Array {:?} needs to have {:?} dimensions.",
            name, dimensions
        )))
    } else {
        Ok(())
    }
}

impl<'source, T, C, R> FromPyObject<'source> for TrajectoryArray<Vec<T>, Vec<C>, Vec<R>>
where
    T: Element + Copy,
    C: Element + Copy,
    R: Element + Copy,
{
    fn extract(obj: &'source PyAny) -> PyResult<Self> {
        let py = obj.py();
        let timestamps = obj.getattr("timestamps")?;
        let timestamps = PyBuffer::get(py, timestamps)?;
        assert_dim(&timestamps, 2, "timestamps")?;
        let count = timestamps.shape()[0];
        let length = timestamps.shape()[1];
        let timestamps = timestamps.to_vec(py)?;

        let components = obj.getattr("components")?;
        let components = PyBuffer::get(py, components)?;
        assert_dim(&components, 3, "components")?;
        if count != components.shape()[0] || length != components.shape()[2] {
            return TypeError::into("shapes of timestamps and components don't match");
        }
        let num_components = components.shape()[1];
        let components = components.to_vec(py)?;

        let reaction_events = obj.getattr("reaction_events")?;
        let reaction_events = if reaction_events.is_none() {
            None
        } else {
            let reaction_events = PyBuffer::get(py, reaction_events)?;
            if count != reaction_events.shape()[0] || length - 1 != reaction_events.shape()[1] {
                return TypeError::into("shapes of timestamps and reaction events don't match");
            }
            assert_dim(&reaction_events, 2, "reaction_events")?;
            Some(reaction_events.to_vec(py)?)
        };

        let inner = Trajectory {
            length,
            num_components,
            timestamps,
            components,
            reaction_events,
        };

        Ok(TrajectoryArray { count, inner })
    }
}

#[derive(Debug)]
struct Simulation<'ext, 'other> {
    current_time: f64,
    ext_progress: usize,
    propensities: &'other mut [f64],
    components: &'other mut [Count],
    ext_trajectory: Option<Trajectory<&'ext [f64], &'ext [Count], &'ext [u32]>>,
    reactions: &'other ReactionNetwork,
}

impl<'ext, 'other> Simulation<'ext, 'other> {
    pub fn new(
        components: &'other mut [Count],
        propensities: &'other mut [f64],
        ext_trajectory: Option<Trajectory<&'ext [f64], &'ext [Count], &'ext [u32]>>,
        reactions: &'other ReactionNetwork,
    ) -> Self {
        Self {
            current_time: 0.0,
            ext_progress: 0,
            propensities,
            components,
            ext_trajectory,
            reactions,
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
        } else {
            if let Some(traj) = &self.ext_trajectory {
                traj.timestamps[self.ext_progress]
            } else {
                panic!("no external trajectory")
            }
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
        let mut random_variate = -rand::random::<f64>().ln();

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
                let selected_reaction = select_reaction(&self.propensities);
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

fn _simulate_trajectories() {}

#[pymodule]
fn accelerate(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "simulate_trajectories")]
    fn simulate_trajectories(
        py: Python,
        out: PyObject,
        reactions: ReactionNetwork,
        ext: Option<PyObject>,
    ) -> PyResult<()> {
        let mut traj: TrajectoryArray<Vec<f64>, Vec<Count>, Vec<u32>> = out.extract(py)?;
        if traj.inner.reaction_events.is_none() {
            return Err(TypeError::py_err(
                "trajectory does not contain reaction events!",
            ));
        }
        let ext: Option<TrajectoryArray<Vec<f64>, Vec<Count>, Vec<u32>>> =
            ext.map(|x| x.extract(py)).transpose()?;
        let ext = ext.as_ref();

        // hot loop
        py.allow_threads(|| {
            let num_components = traj.num_components();
            let num_ext_components = ext.map(|x| x.num_components()).unwrap_or(0);

            let mut components = vec![0.0; num_components + num_ext_components];
            let mut propensities = vec![0 as Count; reactions.len()];
            for r in 0..traj.len() {
                let mut trajectory = traj.get_mut(r);

                let ext_traj;
                if let Some(ext) = ext {
                    let ext_index = r % ext.len(); //< broadcasting
                    ext_traj = Some(ext.get(ext_index));
                } else {
                    ext_traj = None
                }

                // set the initial component values
                for i in 0..num_ext_components {
                    components[i] = ext_traj.as_ref().unwrap().get_component(i)[0];
                }
                for i in 0..num_components {
                    components[i + num_ext_components] = trajectory.get_component(i)[0];
                }

                let mut sim =
                    Simulation::new(&mut components, &mut propensities, ext_traj, &reactions);

                let mut progress = 0;
                while (progress + 1) < trajectory.len() {
                    progress += 1;
                    let (time, selected_reaction) = sim.propagate_time();
                    trajectory.timestamps[progress] = time;
                    trajectory.reaction_events.as_mut().unwrap()[progress - 1] =
                        selected_reaction as u32;
                    for i in 0..num_components {
                        trajectory.get_component_mut(i)[progress] =
                            sim.components[num_ext_components + i];
                    }
                }
            }
        });

        let out = out.as_ref(py);
        let timestamps = PyBuffer::get(py, out.getattr("timestamps")?)?;
        timestamps.copy_from_slice(py, &traj.inner.timestamps)?;
        let trajectory = PyBuffer::get(py, out.getattr("components")?)?;
        trajectory.copy_from_slice(py, &traj.inner.components)?;
        let reaction_events = PyBuffer::get(py, out.getattr("reaction_events")?)?;
        reaction_events.copy_from_slice(py, &traj.inner.reaction_events.unwrap())?;

        Ok(())
    }

    #[pyfn(m, "log_likelihood")]
    fn log_likelihood(
        py: Python,
        response: TrajectoryArray<Vec<f64>, Vec<f64>, Vec<u32>>,
        signal: TrajectoryArray<Vec<f64>, Vec<f64>, Vec<u32>>,
        reactions: ReactionNetwork,
        out: PyObject,
        outer: Option<bool>,
    ) -> PyResult<()> {
        let num_responses = response.len();
        let num_signals = signal.len();
        let outer = outer.unwrap_or(false);

        if !outer {
            if num_responses != num_signals && !(num_responses == 1 || num_signals == 1) {
                return TypeError::into(format!(
                    "Could not broadcast trajectory arrays with lengths {:?} and {:?}.",
                    num_responses, num_signals
                ));
            }
        }

        let out_ref = out.as_ref(py);
        let out = PyBuffer::get(py, &out_ref)?;

        let mut out_vec;
        if !outer {
            assert_dim(&out, 2, "out")?;
            if out.shape()[0] != num_responses.max(num_signals)
                || out.shape()[1] != response.num_steps() - 1
            {
                return TypeError::into("output array has wrong shape");
            }

            out_vec = out.to_vec(py)?;

            py.allow_threads(|| {
                likelihood::log_likelihood(
                    signal.as_ref(),
                    response.as_ref(),
                    &reactions,
                    &mut out_vec,
                );
            });
        } else {
            assert_dim(&out, 3, "out")?;
            let shape = out.shape().to_owned();
            if shape[0] != num_responses
                || shape[1] != num_signals
                || shape[2] != response.num_steps() - 1
            {
                return TypeError::into(format!("output array has wrong shape {:?}", shape));
            }
            out_vec = out.to_vec(py)?;
            py.allow_threads(|| {
                let stride = shape[1] * shape[2];
                for r in 0..response.len() {
                    let out_slice = &mut out_vec[r * stride..(r + 1) * stride];
                    let response = TrajectoryArray::from_trajectory(response.get(r));
                    likelihood::log_likelihood(signal.as_ref(), response, &reactions, out_slice);
                }
            });
        }

        out.copy_from_slice(py, &out_vec)?;

        Ok(())
    }

    Ok(())
}
