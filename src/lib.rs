use std;

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::TypeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use rand;

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
    components: &[u32],
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
struct Trajectory<'data> {
    length: usize,
    num_components: usize,
    timestamps: &'data [f64],
    components: &'data [u32],
    reaction_events: Option<&'data [u32]>,
}

impl<'data> Trajectory<'data> {
    fn len(&self) -> usize {
        self.length
    }

    fn num_components(&self) -> usize {
        self.num_components
    }

    fn get_component(&self, comp: usize, index: usize) -> u32 {
        self.components[self.len() * comp + index]
    }
}

#[derive(Debug)]
struct Simulation<'ext, 'other> {
    current_time: f64,
    ext_progress: usize,
    propensities: &'other mut [f64],
    components: &'other mut [u32],
    ext_trajectory: Option<Trajectory<'ext>>,
    reactions: &'other ReactionNetwork,
}

impl<'ext, 'other> Simulation<'ext, 'other> {
    pub fn new(
        components: &'other mut [u32],
        propensities: &'other mut [f64],
        ext_trajectory: Option<Trajectory<'ext>>,
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

    pub fn next_ext_component(&self, comp_num: usize) -> u32 {
        let traj = self
            .ext_trajectory
            .as_ref()
            .expect("no external trajectory");
        let index = std::cmp::min(self.ext_progress, traj.len() - 1);
        traj.get_component(comp_num, index)
    }

    pub fn update_components(&mut self, selected_reaction: usize) {
        for &reactant in &self.reactions.reactants[selected_reaction] {
            if let Some(reactant) = reactant {
                self.components[reactant as usize] -= 1
            }
        }
        for &product in &self.reactions.products[selected_reaction] {
            if let Some(product) = product {
                self.components[product as usize] += 1
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

fn assert_dim(buffer: &PyBuffer, dimensions: usize, name: &str) -> PyResult<()> {
    if buffer.dimensions() != dimensions {
        Err(TypeError::py_err(format!("Array {:?} needs to have {:?} dimensions.", name, dimensions)))
    } else {
        Ok(())
    }
}

#[pymodule]
fn accelerate(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "simulate_trajectories")]
    fn simulate_trajectories(
        py: Python,
        timestamps: PyObject,
        trajectory: PyObject,
        reaction_events: PyObject,
        reactions: ReactionNetwork,
        ext_timestamps: Option<PyObject>,
        ext_trajectory: Option<PyObject>,
    ) -> PyResult<()> {
        let timestamps = PyBuffer::get(py, timestamps.cast_as(py)?)?;
        let trajectory = PyBuffer::get(py, trajectory.cast_as(py)?)?;
        let reaction_events = PyBuffer::get(py, reaction_events.cast_as(py)?)?;

        let ext_t;
        let ext_c;
        if let (Some(et), Some(ec)) = (ext_timestamps, ext_trajectory) {
            ext_t = Some(PyBuffer::get(py, et.cast_as(py)?)?);
            ext_c = Some(PyBuffer::get(py, ec.cast_as(py)?)?);
            assert_dim(ext_t.as_ref().unwrap(), 2, "ext_timestamps")?;
            assert_dim(ext_c.as_ref().unwrap(), 3, "ext_trajectory")?;
        } else {
            ext_t = None;
            ext_c = None;
        }

        assert_dim(&timestamps, 2, "timestamps")?;
        assert_dim(&trajectory, 3, "trajectory")?;
        assert_dim(&reaction_events, 2, "reaction_events")?;

        let count = timestamps.shape()[0];
        let length = timestamps.shape()[1];
        let num_components = trajectory.shape()[1];

        let mut t = timestamps.to_vec::<f64>(py)?;
        let mut c = trajectory.to_vec::<u32>(py)?;
        let mut re = reaction_events.to_vec::<u32>(py)?;

        let ext_count = ext_t.as_ref().map(|x| x.shape()[0]).unwrap_or(0);
        let ext_length = ext_t.as_ref().map(|x| x.shape()[1]).unwrap_or(0);
        let num_ext_components = ext_c.as_ref().map(|x| x.shape()[1]).unwrap_or(0);

        if ext_count != count && ext_count > 1 {
            return Err(TypeError::py_err("Could not broadcast shapes!"));
        }

        let et = ext_t.map(|x| x.to_vec::<f64>(py)).transpose()?;
        let ec = ext_c.map(|x| x.to_vec::<u32>(py)).transpose()?;

        // hot loop
        py.allow_threads(|| {
            let mut components = vec![0; num_components + num_ext_components];
            let mut propensities = vec![0.0; reactions.len()];
            for r in 0..count {
                let stride = timestamps.shape()[1];
                let t = &mut t[stride * r..stride * (r + 1)];
                let stride = trajectory.shape()[1] * trajectory.shape()[2];
                let c = &mut c[stride * r..stride * (r + 1)];
                let stride = reaction_events.shape()[1];
                let re = &mut re[stride * r..stride * (r + 1)];

                let ext_traj;
                if let (Some(et), Some(ec)) = (&et, &ec) {
                    let ext_index = r % ext_count; //< broadcasting
                    ext_traj = Some(Trajectory {
                        num_components: 1,
                        length: ext_length,
                        timestamps: &et[ext_length * ext_index..ext_length * (ext_index + 1)],
                        components: &ec[ext_length * ext_index..ext_length * (ext_index + 1)],
                        reaction_events: None,
                    })
                } else {
                    ext_traj = None
                }

                // set the initial component values
                for i in 0..num_ext_components {
                    components[i] = ec.as_ref().unwrap()[i * ext_length + 0]
                }
                for i in 0..num_components {
                    components[i + num_ext_components] = c[i * length + 0]
                }

                let mut sim =
                    Simulation::new(&mut components, &mut propensities, ext_traj, &reactions);

                let mut progress = 0;
                while (progress + 1) < length {
                    progress += 1;
                    let (time, selected_reaction) = sim.propagate_time();
                    t[progress] = time;
                    re[progress - 1] = selected_reaction as u32;
                    for i in 0..num_components {
                        let index = i * length + progress;
                        c[index] = sim.components[num_ext_components + i];
                    }
                }
            }
        });

        timestamps.copy_from_slice(py, &t)?;
        trajectory.copy_from_slice(py, &c)?;
        reaction_events.copy_from_slice(py, &re)?;

        Ok(())
    }

    Ok(())
}
