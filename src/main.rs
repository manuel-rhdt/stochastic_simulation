mod gillespie;
mod likelihood;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::prelude::*;
use std::path::PathBuf;

use gillespie::{simulate, ReactionNetwork, TrajectoryArray, MAX_NUM_PRODUCTS, MAX_NUM_REACTANTS};
use likelihood::log_likelihood;

use base64;
use ndarray::{Array, Array1, Array2, Array3, ArrayView1, Axis};
use ndarray_npy::WriteNpyExt;
use rayon;
use rayon::prelude::*;
use serde::Deserialize;
use toml;

fn calculate_hash<T: Hash>(t: &T) -> String {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    let val = s.finish();
    let array: [u8; 8] = unsafe { std::mem::transmute(val) };
    base64::encode_config(&array, base64::URL_SAFE_NO_PAD)
}

fn conditional_likelihood(
    traj_lengths: &[f64],
    num_res: usize,
    batch: usize,
    length: usize,
    sig_network: &ReactionNetwork,
    res_network: &ReactionNetwork,
) -> Array1<f64> {
    let sig = simulate(1, length, &vec![50.0; 1], sig_network, None);
    let mut result = Array3::zeros((num_res / batch, batch, traj_lengths.len()));
    for mut out in result.outer_iter_mut() {
        let res = simulate(
            batch,
            length,
            &vec![50.0; 2],
            res_network,
            Some(sig.as_ref()),
        );
        log_likelihood(
            traj_lengths,
            sig.as_ref(),
            res.as_ref(),
            res_network,
            out.as_slice_mut().unwrap(),
        );
    }
    let result = result.into_shape((num_res, traj_lengths.len())).unwrap();
    result.mean_axis(Axis(0)).unwrap()
}

fn marginal_likelihood(
    traj_lengths: &[f64],
    length: usize,
    signals_pre: TrajectoryArray<&[f64], &[f64], &[u32]>,
    sig_network: &ReactionNetwork,
    res_network: &ReactionNetwork,
) -> Array1<f64> {
    let sig = simulate(1, length, &[50.0; 1], sig_network, None);
    let res = simulate(1, length, &[50.0; 2], res_network, Some(sig.as_ref()));

    let mut result = Array2::zeros((signals_pre.len(), traj_lengths.len()));
    log_likelihood(
        traj_lengths,
        signals_pre,
        res.as_ref(),
        res_network,
        result.as_slice_mut().unwrap(),
    );

    result.map_axis(Axis(0), logsumexp) - (signals_pre.len() as f64).ln()
}

pub fn logsumexp(values: ArrayView1<f64>) -> f64 {
    let max = values.fold(std::f64::NEG_INFINITY, |a, &b| a.max(b));
    values.iter().map(|&x| (x - max).exp()).sum::<f64>().ln() + max
}

#[derive(Deserialize, Debug, Clone, PartialEq)]
struct Config {
    output: PathBuf,
    batch_size: usize,
    length: usize,
    conditional_entropy: ConfigConditionalEntropy,
    marginal_entropy: ConfigMarginalEntropy,
    signal: ConfigReactionNetwork,
    response: ConfigReactionNetwork,
}

#[derive(Deserialize, Debug, Copy, Clone, PartialEq, Eq)]
struct ConfigConditionalEntropy {
    num_signals: usize,
    responses_per_signal: usize,
}

#[derive(Deserialize, Debug, Copy, Clone, PartialEq, Eq)]
struct ConfigMarginalEntropy {
    num_signals: usize,
    num_responses: usize,
}

#[derive(Deserialize, Debug, Clone, PartialEq)]
struct ConfigReactionNetwork {
    initial: f64,
    components: Vec<String>,
    reactions: Vec<Reaction>,
}

impl ConfigReactionNetwork {
    fn to_reaction_network(&self) -> ReactionNetwork {
        let components: HashSet<_> = self.components.iter().collect();
        let mut external_components = HashSet::new();
        for reaction in &self.reactions {
            for name in reaction.reactants.iter().chain(reaction.products.iter()) {
                if !components.contains(name) {
                    external_components.insert(name);
                }
            }
        }
        let num_ext_components = external_components.len();

        let mut name_table = HashMap::new();
        name_table.extend(
            external_components
                .iter()
                .enumerate()
                .map(|(value, &key)| (key, value)),
        );
        name_table.extend(
            components
                .iter()
                .enumerate()
                .map(|(value, &key)| (key, value + num_ext_components)),
        );

        let mut k = Vec::with_capacity(self.reactions.len());
        let mut reactants = Vec::with_capacity(self.reactions.len());
        let mut products = Vec::with_capacity(self.reactions.len());

        for reaction in &self.reactions {
            k.push(reaction.k);
            let mut r = [None; MAX_NUM_REACTANTS];
            for (i, reactant) in reaction.reactants.iter().enumerate() {
                r[i] = Some(name_table[reactant] as u32);
            }
            reactants.push(r);
            let mut p = [None; MAX_NUM_PRODUCTS];
            for (i, product) in reaction.products.iter().enumerate() {
                p[i] = Some(name_table[product] as u32);
            }
            products.push(p);
        }

        ReactionNetwork {
            k,
            reactants,
            products,
        }
    }
}

#[derive(Deserialize, Debug, Clone, PartialEq)]
struct Reaction {
    k: f64,
    reactants: Vec<String>,
    products: Vec<String>,
}

enum EntropyType {
    Conditional,
    Marginal,
}

fn main() -> std::io::Result<()> {
    let mut config_file = File::open("configuration.toml")?;
    let mut contents = String::new();
    config_file.read_to_string(&mut contents)?;
    let conf: Config = toml::from_str(&contents)?;

    match fs::create_dir(&conf.output) {
        Ok(()) => {}
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {}
        other => return other,
    }
    let info_toml_path = &conf.output.join("info.toml");

    let worker_id = std::env::var("GILLESPIE_WORKER_ID")
        .ok()
        .map(|val| calculate_hash(&val));

    match fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(info_toml_path)
    {
        Ok(mut config_file_copy) => write!(config_file_copy, "{}", contents)?,
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
            // sleep a little so the other workers have time to write the info.toml
            std::thread::sleep(std::time::Duration::from_secs(1));

            let mut info_toml = File::open(info_toml_path)?;
            let mut contents = String::new();
            info_toml.read_to_string(&mut contents)?;
            let other_conf: Config = toml::from_str(&contents)?;

            if other_conf == conf {
                println!("configuration.toml matches existing info.toml");
            } else {
                panic!("{:?} does not match configuration.toml", info_toml_path);
            }

            if worker_id.is_none() {
                panic!("Please set GILLESPIE_WORKER_ID to run multiple jobs in parallel");
            }
        }
        Err(other) => return Err(other),
    }

    let sig_network = conf.signal.to_reaction_network();
    let res_network = conf.response.to_reaction_network();

    let length = conf.length;

    let traj_lengths = Array::linspace(0.0, 600.0, length);

    let ce_chunks = rayon::iter::repeatn((), conf.conditional_entropy.num_signals)
        .map(|_| {
            -conditional_likelihood(
                traj_lengths.as_slice().unwrap(),
                conf.conditional_entropy.responses_per_signal,
                conf.batch_size,
                length,
                &sig_network,
                &res_network,
            )
        })
        .chunks(conf.batch_size)
        .enumerate();

    let signals_pre = simulate(
        conf.marginal_entropy.num_signals,
        length,
        &[50.0],
        &sig_network,
        None,
    );

    let me_chunks = rayon::iter::repeatn((), conf.marginal_entropy.num_responses)
        .map(|_| {
            -marginal_likelihood(
                traj_lengths.as_slice().unwrap(),
                length,
                signals_pre.as_ref(),
                &sig_network,
                &res_network,
            )
        })
        .chunks(conf.batch_size)
        .enumerate();

    ce_chunks
        .map(|(chunk_num, log_lh)| (EntropyType::Conditional, chunk_num, log_lh))
        .chain(me_chunks.map(|(chunk_num, log_lh)| (EntropyType::Marginal, chunk_num, log_lh)))
        .for_each(|(entropy_type, chunk_num, log_lh)| {
            let mut array = Array2::zeros((conf.batch_size, traj_lengths.len()));
            for (mut row, ll) in array.outer_iter_mut().zip(log_lh.iter()) {
                row.assign(ll);
            }

            let worker_id = worker_id.as_ref().map(std::ops::Deref::deref).unwrap_or("");
            let filename = match entropy_type {
                EntropyType::Conditional => format!("ce{}-{:?}.npy", worker_id, chunk_num),
                EntropyType::Marginal => format!("me{}-{:?}.npy", worker_id, chunk_num),
            };

            if let Ok(out_file) = File::create(conf.output.join(&filename)) {
                array.write_npy(out_file).expect("could not write npy");
            } else {
                panic!("could not write chunk {:?}", &filename);
            }
        });

    Ok(())
}
