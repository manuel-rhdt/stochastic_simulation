use crate::gillespie::{ReactionNetwork, SimulationCoordinator};

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

use rand::{self, SeedableRng};
use rand_pcg::Pcg64Mcg;
use serde::{Deserialize, Serialize};
use tera::Tera;

pub fn calculate_hash<T: Hash + ?Sized>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    Uint(u32),
    Uints(Vec<u32>),
    Int(i32),
    Ints(Vec<i32>),
    Ulonglong(u64),
    Ulonglongs(Vec<u64>),
    Longlong(i64),
    Longlongs(Vec<i64>),
    Double(f64),
    Doubles(Vec<f64>),
    Datetime(toml::value::Datetime),
    Str(String),
}

impl From<Value> for netcdf::AttrValue {
    fn from(val: Value) -> Self {
        match val {
            Value::Uint(val) => val.into(),
            Value::Uints(val) => val.into(),
            Value::Int(val) => val.into(),
            Value::Ints(val) => val.into(),
            Value::Ulonglong(val) => val.into(),
            Value::Ulonglongs(val) => val.into(),
            Value::Longlong(val) => val.into(),
            Value::Longlongs(val) => val.into(),
            Value::Double(val) => val.into(),
            Value::Doubles(val) => val.into(),
            Value::Datetime(val) => val.to_string().into(),
            Value::Str(val) => val.into(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Config {
    pub output: PathBuf,
    pub length: f64,
    pub num_trajectory_lengths: usize,
    pub p0_samples: Option<usize>,
    pub conditional_entropy: Option<ConfigConditionalEntropy>,
    pub marginal_entropy: Option<ConfigMarginalEntropy>,
    pub signal: ConfigReactionNetwork,
    pub response: ConfigReactionNetwork,
    #[serde(default)]
    pub attributes: BTreeMap<String, Value>,
}

impl Config {
    pub fn hash_relevant<H: Hasher>(&self, hasher: &mut H) {
        self.conditional_entropy.hash(hasher);
        self.marginal_entropy.hash(hasher);
        self.length.to_bits().hash(hasher);
        self.p0_samples.hash(hasher);
        self.num_trajectory_lengths.hash(hasher);
        self.signal.hash(hasher);
        self.response.hash(hasher);
    }

    pub fn get_relevant_hash(&self) -> u64 {
        let mut s = DefaultHasher::new();
        self.hash_relevant(&mut s);
        s.finish()
    }

    pub fn create_coordinator(&self, seed: u64) -> SimulationCoordinator<Pcg64Mcg> {
        SimulationCoordinator {
            trajectory_len: self.length,
            equilibration_time: self.length,

            signal_mean: self.signal.mean,
            response_mean: self.response.mean,

            p0_samples: self.p0_samples.unwrap_or(1000),

            sig_network: self.signal.to_reaction_network(),
            res_network: self.response.to_reaction_network(),

            rng: Pcg64Mcg::seed_from_u64(seed),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ConfigConditionalEntropy {
    pub num_signals: usize,
    pub responses_per_signal: usize,
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ConfigMarginalEntropy {
    pub num_signals: usize,
    pub num_responses: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ConfigReactionNetwork {
    pub initial: f64,
    pub mean: f64,
    pub components: Vec<String>,
    pub reactions: Vec<Reaction>,
}

impl PartialEq for ConfigReactionNetwork {
    fn eq(&self, right: &ConfigReactionNetwork) -> bool {
        self.initial.to_bits() == right.initial.to_bits()
            && self.mean.to_bits() == right.mean.to_bits()
            && self.components == right.components
            && self.reactions == right.reactions
    }
}

impl Hash for ConfigReactionNetwork {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.initial.to_bits().hash(state);
        self.mean.to_bits().hash(state);
        self.components.hash(state);
        self.reactions.hash(state);
    }
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
            let r = reaction
                .reactants
                .iter()
                .map(|reactant| name_table[reactant] as u32)
                .collect();
            reactants.push(r);
            let p = reaction
                .products
                .iter()
                .map(|product| name_table[product] as u32)
                .collect();
            products.push(p);
        }

        ReactionNetwork {
            k,
            reactants,
            products,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Reaction {
    pub k: f64,
    pub reactants: Vec<String>,
    pub products: Vec<String>,
}

impl PartialEq for Reaction {
    fn eq(&self, right: &Reaction) -> bool {
        self.k.to_bits() == right.k.to_bits()
            && self.reactants == right.reactants
            && self.products == right.products
    }
}

impl Hash for Reaction {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.k.to_bits().hash(state);
        self.reactants.hash(state);
        self.products.hash(state);
    }
}

pub fn create_dir_if_not_exists<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<()> {
    match std::fs::create_dir(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        other => other,
    }
}

pub fn parse_configuration(
    path: impl AsRef<std::path::Path>,
) -> Result<Config, Box<dyn std::error::Error>> {
    use std::io::Read;

    let mut config_file = std::fs::File::open(path)?;
    let mut contents = String::new();
    config_file.read_to_string(&mut contents)?;

    let mut context = tera::Context::new();
    for (key, value) in std::env::vars() {
        context.insert(key, &value);
    }
    let contents = Tera::one_off(&contents, &context, false)?;

    Ok(toml::from_str(&contents)?)
}
