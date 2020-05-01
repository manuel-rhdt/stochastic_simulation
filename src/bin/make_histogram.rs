use accelerate::likelihood::log_likelihood;
use accelerate::{
    configuration,
    gillespie::{SimulationCoordinator, TrajectoryIterator},
};

use std::io::Write;

use serde::Serialize;
use serde_json;

use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

use rayon::prelude::*;

fn marginal_likelihood(
    num_signals: usize,
    coordinator: &mut SimulationCoordinator<Pcg64Mcg>,
) -> (Vec<(f64, f64)>, Vec<Vec<(f64, f64)>>) {
    let sig = coordinator.generate_signal().collect();
    let res = coordinator.generate_response(sig.iter()).collect();

    let single_likelihood = {
        let kde = coordinator.equilibrate_respones_dist(sig.iter().components());
        let logp = kde.pdf(res.iter().components()[0]).ln();
        log_likelihood(sig.iter(), res.iter(), &coordinator.res_network)
            .map(|(t, x)| (t, x + logp))
            .collect()
    };

    let signals: Vec<_> = (0..num_signals)
        .map(|_| coordinator.generate_signal().collect())
        .collect();

    let mut likelihoods = vec![];
    signals
        .into_par_iter()
        .map_with(coordinator.clone(), |coordinator, sig| {
            coordinator.rng = Pcg64Mcg::from_entropy();
            let kde = coordinator.equilibrate_respones_dist(sig.iter().components());
            let logp = kde.pdf(res.iter().components()[0]).ln();
            log_likelihood(sig.iter(), res.iter(), &coordinator.res_network)
                .map(|(t, x)| (t, x + logp))
                .collect()
        })
        .collect_into_vec(&mut likelihoods);

    (single_likelihood, likelihoods)
}

#[derive(Debug, Clone, Serialize)]
struct Output {
    single: Vec<f64>,
    marginal: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct Record {
    duration: f64,
    log_likelihood: f64,
    sampled_from: &'static str,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let conf = configuration::parse_configuration("configuration.toml")?;

    let mut coord = conf.create_coordinator(1234);

    let (cond, marg) = marginal_likelihood(10000, &mut coord);

    let mut outp = vec![];
    for &(duration, val) in cond.iter() {
        outp.push(Record {
            duration,
            log_likelihood: val,
            sampled_from: "posterior",
        })
    }

    for traj in marg.iter() {
        for &(duration, val) in traj.iter() {
            outp.push(Record {
                duration,
                log_likelihood: val,
                sampled_from: "prior",
            })
        }
    }

    let outp = serde_json::to_string(&outp)?;

    let mut file = std::fs::File::create("histogram.json")?;
    write!(file, "{}", outp)?;

    Ok(())
}
