use accelerate::likelihood::log_likelihood;
use accelerate::{
    configuration,
    gillespie::{SimulationCoordinator, TrajectoryIterator},
};

use std::io::Write;

use ndarray::{Array, Array1, Array2};
use serde::Serialize;
use serde_json;

use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

use rayon::prelude::*;

fn marginal_likelihood(
    traj_lengths: &[f64],
    num_signals: usize,
    coordinator: &mut SimulationCoordinator<Pcg64Mcg>,
) -> (Array1<f64>, Array2<f64>) {
    let sig = coordinator.generate_signal().collect();
    let res = coordinator.generate_response(sig.iter()).collect();

    let single_likelihood = {
        let kde = coordinator.equilibrate_respones_dist(sig.iter().components());
        let logp = kde.pdf(res.iter().components()[0]).ln();
        log_likelihood(
            traj_lengths,
            sig.iter(),
            res.iter(),
            &coordinator.res_network,
        )
        .map(|x| x + logp)
        .collect()
    };

    let mut likelihoods = Array2::zeros((num_signals, traj_lengths.len()));

    let signals: Vec<_> = (0..num_signals)
        .map(|_| coordinator.generate_signal().collect())
        .collect();
    signals
        .into_par_iter()
        .zip(likelihoods.outer_iter_mut())
        .for_each_with(coordinator.clone(), |coordinator, (sig, out)| {
            coordinator.rng = Pcg64Mcg::from_entropy();
            let kde = coordinator.equilibrate_respones_dist(sig.iter().components());
            let logp = kde.pdf(res.iter().components()[0]).ln();
            for (ll, out) in log_likelihood(
                traj_lengths,
                sig.iter(),
                res.iter(),
                &coordinator.res_network,
            )
            .zip(out)
            {
                *out = logp + ll;
            }
        });

    (single_likelihood, likelihoods)
}

#[derive(Clone, Serialize)]
struct Output {
    single: Vec<f64>,
    marginal: Vec<Vec<f64>>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let conf = configuration::parse_configuration("benches/configuration.toml")?;

    let traj_lengths = Array::linspace(0.0, conf.length, conf.num_trajectory_lengths);

    let mut coord = conf.create_coordinator(1234);

    let (cond, marg) = marginal_likelihood(traj_lengths.as_slice().unwrap(), 10000, &mut coord);

    let outp = Output {
        single: cond.to_vec(),
        marginal: marg.outer_iter().map(|x| x.to_vec()).collect(),
    };

    let outp = serde_json::to_string(&outp)?;

    let mut file = std::fs::File::create("histogram.json")?;
    write!(file, "{}", outp)?;

    Ok(())
}
