mod configuration;
mod gillespie;
mod kde;
mod likelihood;

use std::env;
use std::fs::{File, OpenOptions, Permissions};
use std::io::prelude::*;
use std::os::unix::fs::PermissionsExt;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Mutex,
};

use configuration::{calculate_hash, create_dir_if_not_exists, parse_configuration, Config};
use gillespie::{SimulationCoordinator, Trajectory, TrajectoryIterator};
use likelihood::log_likelihood;

use ndarray::{array, Array, Array1, Array2, Axis};
use netcdf;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use rayon;
use rayon::prelude::*;
use serde::Serialize;
use toml;

use jemalloc_ctl::{epoch, stats};

#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

static MEMORY_LIMIT: AtomicU64 = AtomicU64::new(0);
static UNCAUGHT_SIGNAL: AtomicBool = AtomicBool::new(false);

fn check_oom() {
    let limit = MEMORY_LIMIT.load(Ordering::Relaxed);
    if limit != 0 {
        epoch::advance().unwrap();
        let allocated = stats::allocated::read().unwrap() as u64;
        log::debug!("allocated memory: {}", allocated);
        if allocated > limit {
            log::error!("used too much memory: {} > {}", allocated, limit);
            UNCAUGHT_SIGNAL.store(true, Ordering::SeqCst);
        }
    }
}

macro_rules! check_abort_signal {
    () => {
        check_abort_signal!(Err(
            std::io::Error::from(std::io::ErrorKind::Interrupted).into()
        ));
    };

    ($default:expr) => {
        check_oom();
        if UNCAUGHT_SIGNAL.load(Ordering::SeqCst) {
            return $default;
        }
    };
}

fn conditional_likelihood(
    traj_lengths: &[f64],
    num_res: usize,
    coordinator: &mut SimulationCoordinator<impl rand::Rng>,
) -> (Array1<f64>, Array1<f64>) {
    let res_network = coordinator.res_network.clone();
    let sig = coordinator.generate_signal().collect();

    let kde = coordinator.equilibrate_respones_dist(sig.iter().components());

    let mut result = Array2::zeros((num_res, traj_lengths.len()));
    for (i, mut out) in result.outer_iter_mut().enumerate() {
        if i % 100 == 0 {
            check_abort_signal!((array![], array![]));
        }
        let res = coordinator.generate_response(sig.iter());
        let log_p0 = kde.pdf(res.components()[0]).ln();
        for (ll, out) in log_likelihood(traj_lengths, sig.iter(), res, &res_network).zip(&mut out) {
            *out = log_p0 + ll;
        }
    }
    (
        result.mean_axis(Axis(0)).unwrap(),
        result.std_axis(Axis(0), 1.0) / (num_res as f64).sqrt(),
    )
}

type TrajectoryCollected = Trajectory<Vec<f64>, Vec<f64>, Vec<u32>>;

fn marginal_likelihood(
    traj_lengths: &[f64],
    signals_pre: &[(TrajectoryCollected, kde::NormalKernelDensityEstimate)],
    coordinator: &mut SimulationCoordinator<impl rand::Rng>,
) -> (Array1<f64>, Array1<f64>) {
    let sig = coordinator.generate_signal().collect();
    let res = coordinator.generate_response(sig.iter()).collect();

    let mut likelihoods = Array2::zeros((signals_pre.len(), traj_lengths.len()));
    for (i, ((sig, kde), ref mut out)) in signals_pre
        .iter()
        .zip(likelihoods.outer_iter_mut())
        .enumerate()
    {
        if i % 100 == 0 {
            check_abort_signal!((array![], array![]));
        }
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
    }

    let max_elements = likelihoods.fold_axis(Axis(0), std::f64::NAN, |a, &b| a.max(b));
    likelihoods -= &max_elements;
    likelihoods.mapv_inplace(|x| x.exp());

    let num_signals = signals_pre.len() as f64;

    let mean_likelihood = likelihoods
        .mean_axis(Axis(0))
        .unwrap()
        .mapv_into(|x| x.ln())
        + &max_elements;
    let std_err_likelihood = likelihoods
        .std_axis(Axis(0), 1.0)
        .mapv_into(|x| (x / num_signals.sqrt()).ln())
        + &max_elements;
    // this line is needed to account for error propagation through the logarithm.
    let std_err_likelihood = (std_err_likelihood - &mean_likelihood).mapv_into(|x| x.exp());

    (mean_likelihood, std_err_likelihood)
}

#[derive(Copy, Debug, Clone)]
enum EntropyType {
    Conditional,
    Marginal,
}

fn print_help() -> ! {
    println!(
        "\
Usage:
    gillespie [conf]

Arguments:
    conf - The path to the configuration file. The default is 
           'configuration.toml'.
"
    );
    std::process::exit(0)
}

#[derive(Debug, Clone)]
enum Error {
    NetCdfError(String),
    InterruptSignal,
}

use std::fmt;
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NetCdfError(val) => write!(f, "{}", val),
            Error::InterruptSignal => write!(f, "Process was interrupted by signal."),
        }
    }
}

impl From<netcdf::error::Error> for Error {
    fn from(nerror: netcdf::error::Error) -> Self {
        Error::NetCdfError(nerror.to_string())
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

fn calculate_entropy(
    entropy_type: EntropyType,
    output_file: &Mutex<netcdf::MutableFile>,
    conf: Config,
    traj_lengths: &[f64],
    seed: u64,
) -> Result<(), Error> {
    let dimension_name = match entropy_type {
        EntropyType::Conditional => "ce",
        EntropyType::Marginal => "me",
    };
    let name = match entropy_type {
        EntropyType::Conditional => "conditional_entropy",
        EntropyType::Marginal => "marginal_entropy",
    };

    let mut file = output_file.lock().unwrap();

    match entropy_type {
        EntropyType::Conditional => file.add_dimension(
            dimension_name,
            conf.conditional_entropy.unwrap().num_signals,
        )?,
        EntropyType::Marginal => {
            file.add_dimension(dimension_name, conf.marginal_entropy.unwrap().num_responses)?
        }
    };

    let mut var = file.add_variable::<f64>(name, &[dimension_name, "time"])?;
    var.set_fill_value(std::f64::NAN)?;
    var.add_attribute("units", "nats")?;

    match entropy_type {
        EntropyType::Conditional => {
            var.add_attribute(
                "responses_per_signal",
                conf.conditional_entropy.unwrap().responses_per_signal as u32,
            )?;
        }
        EntropyType::Marginal => {
            var.add_attribute(
                "signals_per_response",
                conf.marginal_entropy.unwrap().num_signals as u32,
            )?;
        }
    }

    let err_name = &format!("{}_err", name);
    let mut err = file.add_variable::<f64>(err_name, &[dimension_name, "time"])?;
    err.set_fill_value(std::f64::NAN)?;
    let timing_name = &format!("{}_timing", name);
    let mut timing = file.add_variable::<f64>(timing_name, &[dimension_name])?;
    timing.add_attribute("unit", "s")?;
    timing.set_fill_value(std::f64::NAN)?;

    // unlock mutex
    std::mem::drop(file);

    let signals_pre = if let EntropyType::Marginal = entropy_type {
        let coordinator = conf.create_coordinator(seed);
        (0..conf.marginal_entropy.unwrap().num_signals)
            .into_par_iter()
            .map_with(coordinator, |coordinator, i| {
                check_abort_signal!(Err(Error::InterruptSignal));

                coordinator.rng = Pcg64Mcg::seed_from_u64(seed ^ i as u64);
                let sig = coordinator.generate_signal().collect();
                let kde = coordinator.equilibrate_respones_dist(sig.iter().components());
                Ok((sig, kde))
            })
            .collect::<Result<Vec<_>, _>>()?
            .into()
    } else {
        None
    };
    let signals_pre = signals_pre.unwrap_or_default();

    let num_samples = match entropy_type {
        EntropyType::Conditional => conf.conditional_entropy.unwrap().num_signals,
        EntropyType::Marginal => conf.marginal_entropy.unwrap().num_responses,
    };

    (0..num_samples)
        .into_par_iter()
        .try_for_each(|row| -> Result<(), Error> {
            let seed = row as u64 ^ seed ^ 0xabcd_abcd;
            let mut coordinator = conf.create_coordinator(seed);
            let before = std::time::Instant::now();

            let (log_lh, log_lh_err) = match entropy_type {
                EntropyType::Conditional => conditional_likelihood(
                    traj_lengths,
                    conf.conditional_entropy.unwrap().responses_per_signal,
                    &mut coordinator,
                ),
                EntropyType::Marginal => {
                    marginal_likelihood(traj_lengths, &signals_pre, &mut coordinator)
                }
            };
            check_abort_signal!(Err(Error::InterruptSignal));

            let time = (std::time::Instant::now() - before).as_secs_f64();

            log::info!("{:?}: Finished row {} in {:.2} s", entropy_type, row, time);

            let mut file = output_file.lock().unwrap();
            let mut var = file.variable_mut(name).expect("could not find variable");
            var.put_values(
                log_lh.as_slice().unwrap(),
                Some(&[row, 0]),
                Some(&[1, log_lh.len()]),
            )?;
            let mut var = file
                .variable_mut(err_name)
                .expect("could not find variable");
            var.put_values(
                log_lh_err.as_slice().unwrap(),
                Some(&[row, 0]),
                Some(&[1, log_lh.len()]),
            )?;
            let mut var = file
                .variable_mut(timing_name)
                .expect("could not find variable");
            var.put_value(time, Some(&[row]))?;
            Ok(())
        })?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    ctrlc::set_handler(move || {
        UNCAUGHT_SIGNAL.store(true, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    env_logger::init();

    let args: Vec<String> = env::args().collect();
    let configuration_filename = match args.len() {
        // no arguments passed
        1 => "configuration.toml",
        // one argument passed
        2 => &args[1],
        _ => print_help(),
    };
    let conf = parse_configuration(configuration_filename)?;

    create_dir_if_not_exists(&conf.output)?;

    let worker_name = env::var("GILLESPIE_WORKER_ID").ok();

    let memory_limit: Option<u64> = env::var("GILLESPIE_MEMORY_LIMIT")
        .ok()
        .and_then(|val| val.parse().ok());
    if let Some(val) = memory_limit {
        log::info!("Setting memory limit to {} bytes", val);
        MEMORY_LIMIT.store(val, Ordering::SeqCst);
    }

    let mut worker_info = WorkerInfo {
        hostname: env::var("HOSTNAME").ok(),
        worker_id: worker_name.clone(),
        start_time: chrono::Local::now()
            .to_rfc3339()
            .parse()
            .expect("could not parse current time"),
        end_time: None,
        error: None,
        version: VersionInfo {
            build_time: env!("VERGEN_BUILD_TIMESTAMP").parse().ok(),
            commit_sha: env!("VERGEN_SHA"),
            commit_date: env!("VERGEN_COMMIT_DATE").parse().ok(),
            version: env!("VERGEN_SEMVER"),
        },
        configuration: conf.clone(),
    };
    write!(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(conf.output.join("worker.toml"))?,
        "{}",
        toml::to_string_pretty(&worker_info)?
    )?;

    let traj_lengths = Array::linspace(0.0, conf.length, conf.num_trajectory_lengths);

    let mut output_file = netcdf::create(conf.output.join("data.nc"))?;
    for (key, val) in &conf.attributes {
        output_file.add_attribute(key, val.clone())?;
    }
    output_file.add_dimension("time", conf.num_trajectory_lengths)?;
    output_file.add_attribute(
        "source",
        format!(
            "`accelerate` (version {}, commit {}) gillespie simulator",
            env!("VERGEN_SEMVER"),
            env!("VERGEN_SHA")
        ),
    )?;
    output_file.add_attribute("date_modified", worker_info.start_time.to_string())?;
    output_file.add_attribute("p0_samples", conf.p0_samples.unwrap_or(1000) as u32)?;
    let mut time_var = output_file.add_variable::<f64>("time", &["time"])?;
    time_var.put_values(traj_lengths.as_slice().unwrap(), None, None)?;

    let output_file = Mutex::new(output_file);

    let seed_base = conf.get_relevant_hash() ^ calculate_hash(&worker_name);

    let result = conf
        .conditional_entropy
        .map(|_| EntropyType::Conditional)
        .into_par_iter()
        .chain(conf.marginal_entropy.map(|_| EntropyType::Marginal))
        .try_for_each(|entropy_type| {
            calculate_entropy(
                entropy_type,
                &output_file,
                conf.clone(),
                traj_lengths.as_slice().unwrap(),
                seed_base ^ 0xabcd_abcd ^ entropy_type as u64,
            )
        });

    if let Err(error) = result {
        log::error!("Abort calculation: {}", error);
        worker_info.error = Some(format!("{}", error))
    }

    worker_info.end_time = Some(
        chrono::Local::now()
            .to_rfc3339()
            .parse()
            .expect("could not parse current time"),
    );
    let mut worker_toml = File::create(conf.output.join("worker.toml"))?;
    write!(worker_toml, "{}", toml::to_string_pretty(&worker_info)?)?;
    worker_toml.set_permissions(Permissions::from_mode(0o444))?;

    Ok(())
}

#[derive(Debug, Clone, Serialize)]
struct WorkerInfo {
    hostname: Option<String>,
    worker_id: Option<String>,
    start_time: toml::value::Datetime,
    end_time: Option<toml::value::Datetime>,
    error: Option<String>,
    version: VersionInfo,
    configuration: Config,
}

#[derive(Debug, Clone, Serialize)]
struct VersionInfo {
    commit_sha: &'static str,
    commit_date: Option<toml::value::Datetime>,
    version: &'static str,
    build_time: Option<toml::value::Datetime>,
}
