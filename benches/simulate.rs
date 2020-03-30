use accelerate::configuration::parse_configuration;
use accelerate::gillespie::TrajectoryIterator;
use accelerate::likelihood::log_likelihood;

use criterion::{criterion_group, criterion_main, Criterion};

fn bench_simulate(c: &mut Criterion) {
    let conf = parse_configuration("benches/configuration.toml").unwrap();
    let mut coordinator = conf.create_coordinator(1);
    let sig = &coordinator.generate_signal().collect();
    c.bench_function("simulate_signal", |b| {
        b.iter(|| coordinator.generate_signal().exhaust())
    });

    c.bench_function("simulate_response", |b| {
        b.iter(|| coordinator.generate_response(sig.iter()).exhaust())
    });

    let res = coordinator.generate_response(sig.iter()).collect();
    let traj_lengths: Vec<_> = std::iter::repeat(1.0)
        .take(800)
        .scan(0.0, |s, x| {
            *s += x;
            Some(*s)
        })
        .collect();
    c.bench_function("log_likelihood", |b| {
        b.iter(|| {
            log_likelihood(
                &traj_lengths,
                sig.iter(),
                res.iter(),
                &coordinator.res_network,
            ).last()
        })
    });
}

criterion_group!(benches, bench_simulate);
criterion_main!(benches);
