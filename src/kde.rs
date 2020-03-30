use ndarray::{array, Array1, Axis};
use ndarray_stats::{interpolate::Linear, Quantile1dExt};
use noisy_float::types::{n64, N64};
use std::f64::consts::PI;

// from https://github.com/JuliaStats/KernelDensity.jl/blob/master/src/univariate.jl
fn default_bandwidth(data: &[f64], alpha: f64) -> f64 {
    let ndata = data.len();
    let mut data: Array1<N64> = data.iter().map(|&x| n64(x)).collect();

    // Calculate width using variance and IQR
    let var_width = data.std_axis(Axis(0), n64(1.0)).into_scalar();
    let q25_q75 = data
        .quantiles_mut(&array![n64(0.25), n64(0.75)], &Linear)
        .unwrap();
    let quantile_width = (q25_q75[1] - q25_q75[0]) / 1.34;

    // Deal with edge cases with 0 IQR or variance
    let mut width = var_width.min(quantile_width);
    if width == 0.0 {
        if var_width == 0.0 {
            width = n64(1.0)
        } else {
            width = var_width
        }
    }

    // Set bandwidth using Silverman's rule of thumb
    (n64(alpha) * width * (ndata as f64).powf(-0.2)).into()
}

pub struct NormalKernelDensityEstimate {
    pub data: Vec<f64>,
    pub bandwidth: f64,
}

impl NormalKernelDensityEstimate {
    pub fn new(data: Vec<f64>) -> Self {
        let bandwidth = default_bandwidth(&data, 0.9);
        NormalKernelDensityEstimate { data, bandwidth }
    }

    pub fn pdf(&self, at: f64) -> f64 {
        let length = self.data.len();

        let sum: f64 = self
            .data
            .iter()
            .map(|&sample| {
                let rescaled: f64 = (at - sample) / self.bandwidth;
                (-0.5 * rescaled.powi(2)).exp()
            })
            .sum();

        let sqrt_2pi = (2.0 * PI).sqrt();
        sum / (sqrt_2pi * length as f64 * self.bandwidth)
    }
}
