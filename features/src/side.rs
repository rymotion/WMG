//! Side features: three hand-crafted "smeared top end" numbers per detector window, appended
//! to the model embedding so the on-device personal head can weigh them. They describe the
//! 8-16 kHz band of a 32 kHz window and are the ones that kept some signal through a phone
//! mic in `wmg_train.handcrafted` (flux 0.68, modulation peak 0.62, flatness 0.59 AUC).
//!
//! The definitions match `wmg_train/handcrafted.py` (`extract`) bin for bin, so the Python
//! reference and this code agree on the raw values; see `side_features_raw`.

use std::f64::consts::PI;
use std::sync::OnceLock;

use realfft::{RealFftPlanner, RealToComplex};

use crate::SPEC;

pub const N_SIDE: usize = 3;
pub const SIDE_FEATURES: [&str; N_SIDE] = ["flux_p2m", "mod_peak_db", "sfm_mean"];
/// `(offset, scale)` per feature: the raw value is mapped to `(v - offset) / scale` so the
/// head sees roughly unit-scale inputs. Taken from the phone-capture medians and spreads.
pub const SIDE_SCALING: [(f64, f64); N_SIDE] = [(4.2, 2.0), (6.0, 2.5), (0.18, 0.08)];

const N_FFT: usize = 1024;
const HOP: usize = 128;
const BAND_HZ: (f64, f64) = (8_000.0, 16_000.0);
const MOD_BAND_HZ: (f64, f64) = (30.0, 120.0);
const WELCH_SECONDS: f64 = 4.0;

fn fft() -> &'static dyn RealToComplex<f64> {
    static F: OnceLock<std::sync::Arc<dyn RealToComplex<f64>>> = OnceLock::new();
    F.get_or_init(|| RealFftPlanner::<f64>::new().plan_fft_forward(N_FFT)).as_ref()
}

fn hann_periodic(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos()).collect()
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 { s[n / 2] } else { 0.5 * (s[n / 2 - 1] + s[n / 2]) }
}

/// numpy's default (linear) percentile.
fn percentile(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n == 0 {
        return f64::NAN;
    }
    let pos = q / 100.0 * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    s[lo] + (s[hi] - s[lo]) * (pos - lo as f64)
}

fn db(p: f64) -> f64 {
    10.0 * p.max(1e-20).log10()
}

/// Power of the 8-16 kHz bins per frame, `[frame][bin]`, from a 1024/128 Hann STFT at 32 kHz.
fn band_power(x: &[f64]) -> Vec<Vec<f64>> {
    let sr = SPEC.sample_rate as f64;
    let lo = (BAND_HZ.0 * N_FFT as f64 / sr).ceil() as usize;
    let hi = ((BAND_HZ.1.min(sr / 2.0)) * N_FFT as f64 / sr).ceil() as usize; // exclusive, f < hi
    let fft = fft();
    let win = hann_periodic(N_FFT);
    let mut padded;
    let x = if x.len() < N_FFT {
        padded = x.to_vec();
        padded.resize(N_FFT, 0.0);
        padded.as_slice()
    } else {
        x
    };
    let n_frames = (x.len() - N_FFT) / HOP + 1;
    let mut input = fft.make_input_vec();
    let mut spec = fft.make_output_vec();
    let mut out = Vec::with_capacity(n_frames);
    for n in 0..n_frames {
        let frame = &x[n * HOP..n * HOP + N_FFT];
        // The reference computes the window product in f32.
        for i in 0..N_FFT {
            input[i] = (frame[i] as f32 * win[i] as f32) as f64;
        }
        if fft.process(&mut input, &mut spec).is_err() {
            return Vec::new();
        }
        out.push(spec[lo..hi].iter().map(|c| c.re * c.re + c.im * c.im).collect());
    }
    out
}

/// Peak within +-1 Hz of `centre` over the median of +-8 Hz around it (excluding the peak), dB.
fn peakiness_db(m: &[f64], f: &[f64], centre: f64) -> f64 {
    let mut near = f64::MIN;
    let mut around = Vec::new();
    for (i, &fi) in f.iter().enumerate() {
        let d = (fi - centre).abs();
        if d <= 1.0 {
            near = near.max(m[i]);
        } else if d <= 8.0 {
            around.push(m[i]);
        }
    }
    if near == f64::MIN || around.len() < 4 {
        return f64::NAN;
    }
    db(near) - db(median(&around))
}

/// Raw `[flux_p2m, mod_peak_db, sfm_mean]` for one 32 kHz mono window; NaN where undefined.
pub fn side_features_raw(x: &[f64]) -> [f64; N_SIDE] {
    let nan = [f64::NAN; N_SIDE];
    let s = band_power(x);
    if s.len() < 16 || s.first().map_or(true, |r| r.len() < 4) {
        return nan;
    }
    let sums: Vec<f64> = s.iter().map(|r| r.iter().sum()).collect();
    let max_sum = sums.iter().cloned().fold(f64::MIN, f64::max);
    if max_sum <= 0.0 {
        return nan; // silence: nothing in the band to describe
    }
    let gate = 10f64.powf((db(max_sum) - 60.0) / 10.0);
    let mut sfm_sum = 0.0;
    let mut n_active = 0usize;
    for (r, &sum) in s.iter().zip(&sums) {
        if sum > gate {
            let k = r.len() as f64;
            let log_mean = r.iter().map(|&p| p.max(1e-20).ln()).sum::<f64>() / k;
            let mean = r.iter().map(|&p| p.max(1e-20)).sum::<f64>() / k;
            sfm_sum += log_mean.exp() / mean;
            n_active += 1;
        }
    }
    let sfm_mean = if n_active > 0 { sfm_sum / n_active as f64 } else { f64::NAN };

    let flux: Vec<f64> = (1..s.len())
        .map(|n| {
            s[n].iter()
                .zip(&s[n - 1])
                .map(|(&a, &b)| (a.max(1e-20).ln() - b.max(1e-20).ln()).max(0.0).powi(2))
                .sum()
        })
        .collect();
    let med = median(&flux);
    let flux_p2m = if med > 0.0 { percentile(&flux, 99.0) / med } else { f64::NAN };

    // Modulation spectrum of the band envelope: Welch over 4 s pieces, 50% overlap, Hann.
    let fr = SPEC.sample_rate as f64 / HOP as f64;
    let env_mean = sums.iter().sum::<f64>() / sums.len() as f64;
    let env: Vec<f64> = sums.iter().map(|v| v / (env_mean + 1e-20)).collect();
    let seg = (WELCH_SECONDS * fr) as usize;
    let mod_peak_db = if env.len() >= seg {
        let m = welch(&env, seg);
        let f: Vec<f64> = (0..m.len()).map(|i| i as f64 * fr / seg as f64).collect();
        f.iter()
            .enumerate()
            .filter(|&(_, &fi)| fi >= MOD_BAND_HZ.0 && fi <= MOD_BAND_HZ.1)
            .map(|(_, &c)| peakiness_db(&m, &f, c))
            .filter(|v| !v.is_nan())
            .fold(f64::NAN, f64::max)
    } else {
        f64::NAN
    };
    [flux_p2m, mod_peak_db, sfm_mean]
}

/// Averaged |rfft|^2 of mean-removed, Hann-windowed `seg`-long pieces at 50% overlap
/// (scipy.signal.welch up to a constant factor, which the peakiness ratio cancels).
fn welch(x: &[f64], seg: usize) -> Vec<f64> {
    let step = seg / 2;
    let n_seg = (x.len() - seg) / step + 1;
    let win = hann_periodic(seg);
    let fft = RealFftPlanner::<f64>::new().plan_fft_forward(seg);
    let mut input = fft.make_input_vec();
    let mut spec = fft.make_output_vec();
    let mut acc = vec![0.0; seg / 2 + 1];
    for k in 0..n_seg {
        let piece = &x[k * step..k * step + seg];
        let mean = piece.iter().sum::<f64>() / seg as f64;
        for i in 0..seg {
            input[i] = (piece[i] - mean) * win[i];
        }
        if fft.process(&mut input, &mut spec).is_err() {
            return acc;
        }
        for (a, c) in acc.iter_mut().zip(&spec) {
            *a += c.re * c.re + c.im * c.im;
        }
    }
    for a in &mut acc {
        *a /= n_seg as f64;
    }
    acc
}

/// Standardised side features for one window: `(raw - offset) / scale`, NaN mapped to 0
/// (the head's starting weight for them is 0, so an undefined value changes nothing).
pub fn side_features(x: &[f64]) -> [f32; N_SIDE] {
    let raw = side_features_raw(x);
    let mut out = [0f32; N_SIDE];
    for i in 0..N_SIDE {
        let (off, scale) = SIDE_SCALING[i];
        let v = (raw[i] - off) / scale;
        out[i] = if v.is_finite() { v as f32 } else { 0.0 };
    }
    out
}
