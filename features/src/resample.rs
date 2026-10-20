//! Deterministic offline resampler (Kaiser-windowed sinc, rational polyphase).
//!
//! Written in-crate rather than pulled from a dependency so training, the server and devices
//! produce bit-identical input for the same audio regardless of library versions.

use std::f64::consts::PI;

/// Zero crossings of the sinc kernel on each side of the centre tap.
const HALF_ZEROS: usize = 32;
/// Kaiser window β (~90 dB stopband).
const BETA: f64 = 9.0;
/// Cutoff as a fraction of the lower Nyquist, leaving room for the transition band.
const ROLLOFF: f64 = 0.95;
/// Above this many polyphase phases the kernel is evaluated on the fly instead of tabled.
const MAX_TABLE_PHASES: usize = 4096;

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Zeroth-order modified Bessel function of the first kind (series expansion).
fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term, q) = (1.0, 1.0, x * x / 4.0);
    for k in 1..50 {
        term *= q / (k * k) as f64;
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

struct Kernel {
    /// Kernel half-width in input samples.
    half: f64,
    /// Normalised cutoff relative to the input rate (cycles per input sample).
    cutoff: f64,
    i0_beta: f64,
}

impl Kernel {
    fn eval(&self, t: f64) -> f64 {
        if t.abs() >= self.half {
            return 0.0;
        }
        let x = 2.0 * self.cutoff * t;
        let sinc = if x.abs() < 1e-12 {
            1.0
        } else {
            (PI * x).sin() / (PI * x)
        };
        let r = t / self.half;
        2.0 * self.cutoff * sinc * bessel_i0(BETA * (1.0 - r * r).max(0.0).sqrt()) / self.i0_beta
    }
}

/// Resample `input` from `sr_in` to `sr_out`. Output length is `round(len · sr_out / sr_in)`,
/// with no group delay (the kernel is centred on each output instant).
pub fn resample(input: &[f64], sr_in: u32, sr_out: u32) -> Vec<f64> {
    if sr_in == sr_out || input.is_empty() {
        return input.to_vec();
    }
    let g = gcd(sr_in as u64, sr_out as u64);
    let up = (sr_out as u64 / g) as usize; // output steps per `down` input samples
    let down = (sr_in as u64 / g) as usize;
    let ratio = sr_out as f64 / sr_in as f64;
    let cutoff = 0.5 * ratio.min(1.0) * ROLLOFF;
    let kernel = Kernel {
        half: HALF_ZEROS as f64 / (2.0 * cutoff),
        cutoff,
        i0_beta: bessel_i0(BETA),
    };
    let taps = 2 * kernel.half.ceil() as usize + 1;

    // Output n sits at input position n·down/up = base + phase/up.
    let table: Option<Vec<Vec<f64>>> = (up <= MAX_TABLE_PHASES).then(|| {
        (0..up)
            .map(|phase| {
                let frac = phase as f64 / up as f64;
                let first = -(kernel.half.ceil() as isize);
                (0..taps)
                    .map(|k| kernel.eval((first + k as isize) as f64 - frac))
                    .collect()
            })
            .collect()
    });

    let out_len = ((input.len() as f64) * ratio).round() as usize;
    let first = -(kernel.half.ceil() as isize);
    let mut out = Vec::with_capacity(out_len);
    for n in 0..out_len {
        let pos = n * down;
        let base = (pos / up) as isize;
        let phase = pos % up;
        let frac = phase as f64 / up as f64;
        let mut acc = 0.0;
        for k in 0..taps {
            let idx = base + first + k as isize;
            if idx < 0 || idx as usize >= input.len() {
                continue;
            }
            let w = match &table {
                Some(t) => t[phase][k],
                None => kernel.eval((first + k as isize) as f64 - frac),
            };
            acc += w * input[idx as usize];
        }
        out.push(acc);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(f: f64, sr: u32, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * f * i as f64 / sr as f64).sin())
            .collect()
    }

    #[test]
    fn identity_when_rates_match() {
        let x = sine(440.0, 32_000, 1000);
        assert_eq!(resample(&x, 32_000, 32_000), x);
    }

    #[test]
    fn preserves_in_band_sine() {
        for (sr_in, sr_out) in [(44_100, 32_000), (16_000, 32_000), (48_000, 32_000)] {
            let x = sine(1000.0, sr_in, sr_in as usize);
            let y = resample(&x, sr_in, sr_out);
            assert_eq!(y.len(), sr_out as usize);
            let want = sine(1000.0, sr_out, sr_out as usize);
            // Ignore edges where the kernel runs off the signal.
            let mid = &y[2000..y.len() - 2000];
            let err = mid
                .iter()
                .zip(&want[2000..])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(err < 1e-3, "{sr_in}->{sr_out}: max err {err}");
        }
    }

    #[test]
    fn removes_content_above_new_nyquist() {
        // 20 kHz at 48 kHz must vanish when going to 32 kHz (Nyquist 16 kHz).
        let x = sine(20_000.0, 48_000, 48_000);
        let y = resample(&x, 48_000, 32_000);
        let rms = (y[2000..30_000].iter().map(|v| v * v).sum::<f64>() / 28_000.0).sqrt();
        assert!(rms < 1e-3, "alias rms {rms}");
    }
}
