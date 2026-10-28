//! C ABI for on-device use (Swift/Kotlin). Built with `--features ffi`; header in
//! `include/wmg_features.h`. Every function is panic-safe: a panic returns an error code.

use std::ffi::{CString, c_char};
use std::panic::catch_unwind;
use std::sync::OnceLock;

use crate::{DetectorMel, SPEC, extract_windows, normalize_level, resample, window_starts};

fn mel() -> Option<&'static DetectorMel> {
    static MEL: OnceLock<Option<DetectorMel>> = OnceLock::new();
    MEL.get_or_init(|| DetectorMel::new().ok()).as_ref()
}

#[unsafe(no_mangle)]
pub extern "C" fn wmg_n_mels() -> u32 {
    SPEC.n_mels as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn wmg_frames_per_window() -> u32 {
    SPEC.frames_per_window() as u32
}

/// NUL-terminated feature spec hash; must match the model's `feature_spec_hash`.
#[unsafe(no_mangle)]
pub extern "C" fn wmg_spec_hash() -> *const c_char {
    static S: OnceLock<CString> = OnceLock::new();
    S.get_or_init(|| CString::new(crate::spec_hash()).expect("hex has no NUL")).as_ptr()
}

/// Upper bound on windows `wmg_extract_windows` can return for `n_samples` at `sample_rate`.
#[unsafe(no_mangle)]
pub extern "C" fn wmg_max_windows(n_samples: usize, sample_rate: u32) -> usize {
    if sample_rate == 0 {
        return 0;
    }
    let n32 = (n_samples as f64 * SPEC.sample_rate as f64 / sample_rate as f64).round() as usize;
    window_starts(n32).len()
}

/// Mono float samples -> detector windows.
///
/// If `normalize` is non-zero the clip is first scaled to the training RMS level
/// (`LEVEL_TARGET_DBFS`). Writes up to `max_windows` log-mels, each `n_mels * frames` floats
/// row-major `[mel][frame]`, into `out_mel`, and their start times (s) into `out_start_s`.
/// Returns the number of windows written, or -1 on bad input / internal error.
///
/// # Safety
/// `samples` must point to `n_samples` floats; `out_mel` to `max_windows * n_mels * frames`
/// floats; `out_start_s` to `max_windows` floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wmg_extract_windows(
    samples: *const f32,
    n_samples: usize,
    sample_rate: u32,
    normalize: i32,
    out_mel: *mut f32,
    out_start_s: *mut f32,
    max_windows: usize,
) -> i64 {
    if samples.is_null() || out_mel.is_null() || out_start_s.is_null() || sample_rate == 0 {
        return -1;
    }
    let result = catch_unwind(|| {
        let input = unsafe { std::slice::from_raw_parts(samples, n_samples) };
        let mut x: Vec<f64> = input.iter().map(|&v| v as f64).collect();
        if normalize != 0 {
            if crate::rms_dbfs(&x) < crate::MIN_INPUT_DBFS {
                return Some(0); // silence: nothing to score
            }
            x = normalize_level(&x);
        }
        let windows = extract_windows(mel()?, &x, sample_rate).ok()?;
        let per = SPEC.n_mels * SPEC.frames_per_window();
        let n = windows.len().min(max_windows);
        let out = unsafe { std::slice::from_raw_parts_mut(out_mel, max_windows * per) };
        let starts = unsafe { std::slice::from_raw_parts_mut(out_start_s, max_windows) };
        for (i, w) in windows.iter().take(n).enumerate() {
            out[i * per..(i + 1) * per].copy_from_slice(w.logmel.as_slice()?);
            starts[i] = w.start_s as f32;
        }
        Some(n as i64)
    });
    result.ok().flatten().unwrap_or(-1)
}

/// Resample helper exposed for tests and tooling. Returns samples written, or -1.
///
/// # Safety
/// `input` must point to `n` floats; `out` to `out_cap` floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wmg_resample(
    input: *const f32,
    n: usize,
    sr_in: u32,
    sr_out: u32,
    out: *mut f32,
    out_cap: usize,
) -> i64 {
    if input.is_null() || out.is_null() || sr_in == 0 || sr_out == 0 {
        return -1;
    }
    let x: Vec<f64> = unsafe { std::slice::from_raw_parts(input, n) }.iter().map(|&v| v as f64).collect();
    let y = resample(&x, sr_in, sr_out);
    let m = y.len().min(out_cap);
    let o = unsafe { std::slice::from_raw_parts_mut(out, out_cap) };
    for i in 0..m {
        o[i] = y[i] as f32;
    }
    m as i64
}
