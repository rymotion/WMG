//! Feature extraction for the AI-vs-human music detector.
//!
//! This crate is the single implementation of the detector's input features. Training (via the
//! `python` feature), the backend and on-device code all call it, so a model never sees tensors
//! computed by a different implementation than the one it was trained on.
//!
//! Pipeline: mono `f64` at any rate → [`resample`] to 32 kHz → 10 s windows with 50% overlap,
//! dropping windows quieter than −45 dBFS → log-mel `[128, 1000]` per window ([`DetectorMel`]).

use ndarray::Array2;
use non_empty_slice::NonEmptySlice;
use serde::Serialize;
use sha2::{Digest, Sha256};
use spectrograms::{
    MelNorm, MelParams, MelPowerSpectrogram, SpectrogramParams, StftParams, WindowType,
};
use std::num::NonZeroUsize;
use std::sync::OnceLock;

#[cfg(feature = "python")]
mod python;
pub mod resample;

pub use resample::resample;

/// Git commit this crate was built from (or `WMG_FEATURES_COMMIT`, or "unknown").
pub const COMMIT: &str = env!("WMG_FEATURES_COMMIT");

#[derive(Debug, thiserror::Error)]
pub enum FeatureError {
    #[error("no audio samples")]
    Empty,
    #[error("sample rate must be > 0")]
    BadSampleRate,
    #[error("spectrogram: {0}")]
    Spectrogram(#[from] spectrograms::SpectrogramError),
}

pub type Result<T> = std::result::Result<T, FeatureError>;

/// Every parameter that affects the feature tensor. Serialised into model bundles and hashed
/// into [`spec_hash`]; any change here must bump `version`.
#[derive(Debug, Clone, Serialize)]
pub struct DetectorMelSpec {
    pub name: &'static str,
    pub version: u32,
    pub sample_rate: u32,
    pub n_mels: usize,
    /// Hann window length in samples (25 ms), zero-padded symmetrically to `n_fft`.
    pub win_length: usize,
    /// Frame hop in samples (10 ms).
    pub hop: usize,
    pub n_fft: usize,
    pub fmin: f64,
    pub fmax: f64,
    pub window: &'static str,
    pub mel_scale: &'static str,
    pub mel_norm: &'static str,
    /// Frames start at sample 0 (no centre padding); the tail is zero-padded so a window of
    /// N samples yields `ceil(N / hop)` frames.
    pub center: bool,
    /// 0 dB = strongest mel bin of a full-scale 1 kHz sine; values clamp at `db_floor`.
    pub db_reference: &'static str,
    pub db_floor: f64,
    pub window_seconds: f64,
    pub window_hop_seconds: f64,
    /// Windows whose RMS (20·log10 of sample RMS, full scale = 1.0) is below this are dropped.
    pub rms_gate_dbfs: f64,
    /// A clip shorter than one window but at least this long is zero-padded to one window.
    pub min_partial_seconds: f64,
}

pub const SPEC: DetectorMelSpec = DetectorMelSpec {
    name: "wmg-detector-mel",
    version: 1,
    sample_rate: 32_000,
    n_mels: 128,
    win_length: 800,
    hop: 320,
    n_fft: 1024,
    fmin: 20.0,
    fmax: 16_000.0,
    window: "hann-periodic",
    mel_scale: "slaney",
    mel_norm: "slaney",
    center: false,
    db_reference: "full-scale 1 kHz sine = 0 dB",
    db_floor: -100.0,
    window_seconds: 10.0,
    window_hop_seconds: 5.0,
    rms_gate_dbfs: -45.0,
    min_partial_seconds: 3.0,
};

impl DetectorMelSpec {
    pub fn window_samples(&self) -> usize {
        (self.window_seconds * self.sample_rate as f64).round() as usize
    }

    pub fn window_hop_samples(&self) -> usize {
        (self.window_hop_seconds * self.sample_rate as f64).round() as usize
    }

    /// Frames per full window (1000 for v1).
    pub fn frames_per_window(&self) -> usize {
        self.window_samples().div_ceil(self.hop)
    }
}

/// Canonical JSON of [`SPEC`], as written into manifests and model bundles.
pub fn spec_json() -> String {
    serde_json::to_string(&SPEC).expect("spec serialises")
}

/// First 16 hex chars of SHA-256 over [`spec_json`].
pub fn spec_hash() -> &'static str {
    static HASH: OnceLock<String> = OnceLock::new();
    HASH.get_or_init(|| {
        let digest = Sha256::digest(spec_json().as_bytes());
        digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
    })
}

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n.max(1)).expect("max(1) is non-zero")
}

/// RMS level in dBFS (full scale = 1.0); −inf for silence.
pub fn rms_dbfs(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return f64::NEG_INFINITY;
    }
    let ms = samples.iter().map(|v| v * v).sum::<f64>() / samples.len() as f64;
    10.0 * ms.log10()
}

/// Log-mel extractor for 32 kHz mono audio, configured from [`SPEC`].
pub struct DetectorMel {
    params: SpectrogramParams,
    mel: MelParams,
    ref_power: f64,
}

impl DetectorMel {
    pub fn new() -> Result<Self> {
        let s = &SPEC;
        // Periodic Hann of `win_length`, centred inside an `n_fft` frame (as torch.stft does
        // with win_length < n_fft).
        let pad = (s.n_fft - s.win_length) / 2;
        let mut coeffs = vec![0.0; s.n_fft];
        for i in 0..s.win_length {
            coeffs[pad + i] =
                0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / s.win_length as f64).cos();
        }
        let window = WindowType::custom(coeffs)?;
        let stft = StftParams::new(nz(s.n_fft), nz(s.hop), window, s.center)?;
        let params = SpectrogramParams::new(stft, s.sample_rate as f64)?;
        let mel = MelParams::with_norm(nz(s.n_mels), s.fmin, s.fmax, MelNorm::Slaney)?;

        let mut me = Self {
            params,
            mel,
            ref_power: 1.0,
        };
        let cal: Vec<f64> = (0..s.sample_rate as usize / 4)
            .map(|i| (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / s.sample_rate as f64).sin())
            .collect();
        me.ref_power = me
            .power(&cal)?
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max)
            .max(1e-30);
        Ok(me)
    }

    fn power(&self, samples: &[f64]) -> Result<Array2<f64>> {
        let slice = NonEmptySlice::new(samples).ok_or(FeatureError::Empty)?;
        let spec = MelPowerSpectrogram::compute(slice, &self.params, &self.mel, None)?;
        Ok((*spec).clone())
    }

    /// Log-mel `[n_mels, ceil(len / hop)]` in dB for 32 kHz mono `samples`.
    pub fn logmel(&self, samples: &[f64]) -> Result<Array2<f32>> {
        if samples.is_empty() {
            return Err(FeatureError::Empty);
        }
        let s = &SPEC;
        let n_frames = samples.len().div_ceil(s.hop);
        let mut padded = samples.to_vec();
        padded.resize((n_frames - 1) * s.hop + s.n_fft, 0.0);
        let power = self.power(&padded)?;
        debug_assert_eq!(power.ncols(), n_frames);
        Ok(power.mapv(|p| (10.0 * (p.max(1e-30) / self.ref_power).log10()).max(s.db_floor) as f32))
    }
}

/// One analysis window of a clip.
#[derive(Debug, Clone)]
pub struct Window {
    /// Start time in seconds relative to the clip.
    pub start_s: f64,
    pub rms_dbfs: f64,
    /// Log-mel `[n_mels, frames_per_window]`.
    pub logmel: Array2<f32>,
}

/// Window start offsets (in 32 kHz samples) for a clip of `len` samples, before RMS gating.
pub fn window_starts(len: usize) -> Vec<usize> {
    let (win, hop) = (SPEC.window_samples(), SPEC.window_hop_samples());
    if len < win {
        let min = (SPEC.min_partial_seconds * SPEC.sample_rate as f64) as usize;
        return if len >= min { vec![0] } else { Vec::new() };
    }
    (0..=(len - win) / hop).map(|i| i * hop).collect()
}

/// Full pipeline: resample `samples` (mono, `sr` Hz) to 32 kHz, cut 10 s / 50% windows,
/// drop windows under the RMS gate, return log-mel per surviving window.
pub fn extract_windows(mel: &DetectorMel, samples: &[f64], sr: u32) -> Result<Vec<Window>> {
    if sr == 0 {
        return Err(FeatureError::BadSampleRate);
    }
    let audio = resample(samples, sr, SPEC.sample_rate);
    let win = SPEC.window_samples();
    let mut out = Vec::new();
    for start in window_starts(audio.len()) {
        let end = (start + win).min(audio.len());
        let rms = rms_dbfs(&audio[start..end]);
        if rms < SPEC.rms_gate_dbfs {
            continue;
        }
        let mut chunk = audio[start..end].to_vec();
        chunk.resize(win, 0.0);
        out.push(Window {
            start_s: start as f64 / SPEC.sample_rate as f64,
            rms_dbfs: rms,
            logmel: mel.logmel(&chunk)?,
        });
    }
    Ok(out)
}
