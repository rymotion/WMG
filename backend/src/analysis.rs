//! DSP built on the `spectrograms` crate. Pure functions over mono `f64` samples.

use ndarray::{Array2, Axis, s};
use non_empty_slice::NonEmptySlice;
use serde::Serialize;
use spectrograms::{
    ChromaNorm, ChromaParams, Chromagram, CqtParams, CqtPowerSpectrogram, LinearPowerSpectrogram,
    MelParams, MelPowerSpectrogram, SpectrogramParams, StftParams, chromagram,
};
use std::num::NonZeroUsize;

use crate::{
    error::{AppError, AppResult},
    params::{AnalysisParams, Kind},
};

/// A 2-D feature matrix (rows = bins, columns = frames) with its axes.
#[derive(Debug, Clone)]
pub struct Matrix {
    pub kind: Kind,
    pub values: Array2<f64>,
    /// Frame centre times in seconds (relative to the analysed window start).
    pub times: Vec<f64>,
    /// Bin centre frequencies in Hz (empty for chroma).
    pub freqs: Vec<f64>,
    /// Bin labels (chroma only).
    pub labels: Option<Vec<String>>,
    pub unit: &'static str,
    /// Colour-scale range used for plotting.
    pub vmin: f64,
    pub vmax: f64,
}

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n.max(1)).expect("max(1) is non-zero")
}

fn non_empty(samples: &[f64]) -> AppResult<&NonEmptySlice<f64>> {
    NonEmptySlice::new(samples)
        .ok_or_else(|| AppError::InvalidParams("selected time window contains no audio".into()))
}

fn spec_params(p: &AnalysisParams, n_fft: usize, hop: usize, sr: f64) -> AppResult<SpectrogramParams> {
    let stft = StftParams::new(nz(n_fft), nz(hop.min(n_fft)), p.window.into(), true)?;
    Ok(SpectrogramParams::new(stft, sr)?)
}

/// Convert power to dB relative to the matrix maximum (like librosa's `ref=np.max`),
/// clamped below at `floor_db`.
pub fn power_to_db(power: &Array2<f64>, floor_db: f64) -> Array2<f64> {
    let max = power.iter().cloned().fold(0.0_f64, f64::max);
    if max <= 0.0 {
        return Array2::from_elem(power.raw_dim(), floor_db);
    }
    power.mapv(|v| (10.0 * (v.max(1e-30) / max).log10()).max(floor_db))
}

pub fn compute(kind: Kind, samples: &[f64], sr: u32, p: &AnalysisParams) -> AppResult<Matrix> {
    match kind {
        Kind::Stft => stft_db(samples, sr, p),
        Kind::Mel => mel_db(samples, sr, p),
        Kind::Chroma => chroma(samples, sr, p),
        Kind::Cqt => cqt_db(samples, sr, p),
        Kind::Waveform => Err(AppError::BadRequest("waveform has no matrix data".into())),
    }
}

pub fn stft_power(samples: &[f64], sr: u32, p: &AnalysisParams) -> AppResult<LinearPowerSpectrogram> {
    let params = spec_params(p, p.n_fft, p.hop, sr as f64)?;
    Ok(LinearPowerSpectrogram::compute(non_empty(samples)?, &params, None)?)
}

pub fn stft_db(samples: &[f64], sr: u32, p: &AnalysisParams) -> AppResult<Matrix> {
    let spec = stft_power(samples, sr, p)?;
    let freqs: Vec<f64> = spec.frequencies().to_vec();
    let times: Vec<f64> = spec.times().to_vec();

    // Optional frequency crop.
    let lo = p.fmin.unwrap_or(0.0);
    let hi = p.fmax.unwrap_or(f64::INFINITY);
    let a = freqs.iter().position(|&f| f >= lo).unwrap_or(0);
    let b = freqs.iter().rposition(|&f| f <= hi).map_or(freqs.len(), |i| i + 1).max(a + 1);
    let power = spec.slice(s![a..b, ..]).to_owned();

    Ok(Matrix {
        kind: Kind::Stft,
        values: power_to_db(&power, p.db_floor),
        times,
        freqs: freqs[a..b].to_vec(),
        labels: None,
        unit: "dB",
        vmin: p.db_floor,
        vmax: 0.0,
    })
}

pub fn mel_db(samples: &[f64], sr: u32, p: &AnalysisParams) -> AppResult<Matrix> {
    let nyquist = sr as f64 / 2.0;
    let fmax = p.fmax.unwrap_or(nyquist).min(nyquist);
    let fmin = p.fmin.unwrap_or(0.0).min(fmax - 1.0).max(0.0);
    let params = spec_params(p, p.n_fft, p.hop, sr as f64)?;
    let mel = MelParams::new(nz(p.n_mels), fmin, fmax)?;
    let spec = MelPowerSpectrogram::compute(non_empty(samples)?, &params, &mel, None)?;
    Ok(Matrix {
        kind: Kind::Mel,
        values: power_to_db(&spec, p.db_floor),
        times: spec.times().to_vec(),
        freqs: spec.frequencies().to_vec(),
        labels: None,
        unit: "dB",
        vmin: p.db_floor,
        vmax: 0.0,
    })
}

pub fn chroma(samples: &[f64], sr: u32, p: &AnalysisParams) -> AppResult<Matrix> {
    let stft = StftParams::new(nz(p.n_fft), nz(p.hop), p.window.into(), true)?;
    let nyquist = sr as f64 / 2.0;
    let fmin = p.fmin.unwrap_or(32.7).max(1.0);
    let fmax = p.fmax.unwrap_or(4186.0).min(nyquist * 0.99);
    let cp = ChromaParams::new(440.0, fmin, fmax.max(fmin * 2.0), ChromaNorm::Max)?;
    let c: Chromagram<f64> = chromagram(non_empty(samples)?, &stft, sr as f64, &cp)?;
    let frames = c.data.ncols();
    let times = (0..frames).map(|i| (i * p.hop) as f64 / sr as f64).collect();
    let labels = Chromagram::<f64>::labels().iter().map(|s| s.to_string()).collect();
    Ok(Matrix {
        kind: Kind::Chroma,
        values: c.data,
        times,
        freqs: Vec::new(),
        labels: Some(labels),
        unit: "relative energy",
        vmin: 0.0,
        vmax: 1.0,
    })
}

pub fn cqt_db(samples: &[f64], sr: u32, p: &AnalysisParams) -> AppResult<Matrix> {
    let nyquist = sr as f64 / 2.0;
    let fmin = p.fmin.unwrap_or(32.703).max(8.0);
    let fmax = p.fmax.unwrap_or(nyquist * 0.9).min(nyquist * 0.95);
    if fmax <= fmin * 2.0 {
        return Err(AppError::InvalidParams("CQT needs fmax at least one octave above fmin".into()));
    }
    let bins_per_octave = 12;
    let n_octaves = ((fmax / fmin).log2().floor() as usize).clamp(1, 9);

    // CQT is a time-domain correlation per frame, so its cost scales with the frame count.
    // Use a longer kernel window and never compute more frames than will be returned.
    let n_fft = p.n_fft.max(8192);
    let hop = p.hop.max(samples.len().div_ceil(p.max_frames)).min(n_fft);
    let params = spec_params(p, n_fft, hop, sr as f64)?;
    let cqt = CqtParams::new(nz(bins_per_octave), nz(n_octaves), fmin)?;
    let spec = CqtPowerSpectrogram::compute(non_empty(samples)?, &params, &cqt, None)?;
    Ok(Matrix {
        kind: Kind::Cqt,
        values: power_to_db(&spec, p.db_floor),
        times: spec.times().to_vec(),
        freqs: spec.frequencies().to_vec(),
        labels: None,
        unit: "dB",
        vmin: p.db_floor,
        vmax: 0.0,
    })
}

impl Matrix {
    /// Average adjacent frames so there are at most `max_frames` columns.
    pub fn downsample_frames(&self, max_frames: usize) -> (Array2<f64>, Vec<f64>) {
        let n = self.values.ncols();
        if n <= max_frames {
            return (self.values.clone(), self.times.clone());
        }
        let group = n.div_ceil(max_frames);
        let out_n = n.div_ceil(group);
        let mut out = Array2::<f64>::zeros((self.values.nrows(), out_n));
        let mut times = Vec::with_capacity(out_n);
        for j in 0..out_n {
            let a = j * group;
            let b = (a + group).min(n);
            let chunk = self.values.slice(s![.., a..b]);
            out.column_mut(j).assign(&chunk.mean_axis(Axis(1)).expect("non-empty chunk"));
            times.push(self.times[a..b].iter().sum::<f64>() / (b - a) as f64);
        }
        (out, times)
    }
}

/// JSON payload for `/data/{kind}`.
#[derive(Debug, Serialize)]
pub struct MatrixData {
    pub kind: Kind,
    pub bins: usize,
    pub frames: usize,
    pub unit: &'static str,
    pub vmin: f64,
    pub vmax: f64,
    pub offset_s: f64,
    pub times: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub freqs: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub labels: Option<Vec<String>>,
    /// Row-major `[bin][frame]`.
    pub values: Vec<Vec<f32>>,
}

impl MatrixData {
    pub fn from_matrix(m: &Matrix, max_frames: usize, offset_s: f64) -> Self {
        let (values, times) = m.downsample_frames(max_frames);
        Self {
            kind: m.kind,
            bins: values.nrows(),
            frames: values.ncols(),
            unit: m.unit,
            vmin: m.vmin,
            vmax: m.vmax,
            offset_s,
            times: times.iter().map(|&t| (t + offset_s) as f32).collect(),
            freqs: (!m.freqs.is_empty()).then(|| m.freqs.iter().map(|&f| f as f32).collect()),
            labels: m.labels.clone(),
            values: values.rows().into_iter().map(|r| r.iter().map(|&v| v as f32).collect()).collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// Summary statistics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct Stats {
    pub duration_s: f64,
    pub sample_rate: u32,
    pub samples: usize,
    pub peak: f64,
    pub peak_dbfs: f64,
    pub rms: f64,
    pub rms_dbfs: f64,
    pub rms_frame_max: f64,
    pub zero_crossing_rate: f64,
    pub spectral_centroid_hz: f64,
    pub tempo_bpm: Option<f64>,
}

fn to_dbfs(x: f64) -> f64 {
    if x > 0.0 { 20.0 * x.log10() } else { f64::NEG_INFINITY }
}

pub fn stats(samples: &[f64], sr: u32, p: &AnalysisParams) -> AppResult<Stats> {
    let n = samples.len();
    if n == 0 {
        return Err(AppError::InvalidParams("selected time window contains no audio".into()));
    }
    let peak = samples.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    let rms = (samples.iter().map(|x| x * x).sum::<f64>() / n as f64).sqrt();
    let rms_frame_max = samples
        .chunks(p.n_fft)
        .map(|c| (c.iter().map(|x| x * x).sum::<f64>() / c.len() as f64).sqrt())
        .fold(0.0, f64::max);
    let crossings = samples.windows(2).filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0)).count();
    let zcr = crossings as f64 / (n.saturating_sub(1).max(1)) as f64;

    // Spectral centroid, energy-weighted across frames.
    let spec = stft_power(samples, sr, p)?;
    let freqs = spec.frequencies();
    let (mut num, mut den) = (0.0, 0.0);
    for frame in spec.columns() {
        for (&pw, &f) in frame.iter().zip(freqs.iter()) {
            num += pw * f;
            den += pw;
        }
    }
    let centroid = if den > 0.0 { num / den } else { 0.0 };

    let tempo = estimate_tempo(samples, sr, p).ok().flatten();

    Ok(Stats {
        duration_s: n as f64 / sr as f64,
        sample_rate: sr,
        samples: n,
        peak,
        peak_dbfs: to_dbfs(peak),
        rms,
        rms_dbfs: to_dbfs(rms),
        rms_frame_max,
        zero_crossing_rate: zcr,
        spectral_centroid_hz: centroid,
        tempo_bpm: tempo,
    })
}

/// Rough global tempo: spectral-flux onset envelope on a log-mel spectrogram, then
/// autocorrelation over 40-220 BPM with a log-normal prior centred on 120 BPM.
pub fn estimate_tempo(samples: &[f64], sr: u32, p: &AnalysisParams) -> AppResult<Option<f64>> {
    let hop = 512;
    let tp = AnalysisParams { n_fft: 2048, hop, n_mels: 64, fmin: None, fmax: None, db_floor: -80.0, ..p.clone() };
    let mel = mel_db(samples, sr, &tp)?.values;
    let frames = mel.ncols();
    if frames < 64 {
        return Ok(None);
    }
    // Positive spectral flux between consecutive frames.
    let mut onset: Vec<f64> = std::iter::once(0.0)
        .chain((1..frames).map(|t| {
            mel.column(t).iter().zip(mel.column(t - 1).iter()).map(|(a, b)| (a - b).max(0.0)).sum::<f64>()
        }))
        .collect();
    let mean = onset.iter().sum::<f64>() / frames as f64;
    onset.iter_mut().for_each(|v| *v -= mean);

    let fps = sr as f64 / hop as f64;
    let lag_min = (fps * 60.0 / 220.0).floor().max(1.0) as usize;
    let lag_max = ((fps * 60.0 / 40.0).ceil() as usize).min(frames - 1);
    let mut best: Option<(f64, usize)> = None;
    for lag in lag_min..=lag_max {
        let ac: f64 = onset[..frames - lag].iter().zip(&onset[lag..]).map(|(a, b)| a * b).sum();
        let bpm = 60.0 * fps / lag as f64;
        let prior = (-0.5 * ((bpm / 120.0).log2()).powi(2)).exp();
        let score = ac * prior;
        if best.is_none_or(|(s, _)| score > s) {
            best = Some((score, lag));
        }
    }
    Ok(best.filter(|(s, _)| *s > 0.0).map(|(_, lag)| (60.0 * fps / lag as f64 * 10.0).round() / 10.0))
}

// ---------------------------------------------------------------------------
// Live (streaming) mel spectrum
// ---------------------------------------------------------------------------

/// Streaming mel spectrogram for `/live`: no centre padding, so consecutive blocks tile
/// seamlessly, and absolute dB (0 dB ≈ a full-scale sine) so colours are stable over time.
pub struct LiveMel {
    params: SpectrogramParams,
    mel: MelParams,
    pub n_fft: usize,
    pub hop: usize,
    pub n_mels: usize,
    pub floor_db: f64,
    pub freqs: Vec<f64>,
    ref_power: f64,
}

impl LiveMel {
    pub fn new(sr: u32, n_mels: usize, floor_db: f64) -> AppResult<Self> {
        // ~32 ms hop, 50% overlap.
        let hop = ((sr as f64 * 0.032).round() as usize).next_power_of_two().clamp(128, 4096);
        let n_fft = hop * 2;
        let stft = StftParams::new(nz(n_fft), nz(hop), spectrograms::WindowType::Hanning, false)?;
        let params = SpectrogramParams::new(stft, sr as f64)?;
        let mel = MelParams::new(nz(n_mels), 0.0, sr as f64 / 2.0)?;

        // Calibrate: the strongest mel bin of a full-scale 1 kHz sine maps to 0 dB.
        let f = 1000.0_f64.min(sr as f64 / 4.0);
        let cal: Vec<f64> = (0..n_fft * 4).map(|i| (2.0 * std::f64::consts::PI * f * i as f64 / sr as f64).sin()).collect();
        let spec = MelPowerSpectrogram::compute(non_empty(&cal)?, &params, &mel, None)?;
        let ref_power = spec.iter().cloned().fold(0.0_f64, f64::max).max(1e-30);
        let freqs = spec.frequencies().to_vec();

        Ok(Self { params, mel, n_fft, hop, n_mels, floor_db, freqs, ref_power })
    }

    /// Frames computable from `samples`, as `[frame][bin]` dB, and how many samples they
    /// consumed (callers drop that many from the front of their buffer).
    pub fn frames(&self, samples: &[f64]) -> AppResult<(Vec<Vec<f64>>, usize)> {
        if samples.len() < self.n_fft {
            return Ok((Vec::new(), 0));
        }
        let n_frames = (samples.len() - self.n_fft) / self.hop + 1;
        let used = &samples[..(n_frames - 1) * self.hop + self.n_fft];
        let spec = MelPowerSpectrogram::compute(non_empty(used)?, &self.params, &self.mel, None)?;
        let frames = spec
            .columns()
            .into_iter()
            .map(|col| col.iter().map(|&p| (10.0 * (p.max(1e-30) / self.ref_power).log10()).max(self.floor_db)).collect())
            .collect();
        Ok((frames, n_frames * self.hop))
    }
}
