//! Query parameters for analysis endpoints, with defaults and bounds checks.

use serde::{Deserialize, Serialize};
use spectrograms::WindowType;

use crate::error::{AppError, AppResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Waveform,
    Stft,
    Mel,
    Chroma,
    Cqt,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Waveform => "waveform",
            Self::Stft => "stft",
            Self::Mel => "mel",
            Self::Chroma => "chroma",
            Self::Cqt => "cqt",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Window {
    #[default]
    Hanning,
    Hamming,
    Blackman,
}

impl From<Window> for WindowType {
    fn from(w: Window) -> Self {
        match w {
            Window::Hanning => WindowType::Hanning,
            Window::Hamming => WindowType::Hamming,
            Window::Blackman => WindowType::Blackman,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Colormap {
    #[default]
    Magma,
    Viridis,
    Inferno,
    Gray,
}

/// Raw query string; every field optional.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct AnalysisQuery {
    pub n_fft: Option<usize>,
    pub hop: Option<usize>,
    pub window: Option<Window>,
    pub n_mels: Option<usize>,
    pub fmin: Option<f64>,
    pub fmax: Option<f64>,
    pub db_floor: Option<f64>,
    pub cmap: Option<Colormap>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub start: Option<f64>,
    pub end: Option<f64>,
    pub max_frames: Option<usize>,
}

/// Validated parameters. Serialized form doubles as the cache key.
#[derive(Debug, Clone, Serialize)]
pub struct AnalysisParams {
    pub n_fft: usize,
    pub hop: usize,
    pub window: Window,
    pub n_mels: usize,
    pub fmin: Option<f64>,
    pub fmax: Option<f64>,
    pub db_floor: f64,
    pub cmap: Colormap,
    pub width: u32,
    pub height: u32,
    pub start: Option<f64>,
    pub end: Option<f64>,
    pub max_frames: usize,
}

fn bad(msg: impl Into<String>) -> AppError {
    AppError::InvalidParams(msg.into())
}

impl AnalysisQuery {
    pub fn validate(self) -> AppResult<AnalysisParams> {
        let n_fft = self.n_fft.unwrap_or(2048);
        if !(64..=16384).contains(&n_fft) || !n_fft.is_power_of_two() {
            return Err(bad("n_fft must be a power of two between 64 and 16384"));
        }
        let hop = self.hop.unwrap_or(512);
        if hop == 0 || hop > n_fft {
            return Err(bad("hop must be between 1 and n_fft"));
        }
        let n_mels = self.n_mels.unwrap_or(128);
        if !(8..=512).contains(&n_mels) {
            return Err(bad("n_mels must be between 8 and 512"));
        }
        for (name, v) in [("fmin", self.fmin), ("fmax", self.fmax)] {
            if v.is_some_and(|f| !f.is_finite() || f < 0.0) {
                return Err(bad(format!("{name} must be a non-negative number")));
            }
        }
        if let (Some(lo), Some(hi)) = (self.fmin, self.fmax)
            && lo >= hi
        {
            return Err(bad("fmin must be less than fmax"));
        }
        let db_floor = self.db_floor.unwrap_or(-80.0);
        if !db_floor.is_finite() || !(-200.0..0.0).contains(&db_floor) {
            return Err(bad("db_floor must be between -200 and 0"));
        }
        let width = self.width.unwrap_or(1200);
        let height = self.height.unwrap_or(500);
        if !(200..=4000).contains(&width) || !(150..=4000).contains(&height) {
            return Err(bad("width must be 200-4000 and height 150-4000"));
        }
        for (name, v) in [("start", self.start), ("end", self.end)] {
            if v.is_some_and(|t| !t.is_finite() || t < 0.0) {
                return Err(bad(format!("{name} must be a non-negative number of seconds")));
            }
        }
        if let (Some(s), Some(e)) = (self.start, self.end)
            && s >= e
        {
            return Err(bad("start must be less than end"));
        }
        let max_frames = self.max_frames.unwrap_or(2000);
        if !(10..=20000).contains(&max_frames) {
            return Err(bad("max_frames must be between 10 and 20000"));
        }
        Ok(AnalysisParams {
            n_fft,
            hop,
            window: self.window.unwrap_or_default(),
            n_mels,
            fmin: self.fmin,
            fmax: self.fmax,
            db_floor,
            cmap: self.cmap.unwrap_or_default(),
            width,
            height,
            start: self.start,
            end: self.end,
            max_frames,
        })
    }
}

impl AnalysisParams {
    /// Stable (FNV-1a) hash of the parameters, used for cache file names.
    pub fn cache_key(&self, kind: Kind) -> String {
        let json = serde_json::to_string(self).unwrap_or_default();
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in kind.as_str().bytes().chain(json.bytes()) {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        format!("{}-{h:016x}", kind.as_str())
    }
}
