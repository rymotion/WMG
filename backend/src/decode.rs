//! Audio decoding via symphonia (pure Rust: WAV, MP3, FLAC, OGG/Vorbis, AAC/M4A, AIFF, ...).

use std::{fs::File, path::Path};

use symphonia::core::{
    audio::SampleBuffer,
    codecs::{CODEC_TYPE_NULL, DecoderOptions},
    errors::Error as SymError,
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};

use crate::error::{AppError, AppResult};

/// File extensions accepted on upload.
pub const SUPPORTED_EXTENSIONS: &[&str] = &[
    "wav", "wave", "mp3", "flac", "ogg", "oga", "m4a", "mp4", "aac", "aif", "aiff", "caf", "mkv", "webm",
];

/// Decoded mono audio.
#[derive(Debug, Clone)]
pub struct DecodedAudio {
    pub samples: Vec<f64>,
    pub sample_rate: u32,
    pub channels: u16,
    pub codec: String,
}

impl DecodedAudio {
    pub fn duration_secs(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate as f64
    }

    /// Return the `[start, end)` window in seconds, clamped to the signal.
    pub fn window(&self, start: Option<f64>, end: Option<f64>) -> &[f64] {
        let sr = self.sample_rate as f64;
        let n = self.samples.len();
        let a = start.map_or(0, |s| ((s.max(0.0) * sr) as usize).min(n));
        let b = end.map_or(n, |e| ((e.max(0.0) * sr) as usize).min(n));
        &self.samples[a..b.max(a)]
    }
}

/// Decode a whole file, downmixing to mono. Fails if longer than `max_duration_secs`.
pub fn decode_file(path: &Path, max_duration_secs: f64) -> AppResult<DecodedAudio> {
    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| AppError::UnsupportedMedia(e.to_string()))?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| AppError::UnsupportedMedia("no audio track found".into()))?;
    let track_id = track.id;
    let codec = symphonia::default::get_codecs()
        .get_codec(track.codec_params.codec)
        .map(|d| d.short_name.to_string())
        .unwrap_or_else(|| "unknown".into());

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| AppError::UnsupportedMedia(e.to_string()))?;

    let mut samples: Vec<f64> = Vec::new();
    let mut sample_rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut channels = track.codec_params.channels.map_or(0, |c| c.count() as u16);
    let mut buf: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::ResetRequired) => break,
            Err(e) => return Err(AppError::UnsupportedMedia(e.to_string())),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            // Corrupt frames are skipped, as most players do.
            Err(SymError::DecodeError(_)) => continue,
            Err(e) => return Err(AppError::UnsupportedMedia(e.to_string())),
        };

        let spec = *decoded.spec();
        sample_rate = spec.rate;
        let ch = spec.channels.count().max(1);
        channels = ch as u16;

        let needed = decoded.capacity() as u64;
        if buf.as_ref().is_none_or(|b| (b.capacity() as u64) < needed * ch as u64) {
            buf = Some(SampleBuffer::<f32>::new(needed, spec));
        }
        let b = buf.as_mut().expect("buffer initialised above");
        b.copy_interleaved_ref(decoded);

        samples.extend(
            b.samples()
                .chunks_exact(ch)
                .map(|frame| frame.iter().map(|&s| s as f64).sum::<f64>() / ch as f64),
        );

        if sample_rate > 0 && samples.len() as f64 / sample_rate as f64 > max_duration_secs {
            return Err(AppError::UnsupportedMedia(format!(
                "audio longer than the {max_duration_secs:.0}s limit"
            )));
        }
    }

    if samples.is_empty() || sample_rate == 0 {
        return Err(AppError::UnsupportedMedia("file contains no decodable audio".into()));
    }

    Ok(DecodedAudio { samples, sample_rate, channels, codec })
}
