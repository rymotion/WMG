//! Minimal ACRCloud identification client (REST `POST /v1/identify`, `data_type=audio`).
//!
//! Docs: https://docs.acrcloud.com/reference/identification-api/identification-api
//! string_to_sign = "POST\n/v1/identify\n{access_key}\naudio\n1\n{timestamp}",
//! signature = base64(HMAC-SHA1(access_secret, string_to_sign)).

use std::{io::Cursor, time::Duration};

use base64::{Engine, engine::general_purpose::STANDARD as B64};
use hmac::{Hmac, Mac};
use reqwest::multipart::{Form, Part};
use serde_json::Value;
use sha1::Sha1;

use crate::{config::AcrConfig, storage::TrackMatch};

const ENDPOINT: &str = "/v1/identify";
const DATA_TYPE: &str = "audio";
const SIGNATURE_VERSION: &str = "1";
/// ACR recommends < 15 s clips; 8 kHz mono 16-bit keeps 10 s at ~160 KB.
pub const CLIP_SAMPLE_RATE: u32 = 8000;

pub fn sign(access_key: &str, access_secret: &str, timestamp: &str) -> String {
    let string_to_sign = format!("POST\n{ENDPOINT}\n{access_key}\n{DATA_TYPE}\n{SIGNATURE_VERSION}\n{timestamp}");
    let mut mac = Hmac::<Sha1>::new_from_slice(access_secret.as_bytes()).expect("HMAC accepts any key length");
    mac.update(string_to_sign.as_bytes());
    B64.encode(mac.finalize().into_bytes())
}

/// Cut `[offset_s, offset_s + len_s)` from mono samples, resample to 8 kHz by
/// box-filter averaging (adequate anti-aliasing for fingerprinting), encode as WAV.
pub fn clip_to_wav(samples: &[f64], sr: u32, offset_s: f64, len_s: f64) -> Option<Vec<u8>> {
    let sr_f = sr as f64;
    let a = ((offset_s * sr_f) as usize).min(samples.len());
    let b = (((offset_s + len_s) * sr_f) as usize).min(samples.len());
    let clip = &samples[a..b];
    if clip.is_empty() {
        return None;
    }

    let ratio = sr_f / CLIP_SAMPLE_RATE as f64;
    let out_len = ((clip.len() as f64) / ratio).floor().max(1.0) as usize;
    let resampled: Vec<f64> = if ratio <= 1.0 {
        // Source already at or below 8 kHz: nearest-sample pick.
        (0..out_len).map(|i| clip[((i as f64 * ratio) as usize).min(clip.len() - 1)]).collect()
    } else {
        (0..out_len)
            .map(|i| {
                let s = (i as f64 * ratio) as usize;
                let e = (((i + 1) as f64 * ratio) as usize).clamp(s + 1, clip.len());
                clip[s..e].iter().sum::<f64>() / (e - s) as f64
            })
            .collect()
    };

    // Peak-normalise quiet clips a little so the fingerprint has energy to work with.
    let peak = resampled.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    let gain = if peak > 0.0 && peak < 0.5 { 0.5 / peak } else { 1.0 };

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: CLIP_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::with_capacity(out_len * 2 + 44));
    {
        let mut w = hound::WavWriter::new(&mut cursor, spec).ok()?;
        for &x in &resampled {
            w.write_sample(((x * gain).clamp(-1.0, 1.0) * i16::MAX as f64) as i16).ok()?;
        }
        w.finalize().ok()?;
    }
    Some(cursor.into_inner())
}

#[derive(Debug)]
pub enum AcrError {
    /// Network failure or 5xx; worth retrying.
    Transport(String),
    /// Non-retryable HTTP or parse error.
    Fatal(String),
}

pub struct AcrClient {
    http: reqwest::Client,
    cfg: AcrConfig,
}

impl AcrClient {
    pub fn new(cfg: AcrConfig) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(15))
            .build()
            .expect("reqwest client builds with static config");
        Self { http, cfg }
    }

    /// Send one clip; returns the raw JSON body.
    pub async fn identify_wav(&self, wav: Vec<u8>) -> Result<Value, AcrError> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or_default()
            .to_string();
        let signature = sign(&self.cfg.access_key, &self.cfg.access_secret, &timestamp);
        let sample_bytes = wav.len().to_string();
        let part = Part::bytes(wav).file_name("sample.wav").mime_str("audio/wav").map_err(|e| AcrError::Fatal(e.to_string()))?;
        let form = Form::new()
            .part("sample", part)
            .text("access_key", self.cfg.access_key.clone())
            .text("sample_bytes", sample_bytes)
            .text("timestamp", timestamp)
            .text("signature", signature)
            .text("data_type", DATA_TYPE)
            .text("signature_version", SIGNATURE_VERSION);

        let url = format!("{}{ENDPOINT}", self.cfg.base_url);
        let resp = self.http.post(url).multipart(form).send().await.map_err(|e| AcrError::Transport(e.without_url().to_string()))?;
        let status = resp.status();
        if status.is_server_error() {
            return Err(AcrError::Transport(format!("ACRCloud returned HTTP {status}")));
        }
        if !status.is_success() {
            return Err(AcrError::Fatal(format!("ACRCloud returned HTTP {status}")));
        }
        resp.json::<Value>().await.map_err(|e| AcrError::Fatal(format!("invalid ACRCloud response: {e}")))
    }
}

/// `(status.code, status.msg)` from an ACR response.
pub fn status_of(raw: &Value) -> (i64, String) {
    let code = raw.pointer("/status/code").and_then(Value::as_i64).unwrap_or(-1);
    let msg = raw.pointer("/status/msg").and_then(Value::as_str).unwrap_or("").to_string();
    (code, msg)
}

/// ACR result buckets, most authoritative first.
pub const SOURCES: [&str; 3] = ["music", "custom_files", "humming"];

/// Normalise `metadata.{music,custom_files,humming}[]` into `TrackMatch`es.
/// Every upstream field is optional, and numbers sometimes arrive as strings.
pub fn parse_matches(raw: &Value) -> Vec<TrackMatch> {
    let s = |v: &Value, p: &str| v.pointer(p).and_then(Value::as_str).map(str::to_string);
    let num = |v: &Value, p: &str| -> Option<f64> {
        let x = v.pointer(p)?;
        x.as_f64().or_else(|| x.as_str()?.parse().ok())
    };
    SOURCES
        .iter()
        .flat_map(|&source| {
            raw.pointer(&format!("/metadata/{source}"))
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(move |m| (source, m))
        })
        .map(|(source, m)| TrackMatch {
            source: source.to_string(),
            title: s(m, "/title"),
            artists: m
                .get("artists")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(|x| s(x, "/name")).collect())
                .unwrap_or_default(),
            album: s(m, "/album/name"),
            label: s(m, "/label"),
            release_date: s(m, "/release_date"),
            score: num(m, "/score"),
            duration_ms: num(m, "/duration_ms").map(|v| v as u64),
            play_offset_ms: num(m, "/play_offset_ms").map(|v| v as u64),
            acrid: s(m, "/acrid"),
            external_ids: m.get("external_ids").cloned().filter(|v| v.as_object().is_some_and(|o| !o.is_empty())),
            external_metadata: m.get("external_metadata").cloned().filter(|v| v.as_object().is_some_and(|o| !o.is_empty())),
        })
        .collect()
}
