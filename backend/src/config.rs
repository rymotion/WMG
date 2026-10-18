use std::{env, path::PathBuf};

/// Runtime configuration, read from the environment (and `.env` in local dev).
#[derive(Debug, Clone)]
pub struct Config {
    pub port: u16,
    pub data_dir: PathBuf,
    pub max_upload_bytes: usize,
    pub max_duration_secs: f64,
    pub cors_origins: Vec<String>,
    pub acr: AcrConfig,
    /// Length of each clip saved from a `/live` stream.
    pub live_clip_secs: f64,
    /// Maximum length of one `/live` session.
    pub live_max_minutes: f64,
}

#[derive(Debug, Clone)]
pub struct AcrConfig {
    /// Base URL, e.g. `https://identify-us-west-2.acrcloud.com`. A bare host gets `https://`.
    pub base_url: String,
    pub access_key: String,
    pub access_secret: String,
    pub concurrency: usize,
    /// Music/custom-file matches scoring below this (0-100) count as `possible_match`.
    pub min_score: f64,
}

impl AcrConfig {
    pub fn enabled(&self) -> bool {
        !self.base_url.is_empty() && !self.access_key.is_empty() && !self.access_secret.is_empty()
    }
}

fn var(name: &str) -> Option<String> {
    env::var(name).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
}

fn parse<T: std::str::FromStr>(name: &str, default: T) -> T {
    var(name).and_then(|v| v.parse().ok()).unwrap_or(default)
}

impl Config {
    pub fn from_env() -> Self {
        let host = var("ACR_HOST").unwrap_or_default();
        let base_url = if host.is_empty() || host.starts_with("http://") || host.starts_with("https://") {
            host
        } else {
            format!("https://{host}")
        };
        Self {
            port: parse("PORT", 8000),
            data_dir: PathBuf::from(var("DATA_DIR").unwrap_or_else(|| "./data".into())),
            max_upload_bytes: parse::<usize>("MAX_UPLOAD_MB", 100) * 1024 * 1024,
            max_duration_secs: parse("MAX_DURATION_SECS", 1200.0),
            cors_origins: var("CORS_ORIGINS")
                .unwrap_or_else(|| "*".into())
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
            live_clip_secs: parse("LIVE_CLIP_SECS", 30.0_f64).max(1.0),
            live_max_minutes: parse("LIVE_MAX_MINUTES", 60.0_f64).max(1.0),
            acr: AcrConfig {
                base_url: base_url.trim_end_matches('/').to_string(),
                access_key: var("ACR_ACCESS_KEY").unwrap_or_default(),
                access_secret: var("ACR_ACCESS_SECRET").unwrap_or_default(),
                concurrency: parse("IDENTIFY_CONCURRENCY", 2usize).max(1),
                min_score: parse("ACR_MIN_SCORE", 50.0_f64),
            },
        }
    }
}
