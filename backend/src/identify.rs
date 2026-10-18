//! Background identification jobs: spawned after upload, results written to the store.

use std::{sync::Arc, time::Duration};

use tokio::sync::Semaphore;
use uuid::Uuid;

use crate::{
    acrcloud::{self, AcrClient, AcrError},
    config::Config,
    decode,
    storage::{AudioStore, Identification, IdentificationStatus},
};

/// Clip offsets tried in order; stops at the first match.
const OFFSETS_S: [f64; 3] = [0.0, 30.0, 60.0];
const CLIP_S: f64 = 10.0;
const MIN_CLIP_S: f64 = 3.0;
const ACR_NO_RESULT: i64 = 1001;

pub struct Identifier {
    client: Option<AcrClient>,
    permits: Semaphore,
    store: Arc<dyn AudioStore>,
    max_duration_secs: f64,
    min_score: f64,
}

impl Identifier {
    pub fn new(cfg: &Config, store: Arc<dyn AudioStore>) -> Self {
        let client = cfg.acr.enabled().then(|| AcrClient::new(cfg.acr.clone()));
        if client.is_none() {
            tracing::warn!("ACRCloud credentials not set; identification disabled");
        }
        Self { client, permits: Semaphore::new(cfg.acr.concurrency), store, max_duration_secs: cfg.max_duration_secs, min_score: cfg.acr.min_score }
    }

    pub fn enabled(&self) -> bool {
        self.client.is_some()
    }

    /// Initial status to record before the job runs.
    pub fn initial_status(&self) -> Identification {
        if self.enabled() {
            Identification::new(IdentificationStatus::Pending)
        } else {
            Identification::with_message(IdentificationStatus::Disabled, "ACRCloud credentials not configured")
        }
    }

    /// Fire-and-forget job. No-op when disabled.
    pub fn spawn(self: &Arc<Self>, id: Uuid) {
        if !self.enabled() {
            return;
        }
        let this = Arc::clone(self);
        tokio::spawn(async move {
            this.run_and_store(id).await;
        });
    }

    /// Run identification for `id`, persist the result, and return it.
    pub async fn run_and_store(&self, id: Uuid) -> Identification {
        let result = self.run(id).await;
        tracing::info!(%id, status = ?result.status, code = ?result.code, "identification finished");
        if let Err(e) = self.store.update_identification(id, result.clone()).await {
            // The upload may have been deleted while the job ran.
            tracing::warn!(%id, error = %e, "could not store identification result");
        }
        result
    }

    async fn run(&self, id: Uuid) -> Identification {
        let Some(client) = &self.client else {
            return self.initial_status();
        };
        let _permit = self.permits.acquire().await.expect("semaphore never closed");

        let path = match self.store.file_path(id).await {
            Ok(p) => p,
            Err(e) => return Identification::with_message(IdentificationStatus::Error, e.to_string()),
        };
        let max = self.max_duration_secs;
        let clips = match tokio::task::spawn_blocking(move || -> Result<Vec<(f64, Vec<u8>)>, String> {
            let audio = decode::decode_file(&path, max).map_err(|e| e.to_string())?;
            let dur = audio.duration_secs();
            Ok(OFFSETS_S
                .iter()
                .filter(|&&o| o == 0.0 || dur - o >= MIN_CLIP_S)
                .filter_map(|&o| acrcloud::clip_to_wav(&audio.samples, audio.sample_rate, o, CLIP_S).map(|w| (o, w)))
                .collect())
        })
        .await
        {
            Ok(Ok(c)) if !c.is_empty() => c,
            Ok(Ok(_)) => return Identification::with_message(IdentificationStatus::Error, "audio too short to identify"),
            Ok(Err(e)) => return Identification::with_message(IdentificationStatus::Error, e),
            Err(e) => return Identification::with_message(IdentificationStatus::Error, e.to_string()),
        };

        let mut last = Identification::new(IdentificationStatus::NoMatch);
        // Best humming-only result, used if no window yields a real recording match.
        let mut possible: Option<Identification> = None;
        for (offset, wav) in clips {
            let raw = match send_with_retry(client, wav).await {
                Ok(raw) => raw,
                Err(msg) => return Identification { window_offset_s: Some(offset), ..Identification::with_message(IdentificationStatus::Error, msg) },
            };
            let (code, msg) = acrcloud::status_of(&raw);
            let mut ident = Identification {
                code: Some(code),
                message: Some(msg),
                window_offset_s: Some(offset),
                ..Identification::new(IdentificationStatus::NoMatch)
            };
            match code {
                0 => {
                    ident.matches = acrcloud::parse_matches(&raw);
                    ident.raw = Some(raw);
                    // Confident recording match: music/custom-file bucket at or above the score threshold.
                    // Steady tones and noise can return music matches with scores around 30.
                    let min = self.min_score;
                    if ident.matches.iter().any(|m| m.source != "humming" && m.score.is_none_or(|s| s >= min)) {
                        ident.status = IdentificationStatus::Matched;
                        return ident;
                    }
                    // Humming or low-score candidates are fuzzy; keep looking for a confident match.
                    ident.status = IdentificationStatus::PossibleMatch;
                    let better = |p: &Identification| best_score(&ident) > best_score(p);
                    if possible.as_ref().is_none_or(better) {
                        possible = Some(ident);
                    }
                }
                ACR_NO_RESULT => last = ident,
                // Auth, quota, or request errors: trying another window will not help.
                _ => {
                    ident.status = IdentificationStatus::Error;
                    ident.raw = Some(raw);
                    return ident;
                }
            }
        }
        possible.unwrap_or(last)
    }
}

fn best_score(i: &Identification) -> f64 {
    i.matches.iter().filter_map(|m| m.score).fold(0.0, f64::max)
}

async fn send_with_retry(client: &AcrClient, wav: Vec<u8>) -> Result<serde_json::Value, String> {
    let mut delay = Duration::from_millis(500);
    for attempt in 0..3 {
        match client.identify_wav(wav.clone()).await {
            Ok(v) => return Ok(v),
            Err(AcrError::Transport(e)) if attempt < 2 => {
                tracing::warn!(attempt, error = %e, "ACRCloud request failed; retrying");
                tokio::time::sleep(delay).await;
                delay *= 2;
            }
            Err(AcrError::Transport(e) | AcrError::Fatal(e)) => return Err(e),
        }
    }
    unreachable!("loop returns on the final attempt")
}
