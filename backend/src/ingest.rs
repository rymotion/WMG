//! Shared path for turning audio bytes into a stored upload (HTTP uploads and live clips).

use chrono::Utc;
use uuid::Uuid;

use crate::{
    AppState, decode,
    error::AppResult,
    storage::{AudioMeta, Identification, IdentificationStatus},
};

/// Live-session context for a clip.
#[derive(Debug, Clone, Copy)]
pub struct SessionClip {
    pub session_id: Uuid,
    pub index: u32,
}

/// Write `bytes`, decode them to validate and read metadata, and create the record.
///
/// When `identify` is true the record starts as `pending` (or `disabled` without
/// credentials); the caller decides how to run the job (`Identifier::spawn` or
/// `Identifier::run_and_store`). Otherwise it is marked `skipped`.
pub async fn ingest(
    state: &AppState,
    filename: String,
    extension: String,
    bytes: &[u8],
    identify: bool,
    session: Option<SessionClip>,
) -> AppResult<AudioMeta> {
    let id = Uuid::new_v4();
    let path = state.store.put_file(id, &extension, bytes).await?;
    let max = state.config.max_duration_secs;
    let decoded = tokio::task::spawn_blocking(move || decode::decode_file(&path, max)).await?;
    let audio = match decoded {
        Ok(a) => a,
        Err(e) => {
            let _ = state.store.delete(id).await;
            return Err(e);
        }
    };

    let meta = AudioMeta {
        id,
        filename,
        extension,
        size_bytes: bytes.len() as u64,
        duration_s: audio.duration_secs(),
        sample_rate: audio.sample_rate,
        channels: audio.channels,
        codec: audio.codec,
        created_at: Utc::now(),
        identification: if identify {
            state.identifier.initial_status()
        } else {
            Identification::with_message(IdentificationStatus::Skipped, "identify=false")
        },
        session_id: session.map(|s| s.session_id),
        clip_index: session.map(|s| s.index),
    };
    state.store.create(&meta).await?;
    Ok(meta)
}
