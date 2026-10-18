//! Upload persistence behind a trait, so a Postgres-backed store can replace the
//! filesystem one later (see `docs/future-postgres-plan.md`).

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::{fs, sync::Mutex};
use uuid::Uuid;

use crate::error::{AppError, AppResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdentificationStatus {
    Pending,
    /// A recording (music or custom-file bucket) was identified.
    Matched,
    /// Only low-confidence humming/cover candidates were returned.
    PossibleMatch,
    NoMatch,
    Error,
    Disabled,
    Skipped,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrackMatch {
    /// ACR result bucket: `music`, `custom_files`, or `humming`.
    pub source: String,
    pub title: Option<String>,
    #[serde(default)]
    pub artists: Vec<String>,
    pub album: Option<String>,
    pub label: Option<String>,
    pub release_date: Option<String>,
    pub score: Option<f64>,
    pub duration_ms: Option<u64>,
    pub play_offset_ms: Option<u64>,
    pub acrid: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_ids: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Identification {
    pub status: IdentificationStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window_offset_s: Option<f64>,
    #[serde(default)]
    pub matches: Vec<TrackMatch>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<serde_json::Value>,
    pub updated_at: DateTime<Utc>,
}

impl Identification {
    pub fn new(status: IdentificationStatus) -> Self {
        Self { status, code: None, message: None, window_offset_s: None, matches: Vec::new(), raw: None, updated_at: Utc::now() }
    }

    pub fn with_message(status: IdentificationStatus, message: impl Into<String>) -> Self {
        Self { message: Some(message.into()), ..Self::new(status) }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioMeta {
    pub id: Uuid,
    pub filename: String,
    pub extension: String,
    pub size_bytes: u64,
    pub duration_s: f64,
    pub sample_rate: u32,
    pub channels: u16,
    pub codec: String,
    pub created_at: DateTime<Utc>,
    pub identification: Identification,
    /// Set for clips recorded through `/live`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<Uuid>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clip_index: Option<u32>,
}

#[async_trait]
pub trait AudioStore: Send + Sync {
    /// Persist the uploaded bytes and return the path they were written to.
    async fn put_file(&self, id: Uuid, extension: &str, bytes: &[u8]) -> AppResult<PathBuf>;
    async fn create(&self, meta: &AudioMeta) -> AppResult<()>;
    async fn get(&self, id: Uuid) -> AppResult<AudioMeta>;
    async fn list(&self) -> AppResult<Vec<AudioMeta>>;
    async fn update_identification(&self, id: Uuid, ident: Identification) -> AppResult<AudioMeta>;
    async fn delete(&self, id: Uuid) -> AppResult<()>;
    /// Local path of the original file.
    async fn file_path(&self, id: Uuid) -> AppResult<PathBuf>;
    async fn cache_get(&self, id: Uuid, key: &str) -> Option<Vec<u8>>;
    async fn cache_put(&self, id: Uuid, key: &str, bytes: &[u8]);
}

/// `DATA_DIR/<uuid>/{original.<ext>, meta.json, cache/}`.
pub struct FsStore {
    root: PathBuf,
    /// Serialises read-modify-write of meta.json (writes are rare).
    write_lock: Mutex<()>,
}

impl FsStore {
    pub async fn new(root: impl Into<PathBuf>) -> AppResult<Self> {
        let root = root.into();
        fs::create_dir_all(&root).await?;
        Ok(Self { root, write_lock: Mutex::new(()) })
    }

    fn dir(&self, id: Uuid) -> PathBuf {
        self.root.join(id.to_string())
    }

    fn meta_path(&self, id: Uuid) -> PathBuf {
        self.dir(id).join("meta.json")
    }

    fn cache_path(&self, id: Uuid, key: &str) -> PathBuf {
        self.dir(id).join("cache").join(key)
    }

    async fn write_meta(&self, meta: &AudioMeta) -> AppResult<()> {
        let path = self.meta_path(meta.id);
        let tmp = path.with_extension("json.tmp");
        let json = serde_json::to_vec_pretty(meta).map_err(AppError::internal)?;
        fs::write(&tmp, json).await?;
        fs::rename(&tmp, &path).await?;
        Ok(())
    }
}

async fn read_json(path: &Path) -> AppResult<AudioMeta> {
    let bytes = fs::read(path).await?;
    serde_json::from_slice(&bytes).map_err(AppError::internal)
}

#[async_trait]
impl AudioStore for FsStore {
    async fn put_file(&self, id: Uuid, extension: &str, bytes: &[u8]) -> AppResult<PathBuf> {
        let dir = self.dir(id);
        fs::create_dir_all(dir.join("cache")).await?;
        let path = dir.join(format!("original.{extension}"));
        fs::write(&path, bytes).await?;
        Ok(path)
    }

    async fn create(&self, meta: &AudioMeta) -> AppResult<()> {
        let _guard = self.write_lock.lock().await;
        self.write_meta(meta).await
    }

    async fn get(&self, id: Uuid) -> AppResult<AudioMeta> {
        read_json(&self.meta_path(id)).await
    }

    async fn list(&self) -> AppResult<Vec<AudioMeta>> {
        let mut out = Vec::new();
        let mut entries = fs::read_dir(&self.root).await?;
        while let Some(entry) = entries.next_entry().await? {
            if let Ok(meta) = read_json(&entry.path().join("meta.json")).await {
                out.push(meta);
            }
        }
        out.sort_by_key(|m| std::cmp::Reverse(m.created_at));
        Ok(out)
    }

    async fn update_identification(&self, id: Uuid, ident: Identification) -> AppResult<AudioMeta> {
        let _guard = self.write_lock.lock().await;
        let mut meta = self.get(id).await?;
        meta.identification = ident;
        self.write_meta(&meta).await?;
        Ok(meta)
    }

    async fn delete(&self, id: Uuid) -> AppResult<()> {
        let _guard = self.write_lock.lock().await;
        let dir = self.dir(id);
        if !fs::try_exists(&dir).await.unwrap_or(false) {
            return Err(AppError::NotFound);
        }
        fs::remove_dir_all(dir).await?;
        Ok(())
    }

    async fn file_path(&self, id: Uuid) -> AppResult<PathBuf> {
        let meta = self.get(id).await?;
        Ok(self.dir(id).join(format!("original.{}", meta.extension)))
    }

    async fn cache_get(&self, id: Uuid, key: &str) -> Option<Vec<u8>> {
        fs::read(self.cache_path(id, key)).await.ok()
    }

    async fn cache_put(&self, id: Uuid, key: &str, bytes: &[u8]) {
        let path = self.cache_path(id, key);
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent).await;
        }
        if let Err(e) = fs::write(&path, bytes).await {
            tracing::warn!(%id, key, error = %e, "failed to write cache entry");
        }
    }
}
