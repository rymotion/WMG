use axum::{
    Json,
    body::Body,
    extract::{Multipart, Path, Query, Request, State, multipart::MultipartError},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Deserialize;
use tower::ServiceExt;
use tower_http::services::ServeFile;
use uuid::Uuid;

use crate::{
    AppState, decode,
    error::{AppError, AppResult},
    ingest,
    storage::{AudioMeta, Identification},
};

#[derive(Debug, Deserialize)]
pub struct UploadQuery {
    /// Run ACRCloud identification in the background (default true).
    pub identify: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct ListQuery {
    /// Only clips recorded in this `/live` session, ordered by clip index.
    pub session_id: Option<Uuid>,
}

fn multipart_err(e: MultipartError) -> AppError {
    if e.status() == StatusCode::PAYLOAD_TOO_LARGE {
        AppError::PayloadTooLarge
    } else {
        AppError::BadRequest(e.body_text())
    }
}

/// `POST /audio` — multipart field `file`.
pub async fn upload(
    State(state): State<AppState>,
    Query(q): Query<UploadQuery>,
    mut multipart: Multipart,
) -> AppResult<(StatusCode, Json<AudioMeta>)> {
    let mut upload = None;
    while let Some(field) = multipart.next_field().await.map_err(multipart_err)? {
        if field.name() == Some("file") {
            let filename = field.file_name().unwrap_or("upload").to_string();
            let bytes = field.bytes().await.map_err(multipart_err)?;
            upload = Some((filename, bytes));
            break;
        }
    }
    let (filename, bytes) = upload.ok_or_else(|| AppError::BadRequest("missing multipart field `file`".into()))?;
    if bytes.is_empty() {
        return Err(AppError::BadRequest("uploaded file is empty".into()));
    }

    let extension = std::path::Path::new(&filename)
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .filter(|e| decode::SUPPORTED_EXTENSIONS.contains(&e.as_str()))
        .ok_or_else(|| {
            AppError::UnsupportedMedia(format!("supported extensions: {}", decode::SUPPORTED_EXTENSIONS.join(", ")))
        })?;

    let identify = q.identify.unwrap_or(true);
    let meta = ingest::ingest(&state, filename, extension, &bytes, identify, None).await?;
    if identify {
        state.identifier.spawn(meta.id);
    }
    let id = meta.id;
    tracing::info!(%id, filename = %meta.filename, duration = meta.duration_s, "uploaded");
    Ok((StatusCode::CREATED, Json(meta)))
}

pub async fn list(State(state): State<AppState>, Query(q): Query<ListQuery>) -> AppResult<Json<Vec<AudioMeta>>> {
    let mut items = state.store.list().await?;
    if let Some(session) = q.session_id {
        items.retain(|m| m.session_id == Some(session));
        items.sort_by_key(|m| m.clip_index);
    }
    Ok(Json(items))
}

pub async fn get(State(state): State<AppState>, Path(id): Path<Uuid>) -> AppResult<Json<AudioMeta>> {
    Ok(Json(state.store.get(id).await?))
}

pub async fn delete(State(state): State<AppState>, Path(id): Path<Uuid>) -> AppResult<StatusCode> {
    state.store.delete(id).await?;
    Ok(StatusCode::NO_CONTENT)
}

pub async fn identification(State(state): State<AppState>, Path(id): Path<Uuid>) -> AppResult<Json<Identification>> {
    Ok(Json(state.store.get(id).await?.identification))
}

/// `POST /audio/{id}/identify` — re-run identification.
pub async fn reidentify(State(state): State<AppState>, Path(id): Path<Uuid>) -> AppResult<(StatusCode, Json<Identification>)> {
    state.store.get(id).await?;
    if !state.identifier.enabled() {
        return Err(AppError::BadRequest("identification is disabled (ACRCloud credentials not configured)".into()));
    }
    let pending = state.identifier.initial_status();
    state.store.update_identification(id, pending.clone()).await?;
    state.identifier.spawn(id);
    Ok((StatusCode::ACCEPTED, Json(pending)))
}

/// `GET /audio/{id}/stream` — original file with HTTP Range support for playback/seeking.
pub async fn stream(State(state): State<AppState>, Path(id): Path<Uuid>, req: Request) -> AppResult<Response> {
    let path = state.store.file_path(id).await?;
    let res = ServeFile::new(path).oneshot(req).await.map_err(AppError::internal)?;
    Ok(res.map(Body::new).into_response())
}
