//! Personalised detector heads uploaded from the app (`.wmgmodel`, format `wmg-personal-head/1`).
//!
//! The app fine-tunes the model's final layer on device and sends it back here, together with
//! the labelled embeddings it was trained on. Files are stored as-is under
//! `DATA_DIR/personalized/<id>.wmgmodel.json`; `python -m wmg_train.personalize` merges them
//! into a full model.

use axum::{
    Json,
    extract::{Path, State},
    http::{StatusCode, header},
    response::IntoResponse,
};
use serde_json::{Value, json};
use tokio::fs;

use crate::{
    AppState,
    error::{AppError, AppResult},
};

const FORMAT: &str = "wmg-personal-head/1";

fn dir(state: &AppState) -> std::path::PathBuf {
    state.config.data_dir.join("personalized")
}

fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 80 && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
}

fn summary(id: &str, v: &Value, size: u64) -> Value {
    json!({
        "id": id,
        "base_model_run": v["base_model_run"],
        "feature_spec_hash": v["feature_spec_hash"],
        "device": v["device"],
        "created_at": v["created_at"],
        "exported_at": v["exported_at"],
        "samples": v["samples"].as_array().map_or(0, |a| a.len()),
        "train": v["train"],
        "size_bytes": size,
    })
}

/// `POST /models/personalized` with the `.wmgmodel` JSON as the body.
pub async fn upload(State(state): State<AppState>, Json(body): Json<Value>) -> AppResult<impl IntoResponse> {
    if body["format"] != FORMAT {
        return Err(AppError::BadRequest(format!("expected format {FORMAT}")));
    }
    let weights = body["head"]["weights"].as_array().ok_or_else(|| AppError::BadRequest("head.weights missing".into()))?;
    if weights.is_empty() || !weights.iter().all(Value::is_number) || !body["head"]["bias"].is_number() {
        return Err(AppError::BadRequest("head.weights / head.bias must be numbers".into()));
    }
    if !body["base_model_run"].is_string() || !body["feature_spec_hash"].is_string() {
        return Err(AppError::BadRequest("base_model_run and feature_spec_hash are required".into()));
    }
    let head_id = body["id"].as_str().filter(|s| valid_id(s)).unwrap_or("head");
    let id = format!(
        "{}-{}-{}",
        chrono::Utc::now().format("%Y%m%d-%H%M%S"),
        head_id,
        &uuid::Uuid::new_v4().simple().to_string()[..6]
    );
    let d = dir(&state);
    fs::create_dir_all(&d).await?;
    let bytes = serde_json::to_vec(&body).map_err(AppError::internal)?;
    fs::write(d.join(format!("{id}.wmgmodel.json")), &bytes).await?;
    tracing::info!(%id, size = bytes.len(), "personalised head uploaded");
    Ok((StatusCode::CREATED, Json(summary(&id, &body, bytes.len() as u64))))
}

/// `GET /models/personalized`: summaries, newest first.
pub async fn list(State(state): State<AppState>) -> AppResult<Json<Vec<Value>>> {
    let mut out = Vec::new();
    let Ok(mut rd) = fs::read_dir(dir(&state)).await else { return Ok(Json(out)) };
    while let Some(e) = rd.next_entry().await? {
        let name = e.file_name().to_string_lossy().to_string();
        let Some(id) = name.strip_suffix(".wmgmodel.json") else { continue };
        let bytes = fs::read(e.path()).await?;
        if let Ok(v) = serde_json::from_slice::<Value>(&bytes) {
            out.push(summary(id, &v, bytes.len() as u64));
        }
    }
    out.sort_by(|a, b| b["id"].as_str().cmp(&a["id"].as_str()));
    Ok(Json(out))
}

/// `GET /models/personalized/{id}`: the full `.wmgmodel` file.
pub async fn get(State(state): State<AppState>, Path(id): Path<String>) -> AppResult<impl IntoResponse> {
    if !valid_id(&id) {
        return Err(AppError::NotFound);
    }
    let bytes = fs::read(dir(&state).join(format!("{id}.wmgmodel.json"))).await?;
    Ok(([(header::CONTENT_TYPE, "application/json")], bytes))
}
