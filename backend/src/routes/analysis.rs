use axum::{
    Json,
    extract::{Path, Query, State},
    http::{HeaderValue, header},
    response::{IntoResponse, Response},
};
use uuid::Uuid;

use crate::{
    AppState,
    analysis::{self, MatrixData, Stats},
    decode,
    error::{AppError, AppResult},
    params::{AnalysisParams, AnalysisQuery, Kind},
    plot,
};

fn cached(bytes: Vec<u8>, content_type: &'static str, key: &str) -> Response {
    let mut res = bytes.into_response();
    let h = res.headers_mut();
    h.insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("public, max-age=3600"));
    if let Ok(v) = HeaderValue::from_str(&format!("\"{key}\"")) {
        h.insert(header::ETAG, v);
    }
    res
}

/// Decode the file and run `f` on the requested time window, off the async runtime.
async fn with_window<T, F>(state: &AppState, id: Uuid, p: &AnalysisParams, f: F) -> AppResult<T>
where
    T: Send + 'static,
    F: FnOnce(&[f64], u32, f64) -> AppResult<T> + Send + 'static,
{
    let path = state.store.file_path(id).await?;
    let max = state.config.max_duration_secs;
    let (start, end) = (p.start, p.end);
    tokio::task::spawn_blocking(move || {
        let audio = decode::decode_file(&path, max)?;
        let offset = start.unwrap_or(0.0).min(audio.duration_secs());
        f(audio.window(start, end), audio.sample_rate, offset)
    })
    .await?
}

/// `GET /audio/{id}/plot/{kind}` — PNG.
pub async fn plot(
    State(state): State<AppState>,
    Path((id, kind)): Path<(Uuid, Kind)>,
    Query(q): Query<AnalysisQuery>,
) -> AppResult<Response> {
    let p = q.validate()?;
    let key = format!("{}.png", p.cache_key(kind));
    if let Some(bytes) = state.store.cache_get(id, &key).await {
        return Ok(cached(bytes, "image/png", &key));
    }
    let meta = state.store.get(id).await?;
    let title = format!("{} — {}", kind_title(kind), meta.filename);
    let pp = p.clone();
    let png = with_window(&state, id, &p, move |samples, sr, offset| {
        if samples.is_empty() {
            return Err(AppError::InvalidParams("selected time window contains no audio".into()));
        }
        match kind {
            Kind::Waveform => plot::waveform_png(samples, sr, &pp, &title, offset),
            _ => plot::heatmap_png(&analysis::compute(kind, samples, sr, &pp)?, &pp, &title, offset),
        }
    })
    .await?;
    state.store.cache_put(id, &key, &png).await;
    Ok(cached(png, "image/png", &key))
}

/// `GET /audio/{id}/data/{kind}` — JSON matrix for client-side rendering.
pub async fn data(
    State(state): State<AppState>,
    Path((id, kind)): Path<(Uuid, Kind)>,
    Query(q): Query<AnalysisQuery>,
) -> AppResult<Response> {
    if kind == Kind::Waveform {
        return Err(AppError::BadRequest("use /plot/waveform or /stream for waveform data".into()));
    }
    let p = q.validate()?;
    let key = format!("{}.json", p.cache_key(kind));
    if let Some(bytes) = state.store.cache_get(id, &key).await {
        return Ok(cached(bytes, "application/json", &key));
    }
    let pp = p.clone();
    let json = with_window(&state, id, &p, move |samples, sr, offset| {
        let m = analysis::compute(kind, samples, sr, &pp)?;
        serde_json::to_vec(&MatrixData::from_matrix(&m, pp.max_frames, offset)).map_err(AppError::internal)
    })
    .await?;
    state.store.cache_put(id, &key, &json).await;
    Ok(cached(json, "application/json", &key))
}

/// `GET /audio/{id}/stats`
pub async fn stats(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Query(q): Query<AnalysisQuery>,
) -> AppResult<Json<Stats>> {
    let p = q.validate()?;
    let pp = p.clone();
    let stats = with_window(&state, id, &p, move |samples, sr, _| analysis::stats(samples, sr, &pp)).await?;
    Ok(Json(stats))
}

fn kind_title(kind: Kind) -> &'static str {
    match kind {
        Kind::Waveform => "Waveform",
        Kind::Stft => "Linear STFT (dB)",
        Kind::Mel => "Mel spectrogram (dB)",
        Kind::Chroma => "Chromagram",
        Kind::Cqt => "Constant-Q transform (dB)",
    }
}
