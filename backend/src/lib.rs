pub mod acrcloud;
pub mod analysis;
pub mod colormap;
pub mod config;
pub mod decode;
pub mod error;
pub mod identify;
pub mod ingest;
pub mod live;
pub mod params;
pub mod plot;
pub mod routes;
pub mod storage;

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::DefaultBodyLimit,
    http::{HeaderValue, Method},
    routing::{get, post},
};
use serde_json::json;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};

use crate::{config::Config, identify::Identifier, storage::AudioStore};

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub store: Arc<dyn AudioStore>,
    pub identifier: Arc<Identifier>,
}

impl AppState {
    pub fn new(config: Config, store: Arc<dyn AudioStore>) -> Self {
        let identifier = Arc::new(Identifier::new(&config, Arc::clone(&store)));
        Self { config: Arc::new(config), store, identifier }
    }
}

fn cors(origins: &[String]) -> CorsLayer {
    let layer = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST, Method::DELETE, Method::OPTIONS])
        .allow_headers(Any)
        .expose_headers(Any);
    if origins.iter().any(|o| o == "*") {
        layer.allow_origin(Any)
    } else {
        layer.allow_origin(origins.iter().filter_map(|o| HeaderValue::from_str(o).ok()).collect::<Vec<_>>())
    }
}

pub fn app(state: AppState) -> Router {
    use routes::{analysis as an, audio};
    let body_limit = state.config.max_upload_bytes;
    Router::new()
        .route("/health", get(|| async { Json(json!({ "status": "ok" })) }))
        .route("/audio", post(audio::upload).get(audio::list))
        .route("/audio/{id}", get(audio::get).delete(audio::delete))
        .route("/audio/{id}/stream", get(audio::stream))
        .route("/audio/{id}/identification", get(audio::identification))
        .route("/audio/{id}/identify", post(audio::reidentify))
        .route("/audio/{id}/stats", get(an::stats))
        .route("/audio/{id}/plot/{kind}", get(an::plot))
        .route("/audio/{id}/data/{kind}", get(an::data))
        .route("/live", get(live::handler))
        .layer(DefaultBodyLimit::max(body_limit))
        .layer(cors(&state.config.cors_origins))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
