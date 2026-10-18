use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::json;

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("not found")]
    NotFound,
    #[error("upload exceeds the size limit")]
    PayloadTooLarge,
    #[error("unsupported or undecodable audio: {0}")]
    UnsupportedMedia(String),
    #[error("{0}")]
    BadRequest(String),
    #[error("invalid parameters: {0}")]
    InvalidParams(String),
    #[error("internal error: {0}")]
    Internal(String),
}

pub type AppResult<T> = Result<T, AppError>;

impl AppError {
    pub fn internal(e: impl std::fmt::Display) -> Self {
        Self::Internal(e.to_string())
    }
}

impl From<std::io::Error> for AppError {
    fn from(e: std::io::Error) -> Self {
        if e.kind() == std::io::ErrorKind::NotFound {
            Self::NotFound
        } else {
            Self::internal(e)
        }
    }
}

impl From<spectrograms::SpectrogramError> for AppError {
    fn from(e: spectrograms::SpectrogramError) -> Self {
        Self::InvalidParams(e.to_string())
    }
}

impl From<tokio::task::JoinError> for AppError {
    fn from(e: tokio::task::JoinError) -> Self {
        Self::internal(e)
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = match &self {
            Self::NotFound => StatusCode::NOT_FOUND,
            Self::PayloadTooLarge => StatusCode::PAYLOAD_TOO_LARGE,
            Self::UnsupportedMedia(_) => StatusCode::UNSUPPORTED_MEDIA_TYPE,
            Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            Self::InvalidParams(_) => StatusCode::UNPROCESSABLE_ENTITY,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        if status.is_server_error() {
            tracing::error!(error = %self, "request failed");
        }
        (status, Json(json!({ "error": self.to_string() }))).into_response()
    }
}
