use std::{
    io::{Read, Write},
    net::TcpStream,
    sync::Arc,
    time::Duration,
};

use spectro_api::{AppState, app, config::Config, storage::FsStore};
use tracing_subscriber::EnvFilter;

/// `--healthcheck`: tiny HTTP probe for Docker's HEALTHCHECK (the slim image has no curl).
fn healthcheck(port: u16) -> ! {
    let ok = (|| -> std::io::Result<bool> {
        let mut s = TcpStream::connect(("127.0.0.1", port))?;
        s.set_read_timeout(Some(Duration::from_secs(3)))?;
        s.write_all(b"GET /health HTTP/1.0\r\nHost: localhost\r\n\r\n")?;
        let mut buf = String::new();
        s.read_to_string(&mut buf)?;
        Ok(buf.starts_with("HTTP/1.1 200") || buf.starts_with("HTTP/1.0 200"))
    })()
    .unwrap_or(false);
    std::process::exit(if ok { 0 } else { 1 });
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();
    let config = Config::from_env();
    if std::env::args().any(|a| a == "--healthcheck") {
        healthcheck(config.port);
    }

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info,tower_http=info".into()))
        .init();

    let store = Arc::new(FsStore::new(&config.data_dir).await?);
    let port = config.port;
    tracing::info!(
        data_dir = %config.data_dir.display(),
        identification = config.acr.enabled(),
        "starting spectro-api on :{port}"
    );
    let router = app(AppState::new(config, store));

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

/// Resolves on Ctrl-C or SIGTERM (what `docker stop` sends).
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = term => {},
    }
    tracing::info!("shutting down");
}
