//! Embed the git commit the crate was built from, so every manifest row and model bundle can
//! record exactly which feature implementation produced its tensors.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=WMG_FEATURES_COMMIT");
    println!("cargo:rerun-if-changed=src");
    // Docker builds have no .git, so CI passes the commit in explicitly.
    let commit = std::env::var("WMG_FEATURES_COMMIT")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| {
            let out = Command::new("git")
                .args(["rev-parse", "HEAD"])
                .output()
                .ok()?;
            out.status
                .success()
                .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
        });
    println!(
        "cargo:rustc-env=WMG_FEATURES_COMMIT={}",
        commit.unwrap_or_else(|| "unknown".into())
    );
}
