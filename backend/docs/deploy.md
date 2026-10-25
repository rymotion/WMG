# Deploying spectro-api (Fly.io + GitHub Actions)

Production runs as **one Fly.io machine with one volume** at `https://wmg-spectro-api.fly.dev`.
Every push to `main` runs clippy and the tests in GitHub Actions. If they pass, the job
deploys to Fly.

```
git push origin main ──► GitHub Actions: clippy + cargo test ──► flyctl deploy --remote-only
                                                                    │ (Fly builds the amd64 image)
                                                                    ▼
                                         Fly machine (sjc, 1 GB) ── volume spectro_data → /data
```

## Why only one machine?

The app keeps its state on local disk and runs identification jobs inside the process:
- uploads, `meta.json` files, and plot caches are all in `/data`;
- identification jobs run in the same process.

A Fly volume can be attached to only one machine. A second machine would therefore see a
different, empty `/data`. To scale out, storage first has to move to Postgres plus object
storage, as described in [future-postgres-plan.md](future-postgres-plan.md).

For the same reason, `auto_stop_machines` is `off`. Fly would otherwise stop an idle machine
in the middle of a `/live` WebSocket or a background identification.

## One-time setup

1. Install the Fly CLI and sign in. Sign-in opens a browser, and the Fly account needs a
   payment method.
   ```bash
   brew install flyctl
   ```
   ```bash
   fly auth login
   ```
2. Create the app from `fly.toml` without deploying yet. If `wmg-spectro-api` is already
   taken on Fly, choose another name and update `app` in `fly.toml`.
   ```bash
   fly launch --no-deploy --copy-config --name wmg-spectro-api --region sjc --yes
   ```
3. Create the data volume.
   ```bash
   fly volumes create spectro_data --region sjc --size 3 --yes
   ```
4. Add the ACRCloud credentials as secrets. They are encrypted on Fly and are never stored in
   `fly.toml` or git.
   ```bash
   fly secrets set ACR_ACCESS_KEY=... ACR_ACCESS_SECRET=... --stage
   ```
5. Run the first deploy, and keep it to a single machine.
   ```bash
   fly deploy --remote-only --ha=false
   ```
6. Let GitHub Actions deploy. Create a deploy token scoped to this app and store it as a
   repository secret. The token is piped straight into the secret, so it is never shown.
   ```bash
   fly tokens create deploy -x 8760h | gh secret set FLY_API_TOKEN
   ```

## Day to day

| Task | Command |
|---|---|
| Deploy | Push to `main`, or run `fly deploy --remote-only` |
| Logs | `fly logs` |
| Status and health checks | `fly status`, `fly checks list` |
| Shell on the machine | `fly ssh console` (data is in `/data`) |
| Restart | `fly machine restart` |
| Rotate ACRCloud keys | `fly secrets set ACR_ACCESS_KEY=… ACR_ACCESS_SECRET=…` (this restarts the app) |
| Change settings | Edit `[env]` in `fly.toml` and deploy. Set `CORS_ORIGINS` to the frontend's origin once there is one. |
| Grow the disk | `fly volumes list`, then `fly volumes extend <id> --size 10` |
| Backups | Fly snapshots volumes daily. List them with `fly volumes snapshots list <id>`. |

## Endpoints in production

- REST: `https://wmg-spectro-api.fly.dev/audio`, and so on. See the [README](../README.md).
- Live microphone streaming: `wss://wmg-spectro-api.fly.dev/live?sample_rate=16000`. Browsers
  allow microphone access on this HTTPS origin. See
  [live-streaming-client.md](live-streaming-client.md).

## Cost (approximate)

| Item | Approx. per month |
|---|---|
| shared-cpu-1x machine with 1 GB RAM, always on | ~$5–6 |
| 3 GB volume | ~$0.45 |

Outbound bandwidth is free up to Fly's allowance. Check Fly's pricing page for current rates.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Permission denied` when writing `/data` | The image's entrypoint runs `chown` on the volume at startup. Check that `docker-entrypoint.sh` is executable and present in the image. |
| Deploy says the volume is missing | Run `fly volumes list` and create `spectro_data` in the app's region. |
| Two machines are running | Run `fly scale count 1`. The app needs exactly one machine, because state lives on its volume. |
| Out-of-memory restarts on long uploads | Lower `MAX_DURATION_SECS` (default 1200 s; a 20-minute file uses about 420 MB of RAM to decode), or run `fly scale memory 2048`. |
| Every identification returns `error` | The secrets are missing or wrong. Run `fly secrets list`, then `fly logs` and look for ACRCloud code 3001 or 3014. |
