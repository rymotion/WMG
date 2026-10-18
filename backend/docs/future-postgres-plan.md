# Future plan: Postgres persistence (Supabase-compatible)

Status: **not implemented**. The hackathon build stores everything on disk (`FsStore`, one
`meta.json` per upload). This document describes how to move to Postgres without tying the
service to a single vendor.

## Goals
- Run unchanged against plain Postgres (docker compose), Supabase, Neon, or RDS.
- Keep the HTTP API identical; only the storage backend changes.
- Survive restarts: identification jobs and analysis caches must not be lost.

## Backend selection
- Add `PgStore` implementing the existing `AudioStore` trait (`src/storage.rs`).
- Choose at startup: `DATABASE_URL` set → `PgStore`, otherwise `FsStore`.
- Use `sqlx` (features `postgres`, `runtime-tokio`, `tls-rustls`, `uuid`, `chrono`, `json`,
  `migrate`) with migrations in `migrations/` applied via `sqlx::migrate!()` at boot.
- With Supabase, connect through the Supavisor pooler in **session mode** (port 5432) or set
  `statement_cache_capacity = 0` if using transaction mode (port 6543).

## Schema
```sql
create type identification_status as enum ('pending','matched','no_match','error','disabled');

create table audio_files (
  id           uuid primary key,
  filename     text not null,
  mime         text not null,
  size_bytes   bigint not null,
  duration_s   double precision not null,
  sample_rate  integer not null,
  channels     smallint not null,
  codec        text,
  storage_key  text not null,          -- blob location (local path or s3 key)
  created_at   timestamptz not null default now()
);
create index on audio_files (created_at desc);

create table identifications (
  id             uuid primary key default gen_random_uuid(),
  audio_id       uuid not null references audio_files(id) on delete cascade,
  status         identification_status not null,
  provider       text not null default 'acrcloud',
  code           integer,
  message        text,
  best_title     text,
  best_artist    text,
  window_offset_s double precision,
  raw            jsonb,
  updated_at     timestamptz not null default now()
);
create unique index on identifications (audio_id, provider);
create index on identifications (status);

create table analysis_cache (
  audio_id      uuid not null references audio_files(id) on delete cascade,
  kind          text not null,
  params_hash   text not null,
  content_type  text not null,
  storage_key   text not null,
  created_at    timestamptz not null default now(),
  primary key (audio_id, kind, params_hash, content_type)
);

create table jobs (
  id          bigserial primary key,
  kind        text not null,            -- 'identify'
  audio_id    uuid not null references audio_files(id) on delete cascade,
  attempts    integer not null default 0,
  run_after   timestamptz not null default now(),
  locked_at   timestamptz,
  last_error  text
);
create index on jobs (run_after) where locked_at is null;
```

## Blob storage
- Introduce `trait BlobStore { put, get_range, delete, url }` with `LocalFs` and `S3`
  implementations (the `object_store` crate covers S3, GCS, Azure).
- Supabase Storage, Cloudflare R2, and MinIO are all S3-compatible.
- `/audio/{id}/stream` keeps Range support: proxy ranged reads, or return a 302 to a presigned URL.

## Background jobs
- Replace in-process `tokio::spawn` with a Postgres queue:
  `SELECT … FROM jobs WHERE locked_at IS NULL AND run_after <= now() ORDER BY id FOR UPDATE SKIP LOCKED LIMIT 1`.
- N worker tasks poll (or `LISTEN/NOTIFY`), retry with exponential backoff via `run_after`,
  and give up after a max attempt count, writing `status = 'error'`.

## Supabase extras (optional)
- **Realtime** on `identifications` so the frontend gets pushed updates instead of polling.
- **Auth + RLS**: verify the Supabase JWT in an axum middleware, add `owner_id uuid` columns,
  and add RLS policies `owner_id = auth.uid()` for direct client access.

## Migration from the hackathon build
- A one-off `backend migrate-fs` subcommand walks `DATA_DIR/*/meta.json`, inserts rows, and
  uploads blobs to the configured `BlobStore`.
