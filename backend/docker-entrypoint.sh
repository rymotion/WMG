#!/bin/sh
# Volumes on Fly.io (and fresh Docker volumes) are root-owned. Start as root just long
# enough to hand /data to the unprivileged `app` user, then drop privileges.
set -e
if [ "$(id -u)" = 0 ]; then
  chown -R app:app "${DATA_DIR:-/data}"
  exec setpriv --reuid=app --regid=app --init-groups "$@"
fi
exec "$@"
