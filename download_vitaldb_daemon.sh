#!/usr/bin/env bash
# Pre-download the whole VitalDB dataset (~95 GB) into the local cache (data/vitaldb)
# as a transient systemd user service (survives terminal close).
# Extra arguments go to `vitaldb-cache download`, e.g. case IDs or --jobs 4.
# Logs:   tail -f data/vitaldb-download.log
# Status: systemctl --user status vitaldb-download
# Stop:   systemctl --user stop vitaldb-download
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
# Restart on failure is safe: cached files are verified and skipped.
systemd-run --user --unit=vitaldb-download --collect \
  -p Restart=on-failure -p RestartSec=30 \
  -p WorkingDirectory="$PWD" \
  -p StandardOutput=append:"$PWD/data/vitaldb-download.log" \
  -p StandardError=append:"$PWD/data/vitaldb-download.log" \
  "$(command -v uv)" run vitaldb-cache download "$@"
echo "Started. Follow logs with: tail -f $PWD/data/vitaldb-download.log"
