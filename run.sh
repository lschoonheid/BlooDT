#!/usr/bin/env bash
# Start the live stack and replay a case into the dashboard.
#
#   ./run.sh                 # replay case 17 with run "lean"
#   ./run.sh 45              # another case
#   ./run.sh 17 sixth        # another training run (models/<run>)
#   SPEED=120 ./run.sh       # faster replay (default 30x)
#   ./run.sh stop            # stop the stack and the Docker VM (data is kept)
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == stop ]]; then
  docker compose down
  command -v colima >/dev/null && colima stop
  exit 0
fi

CASE="${1:-17}"
export RUN="${2:-${RUN:-lean}}"
[[ -f "models/$RUN/models.joblib" ]] || { echo "no models/$RUN/models.joblib" >&2; exit 1; }

# the Docker engine: Colima if it's installed, else Docker Desktop
if ! docker info >/dev/null 2>&1; then
  if command -v colima >/dev/null; then colima start; else open -a Docker; fi
  until docker info >/dev/null 2>&1; do sleep 2; done
fi

docker compose up -d --build
echo "Grafana: log in with admin/admin; open the link printed below. Ctrl-C stops the replay only."
docker compose run --rm replayer "$CASE" --speed "${SPEED:-30}"
