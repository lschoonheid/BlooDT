#!/usr/bin/env bash
# Set up the repo with already-trained models and (optionally) bring up the live stack.
#
#   ./setup.sh                    # python env, models/sixth, cases 17 and 45, tests
#   ./setup.sh --stack            # ... and Docker + the live stack (Grafana on :3000)
#   ./setup.sh --stack --replay   # ... and replay the first case through it
#
# Options:
#   --run NAME       training run under models/ (default: sixth). If models/NAME is missing but
#                    models/weights_NAME.zip exists, it is unzipped.
#   --cases "A B"    VitalDB cases to download (default: "17 45"; 45 is used by the tests)
#   --stack          install Docker tooling if needed (macOS: Homebrew compose + Colima), start the stack
#   --replay         replay the first of --cases through the stack (implies --stack)
#   --skip-tests     don't run pytest
set -euo pipefail
cd "$(dirname "$0")"

RUN=sixth
CASES="17 45"
STACK=0
REPLAY=0
TESTS=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN="$2"; shift 2 ;;
    --cases) CASES="$2"; shift 2 ;;
    --stack) STACK=1; shift ;;
    --replay) STACK=1; REPLAY=1; shift ;;
    --skip-tests) TESTS=0; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# --- Python environment ---------------------------------------------------------------------
step "Python environment (uv sync)"
command -v git >/dev/null || die "git is needed (PAS is installed from GitHub)"
if ! command -v uv >/dev/null; then
  echo "uv not found; installing it"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync

# --- Models ---------------------------------------------------------------------------------
step "Models: models/$RUN"
if [[ ! -f "models/$RUN/models.joblib" ]]; then
  if [[ -f "models/weights_$RUN.zip" ]]; then
    echo "unzipping models/weights_$RUN.zip"
    mkdir -p "models/$RUN"
    unzip -o -q "models/weights_$RUN.zip" -d "models/$RUN"
  else
    die "models/$RUN/models.joblib not found; copy the run folder to models/$RUN or put weights_$RUN.zip in models/"
  fi
fi
for f in models.joblib config.json; do
  [[ -f "models/$RUN/$f" ]] || die "models/$RUN/$f missing"
done
[[ -L "models/$RUN" ]] && echo "note: models/$RUN is a symlink; Docker can't follow links outside ./models"
echo "ok: $(ls "models/$RUN" | tr '\n' ' ')"

# --- Data -----------------------------------------------------------------------------------
step "VitalDB cases: $CASES"
# shellcheck disable=SC2086
uv run vitaldb-cache download $CASES

# --- Tests ----------------------------------------------------------------------------------
if [[ $TESTS == 1 ]]; then
  step "Tests"
  uv run pytest -q
fi

[[ $STACK == 1 ]] || {
  step "Done"
  echo "Replay a case without Docker:  uv run dte replay ${CASES%% *} --run $RUN"
  echo "Live stack + dashboard:        ./setup.sh --stack --replay"
  exit 0
}

# --- Docker ---------------------------------------------------------------------------------
step "Docker"
command -v docker >/dev/null || {
  [[ "$(uname)" == Darwin ]] && command -v brew >/dev/null || die "install Docker (with compose) first"
  brew install docker
}
if ! docker compose version >/dev/null 2>&1; then
  [[ "$(uname)" == Darwin ]] && command -v brew >/dev/null || die "the docker compose plugin is missing; install it"
  echo "installing the compose plugin"
  brew install docker-compose
  # let the docker CLI find Homebrew's plugin
  cfg="$HOME/.docker/config.json"
  plugins="$(brew --prefix)/lib/docker/cli-plugins"
  mkdir -p "$HOME/.docker"
  if [[ ! -s "$cfg" ]]; then
    printf '{"cliPluginsExtraDirs": ["%s"]}\n' "$plugins" > "$cfg"
  elif ! grep -q cliPluginsExtraDirs "$cfg"; then
    die "add \"cliPluginsExtraDirs\": [\"$plugins\"] to $cfg, then rerun"
  fi
  docker compose version >/dev/null || die "docker compose still not found"
fi
if ! docker info >/dev/null 2>&1; then
  if [[ "$(uname)" == Darwin ]]; then
    if [[ -d /Applications/Docker.app ]]; then
      echo "starting Docker Desktop"
      open -a Docker
    else
      command -v colima >/dev/null || brew install colima
      echo "starting Colima"
      colima start --cpu 4 --memory 8
    fi
    for _ in $(seq 60); do docker info >/dev/null 2>&1 && break; sleep 2; done
  fi
  docker info >/dev/null 2>&1 || die "the Docker engine isn't running"
fi
echo "ok: $(docker compose version)"

# --- Live stack -----------------------------------------------------------------------------
step "Live stack (RUN=$RUN)"
export RUN
docker compose up -d --build
echo "Grafana: http://localhost:3000 (admin/admin), dashboard \"Anaesthesia twin\""

if [[ $REPLAY == 1 ]]; then
  case_id="${CASES%% *}"
  step "Replaying case $case_id at 30x (Ctrl-C stops the replay; the stack keeps running)"
  docker compose run --rm replayer "$case_id"
else
  echo "Replay a case:  RUN=$RUN docker compose run --rm replayer ${CASES%% *}"
fi
echo "Stop the stack: docker compose down"
