#!/usr/bin/env bash
#
# serve.sh — launch the apex streaming-TA API server.
#
# Why this exists: the app reads os.environ only (src/api/server.py) and does NOT
# auto-load .env. This script loads .env into the environment, runs a few non-fatal
# preflight checks, prints what apex will actually see, then execs the server.
#
# Usage:
#   scripts/serve.sh                 # load .env, preflight, run on APEX_API_PORT (default 8322)
#   scripts/serve.sh -p 9001         # override port
#   scripts/serve.sh --no-preflight  # skip lake/PG reachability checks
#   APEX_ENV_FILE=.env.prod scripts/serve.sh
#
# Precedence: real environment variables win over .env (export APEX_PG_URL=... overrides
# the file). Values may use a leading ~ (expanded to $HOME) even though uv/Path would not.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${APEX_ENV_FILE:-$ROOT/.env}"

# --- pretty output (only colorize on a tty) -----------------------------------
if [ -t 1 ]; then C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'; C_DIM=$'\033[2m'; C_RST=$'\033[0m'
else C_OK=""; C_WARN=""; C_ERR=""; C_DIM=""; C_RST=""; fi
ok()   { printf '%s  ok %s %s\n'   "$C_OK"   "$C_RST" "$*"; }
warn() { printf '%swarn %s %s\n'   "$C_WARN" "$C_RST" "$*"; }
die()  { printf '%serror%s %s\n'   "$C_ERR"  "$C_RST" "$*" >&2; exit 1; }

usage() { awk 'NR>=3 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"; }

# --- args ---------------------------------------------------------------------
PORT_OVERRIDE=""
DO_PREFLIGHT=1
while [ $# -gt 0 ]; do
  case "$1" in
    -p|--port)      PORT_OVERRIDE="${2:?--port needs a value}"; shift 2 ;;
    --no-preflight) DO_PREFLIGHT=0; shift ;;
    -h|--help)      usage; exit 0 ;;
    *)              die "unknown argument: $1 (try --help)" ;;
  esac
done

# --- load .env (real env wins; expand leading ~) ------------------------------
load_env_file() {
  local file="$1" line key val
  [ -f "$file" ] || { warn "no env file at $file — booting with whatever is already in the environment"; return 0; }
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line#"${line%%[![:space:]]*}"}"        # ltrim
    [ -z "$line" ] && continue
    case "$line" in \#*) continue ;; esac          # comment
    line="${line#export }"
    case "$line" in *=*) ;; *) continue ;; esac     # must be KEY=VALUE
    key="${line%%=*}"; key="${key//[[:space:]]/}"
    val="${line#*=}"
    case "$val" in                                   # strip surrounding quotes
      \"*\") val="${val#\"}"; val="${val%\"}" ;;
      \'*\') val="${val#\'}"; val="${val%\'}" ;;
    esac
    case "$val" in                                   # expand leading ~
      "~")    val="$HOME" ;;
      "~/"*)  val="$HOME/${val#\~/}" ;;
    esac
    if [ -z "${!key:-}" ]; then export "$key=$val"; fi
  done < "$file"
  ok "loaded env file: $file"
}

mask_dsn() { printf '%s' "$1" | sed -E 's#(://[^:/@]+:)[^@]*@#\1***@#'; }

# --- preflight (non-fatal except missing uv) ----------------------------------
preflight() {
  command -v uv >/dev/null 2>&1 || die "uv not found on PATH (install uv, or run: pipx install uv)"

  if [ -n "${APEX_LIVEWIRE_ROOT:-}" ]; then
    if [ -d "$APEX_LIVEWIRE_ROOT" ]; then ok "livewire lake: $APEX_LIVEWIRE_ROOT"
    else warn "APEX_LIVEWIRE_ROOT does not exist: $APEX_LIVEWIRE_ROOT  → /bars,/indicators return 503"; fi
  else
    warn "APEX_LIVEWIRE_ROOT unset → /bars,/indicators return 503, no warmup/live stream"
  fi

  if [ -n "${APEX_PG_URL:-}" ]; then
    if command -v pg_isready >/dev/null 2>&1; then
      if pg_isready -d "$APEX_PG_URL" >/dev/null 2>&1; then ok "postgres: reachable  ($(mask_dsn "$APEX_PG_URL"))"
      else warn "APEX_PG_URL set but unreachable  ($(mask_dsn "$APEX_PG_URL"))  → /signals,/confluence return 503"; fi
    else
      ok "APEX_PG_URL set  ($(mask_dsn "$APEX_PG_URL"))  [pg_isready absent; skipped reachability]"
    fi
  else
    warn "APEX_PG_URL unset → /signals,/confluence return 503, persistence off"
  fi
}

# --- run ----------------------------------------------------------------------
cd "$ROOT"
load_env_file "$ENV_FILE"
[ -n "$PORT_OVERRIDE" ] && export APEX_API_PORT="$PORT_OVERRIDE"
export APEX_API_PORT="${APEX_API_PORT:-8322}"

[ "$DO_PREFLIGHT" -eq 1 ] && preflight

printf '%s' "$C_DIM"
printf 'apex → http://127.0.0.1:%s   (health: /health)\n' "$APEX_API_PORT"
printf 'xenon ticks: %s\n' "${APEX_XENON_WS_URL:-ws://127.0.0.1:8765 (default)}"
printf 'timeframes : %s\n' "${APEX_TIMEFRAMES:-1d (default)}"
printf '%s\n' "$C_RST"

exec uv run python -m src.api.server
