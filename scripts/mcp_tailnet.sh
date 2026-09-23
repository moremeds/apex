#!/usr/bin/env bash
# Run the lake MCP on this Docker host, bound to loopback, and expose it to the tailnet
# only (docs/mcp-operator-guide.md). The same steps serve a test instance of a candidate
# image and the production service; they differ in IMAGE, PORT and the env file.
#
#   scripts/mcp_tailnet.sh up IMAGE [PORT]   start or replace the service, add the
#                                            tailnet forward, then run check
#   scripts/mcp_tailnet.sh check [PORT]      liveness, 401 without the key, then an
#                                            authenticated tools/list and tools/call
#   scripts/mcp_tailnet.sh down [PORT]       stop the service, remove the forward
#
# PORT defaults to 8333 (production); use another, e.g. 8334, for a test instance.
# APEX_LAKE_HOST_ROOT is required for up. APEX_MCP_ENV_FILE defaults to
# ~/.config/apex-mcp/<PORT>.env; if it does not exist, up creates it (mode 0600) with a
# fresh key and this host's tailnet name and address as the allowed Host values.
# Needs docker-compose (or the docker compose plugin), tailscale, curl and python3.
set -euo pipefail

cmd=${1:?usage: up IMAGE [PORT] | check [PORT] | down [PORT]}
shift
if [ "$cmd" = up ]; then
    image=${1:?up needs IMAGE}
    shift
fi
port=${1:-8333}
compose_file=$(cd "$(dirname "$0")/.." && pwd)/docker/mcp.compose.yml
env_file=${APEX_MCP_ENV_FILE:-$HOME/.config/apex-mcp/$port.env}
project=apex-mcp-$port
name=$(tailscale status --self --json 2>/dev/null |
    python3 -c 'import json, sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))')
ip=$(tailscale ip -4 2>/dev/null | head -1)
url=http://$name:$port

dc() {
    if command -v docker-compose >/dev/null; then docker-compose "$@"; else docker compose "$@"; fi
}

check() {
    local key
    curl -fsS -m 10 "$url/healthz" | grep -q '"ok"' || { echo "FAIL healthz $url" >&2; return 1; }
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 10 -X POST "$url/mcp")" = 401 ] ||
        { echo "FAIL a request without the key was not 401" >&2; return 1; }
    key=$(grep '^APEX_MCP_API_KEY=' "$env_file" | cut -d= -f2-)
    rpc() {
        curl -fsS -m 60 "$url/mcp" -H "Authorization: Bearer $key" \
            -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
            -d "$1"
    }
    rpc '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python3 -c '
import json, sys
n = len(json.load(sys.stdin)["result"]["tools"])
assert n == 20, f"tools/list returned {n} tools, expected 20"
print("tools/list: 20 tools")'
    rpc '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_lake_status","arguments":{}}}' |
        python3 -c '
import json, sys
result = json.load(sys.stdin)["result"]
assert not result.get("isError"), result["content"][0]["text"]
print("tools/call get_lake_status: ok")'
    echo "OK $url/mcp (key: $env_file)"
}

case $cmd in
up)
    : "${APEX_LAKE_HOST_ROOT:?set the host lake root}"
    if [ ! -f "$env_file" ]; then
        mkdir -p "$(dirname "$env_file")"
        (
            umask 077
            printf 'APEX_MCP_API_KEY=%s\nAPEX_MCP_ALLOWED_HOSTS=%s\n' \
                "$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')" \
                "$name:$port,$ip:$port,127.0.0.1:$port" >"$env_file"
        )
        echo "created $env_file"
    fi
    APEX_MCP_IMAGE=$image APEX_MCP_HOST_PORT=$port APEX_MCP_ENV_FILE=$env_file \
        dc -p "$project" -f "$compose_file" up -d
    tailscale serve --bg --tcp "$port" "tcp://127.0.0.1:$port" >/dev/null
    for _ in $(seq 30); do
        curl -fsS -m 2 "$url/healthz" >/dev/null 2>&1 && break
        sleep 2
    done
    check
    ;;
check) check ;;
down)
    dc -p "$project" down
    tailscale serve --tcp="$port" off >/dev/null 2>&1 || true
    ;;
*)
    echo "unknown command $cmd" >&2
    exit 2
    ;;
esac
