# Apex lake MCP — operator guide

A read-only [MCP](https://modelcontextprotocol.io) server over the Livewire lake, for
Claude Code (direct, over the tailnet) and ChatGPT (through OpenAI's Secure MCP
Tunnel). It is a separate process from the REST API and uses the same lake queries, so
a tool and its REST twin return the same data. Source: `src/mcp_server/`.

It does not write to the lake, connect to PostgreSQL, start streaming, or serve
indicators, signals, options, or the ledger.

## Tools (20)

| Group | Tool | REST twin |
|---|---|---|
| discovery | `list_asset_classes` | `GET /v1/lake/asset-classes` |
| | `search_instruments` | `GET /v1/instruments` (same `listing` and 1..5000 `limit`; default 100, not 500) |
| | `get_instrument` | `GET /v1/{asset_class}/{symbol}` |
| | `get_coverage` | `GET /v1/lake/coverage` |
| | `find_gaps` | `GET /v1/{asset_class}/{symbol}/gaps` |
| | `get_lake_status` | `GET /v1/lake/status` |
| | `list_futures_contracts` | `GET /v1/futures/{root}/contracts` |
| series | `get_bars` | `GET /v1/{asset_class}/{symbol}/bars` |
| | `get_bulk_bars` | `GET /v1/equity/bars` |
| | `get_rate_series` | `GET /v1/rates/{symbol}/series` |
| identity | `get_corporate_actions` | `GET /v1/equity/{symbol}/actions` |
| | `get_delisting` | `GET /v1/equity/{symbol}/delisting` |
| | `resolve_security` | `GET /v1/security/{symbol}` |
| | `list_indices` | `GET /v1/membership/indices` |
| | `get_index_members` | `GET /v1/membership/{index_id}` |
| | `get_membership_history` | `GET /v1/membership/history` |
| revisions | `list_silver_revisions` | `GET /v1/lake/silver-revisions` |
| | `get_silver_revision` | `GET /v1/lake/silver-revisions/{n}` (omit `revision` for current) |
| | `list_pit_revisions` | `GET /v1/lake/pit-revisions` |
| | `get_pit_revision` | `GET /v1/lake/pit-revisions/{n}` (always explicit) |

Every tool is marked read-only and idempotent.

**How results differ from REST:**
- **Series format.** Bars, bulk bars, and yields come back as `columns` + `rows`, oldest first.
- **Row limits.** Every series is capped:
  - bars: default 250 rows, at most 5000;
  - bulk bars: 1–200 symbols, default 50 rows each, at most 2000 each, and symbols × limit ≤ 10000;
  - yields: default 500 points.
  
  `truncated: true` means older rows exist; narrow the window or lower the limit.
- **Pagination.** List tools always page: `limit` defaults to 100 (max 2000) and `offset` to 0. The response includes `returned`, `truncated`, and `next_offset`.
- **Size budget.** A result larger than 2 MiB fails with `result_too_large` instead of being cut off.

Errors come back as a tool result with `isError: true`. Its text is exactly the REST
error envelope, with no prefix, so a client can `json.loads` it:
`{"error": {"code", "message", "symbol"?, "asset_class"?, "details"?}}`. The codes are
the same as REST's, for example `unknown_symbol`, `invalid_parameter`,
`unknown_revision`, `pit_unavailable`, and `query_timeout`.

A few failures come from the MCP layer itself rather than the lake:
- **Invalid arguments** (a wrong type or a malformed timestamp, rejected by the tool's
  input schema): `invalid_parameter` with `details.source = "arguments"` and a
  `problems` list. This is REST's 422 class. An out-of-range `limit` or `offset` on a
  list tool is the lake's own `invalid_parameter` (REST's 400), exactly as on REST.
- **Unknown tool:** `invalid_parameter` with `details.source = "tool"`.
- **Result over the 2 MiB budget:** `result_too_large`, which is MCP only.

**Every call has a deadline.** A call gets `APEX_MCP_CALL_TIMEOUT_SECONDS` (default 60)
in total. When it runs out, the call is cancelled and returns `query_timeout`: any lake
read still in flight is interrupted, while a catalog lookup already running in a worker
thread is left to finish on its own. This cap covers the whole call. The separate
per-query lake deadline, `APEX_LAKE_QUERY_TIMEOUT_SECONDS`, bounds only a single parquet
read, and some calls make many reads (bulk bars, futures contracts) or none (status,
catalog).

The deadline matters because the server runs stateless over HTTP, and with mcp 2.2.0
a client that times out or disconnects does not cancel the call on the server. That
behavior is kept visible as a narrow `xfail(strict=True)` in
`tests/unit/mcp_server/test_transport.py`.

## Configuration

The MCP server reads the same lake variables as REST (`APEX_LIVEWIRE_ROOT`,
`APEX_LIVEWIRE_SILVER_ROOT`, `APEX_LIVEWIRE_PRICE_MODE`, `APEX_LIVEWIRE_COVERAGE_DB`,
`APEX_LIVEWIRE_LAKE_ROOT`, `APEX_LIVEWIRE_DELISTED_ROOT`, `APEX_LIVEWIRE_REPAIRS_ROOT`,
`APEX_LAKE_QUERY_TIMEOUT_SECONDS`). A lake variable left unset disables only the tools
that need it. MCP-specific settings:

| Variable | Default | Meaning |
|---|---|---|
| `APEX_MCP_API_KEY` | none — **required** | Bearer key for every request except `/healthz`. The server will not start without it; there is no unauthenticated mode. |
| `APEX_MCP_HOST` | `127.0.0.1` | Listen address. |
| `APEX_MCP_PORT` | `8333` | Listen port. |
| `APEX_MCP_CALL_TIMEOUT_SECONDS` | `60` | Deadline for one whole tool call (see above). |
| `APEX_MCP_ALLOWED_HOSTS` | `127.0.0.1:<port>,localhost:<port>` | Comma-separated `Host` header values that clients may send. Anything else gets 421. An `Origin` header, if present, must be `http(s)://` followed by one of these values, or the request gets 403. |

Generate a key with `python -c "import secrets; print(secrets.token_urlsafe(32))"`.

## Run locally

```sh
APEX_MCP_API_KEY=<key> APEX_LIVEWIRE_ROOT=<lake>/bronze ... make mcp-server
```

## Deploy (compose)

The `mcp` service in `docker-compose.yml` runs the API image with
`python -m src.mcp_server.server`. It needs two private files next to the compose file,
and neither is committed:

- `.env` — shared with the API service for compose interpolation. It must contain:
  - `APEX_LAKE_HOST_ROOT` — the host lake root;
  - `APEX_MCP_BIND` — the host's tailnet address. Only this address gets the published
    port, never `0.0.0.0`.
- `mcp.env` — the only file the MCP container reads. It must contain:
  - `APEX_MCP_API_KEY`;
  - `APEX_MCP_ALLOWED_HOSTS` — the tailnet name and/or address with `:8333`.

The MCP container never reads the API's `.env`, so it never receives the PostgreSQL
credentials. Lake mounts are read-only.

Check both of these after a deploy:

- **Liveness:** `curl http://<bind>:8333/healthz` returns `{"status":"ok"}`. It
  requires no auth and reads no data.
- **Readiness:** an authenticated tool call. Any other HTTP response, including a 405
  or a 200 from `/healthz`, does not show that the tools work:

```sh
curl -s http://<bind>:8333/mcp \
  -H "Authorization: Bearer <key>" \
  -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_lake_status","arguments":{}}}'
```

## Claude Code

```sh
claude mcp add --transport http apex-lake http://<tailnet-name>:8333/mcp \
  --header "Authorization: Bearer <key>"
```

`<tailnet-name>:8333` must be listed in `APEX_MCP_ALLOWED_HOSTS`. Keep the key out of
project-scoped config that gets committed; the default `local` scope is private.

## ChatGPT (Secure MCP Tunnel)

ChatGPT reaches the server through OpenAI's
[Secure MCP Tunnel](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels):
a tunnel client on the host makes an outbound HTTPS connection, and nothing is exposed
publicly. Follow that guide for installing and associating the tunnel client.

Before going live, record the following in the private runbook:
- the tunnel client version you installed;
- how it forwards the `Authorization: Bearer` header to this server;
- the `Host` header it sends, which must be added to `APEX_MCP_ALLOWED_HOSTS`.

Workspace developer mode and the tunnel's organization permission are separate
prerequisites.

Acceptance requires a successful `tools/call` from ChatGPT. Seeing the connector in the
list, or getting any response other than 401, does not count.

## What is and is not verified

This repository verifies the following:
- the 20-tool set, schemas, and read-only annotations;
- REST parity on real frozen fixtures;
- auth, Host, and Origin rejection;
- real-socket transport, shutdown, and cancellation.

The following are operator acceptance steps per deployment and are not verified here:
- the running image digest;
- a direct tailnet call from Claude Code;
- a ChatGPT `tools/call` through the tunnel.
