# APEX Observability Setup Guide

Complete guide to setting up Prometheus, Grafana, and Alertmanager for monitoring APEX.

## Table of Contents

1. [Quick Start (Docker)](#quick-start-docker)
2. [Architecture Overview](#architecture-overview)
3. [Manual Setup](#manual-setup)
4. [Available Metrics](#available-metrics)
5. [Grafana Dashboards](#grafana-dashboards)
6. [Alert Configuration](#alert-configuration)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start (Docker)

The fastest way to get started with monitoring:

```bash
# 1. Start APEX with metrics enabled (default port 8000)
python orchestrator.py --env dev --metrics-port 8000

# 2. In another terminal, start the observability stack
docker-compose -f docker-compose.observability.yml up -d

# 3. Access the services
#    - APEX Metrics:  http://localhost:8000/metrics
#    - Prometheus:    http://localhost:9090
#    - Alertmanager:  http://localhost:9093
#    - Grafana:       http://localhost:3000 (admin/admin)
```

### Stop the Stack

```bash
docker-compose -f docker-compose.observability.yml down

# To also remove data volumes:
docker-compose -f docker-compose.observability.yml down -v
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         APEX Application                         │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  MetricsManager (OpenTelemetry + Prometheus Exporter)       ││
│  │  ├── RiskMetrics     (Greeks, P&L, breaches)                ││
│  │  ├── HealthMetrics   (connections, coverage, queues)        ││
│  │  └── AdapterMetrics  (throughput, latency)                  ││
│  └─────────────────────────────────────────────────────────────┘│
│                            │                                     │
│                   :8000/metrics                                  │
└────────────────────────────┼─────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   Prometheus    │ ← Scrapes every 5s
                    │   :9090         │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
     ┌────────▼────────┐         ┌──────────▼──────────┐
     │  Alertmanager   │         │      Grafana        │
     │  :9093          │         │      :3000          │
     └────────┬────────┘         └─────────────────────┘
              │
     ┌────────▼────────┐
     │  Notifications  │
     │  (Slack/Email)  │
     └─────────────────┘
```

### Data Flow

1. **APEX** exposes metrics on `:8000/metrics` (Prometheus format)
2. **Prometheus** scrapes metrics every 5 seconds
3. **Prometheus** evaluates alert rules continuously
4. **Alertmanager** receives fired alerts and routes to channels
5. **Grafana** queries Prometheus for visualization

---

## Manual Setup

If you prefer to run without Docker:

### 1. Install Dependencies

```bash
# Install observability extras for APEX
uv pip install -e ".[observability]"

# Or with pip
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-prometheus prometheus-client
```

### 2. Install Prometheus

**macOS:**
```bash
brew install prometheus
```

**Linux:**
```bash
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
tar xvfz prometheus-2.45.0.linux-amd64.tar.gz
cd prometheus-2.45.0.linux-amd64
```

**Configuration:**
Create `prometheus.yml`:
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['localhost:9093']

rule_files:
  - "alerts/*.yml"

scrape_configs:
  - job_name: 'apex'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 5s
```

**Start Prometheus:**
```bash
prometheus --config.file=prometheus.yml
```

### 3. Install Grafana

**macOS:**
```bash
brew install grafana
brew services start grafana
```

**Linux:**
```bash
sudo apt-get install -y grafana
sudo systemctl start grafana-server
```

Access at http://localhost:3000 (default: admin/admin)

### 4. Install Alertmanager (Optional)

**macOS:**
```bash
brew install alertmanager
```

**Linux:**
```bash
wget https://github.com/prometheus/alertmanager/releases/download/v0.25.0/alertmanager-0.25.0.linux-amd64.tar.gz
tar xvfz alertmanager-0.25.0.linux-amd64.tar.gz
```

---

## Available Metrics

### Risk Metrics (`apex_*`)

| Metric | Type | Description |
|--------|------|-------------|
| `apex_risk_breach_level` | Gauge | 0=OK, 1=soft breach, 2=hard breach |
| `apex_portfolio_delta` | Gauge | Total portfolio delta |
| `apex_portfolio_gamma` | Gauge | Total portfolio gamma |
| `apex_portfolio_vega` | Gauge | Total portfolio vega |
| `apex_portfolio_theta` | Gauge | Total portfolio theta |
| `apex_unrealized_pnl` | Gauge | Unrealized P&L (USD) |
| `apex_daily_pnl` | Gauge | Daily P&L (USD) |
| `apex_gross_notional` | Gauge | Total gross notional |
| `apex_net_notional` | Gauge | Total net notional |
| `apex_max_single_name_pct` | Gauge | Max single-name concentration % |
| `apex_near_term_gamma_notional` | Gauge | 0-7 DTE gamma notional |
| `apex_near_term_vega_notional` | Gauge | 0-30 DTE vega notional |
| `apex_margin_utilization` | Gauge | Margin utilization % |
| `apex_buying_power` | Gauge | Available buying power |
| `apex_net_liquidation` | Gauge | Net liquidation value |
| `apex_total_positions` | Gauge | Total position count |
| `apex_positions_missing_md` | Gauge | Positions without market data |
| `apex_positions_missing_greeks` | Gauge | Positions without Greeks |
| `apex_risk_calc_duration_ms` | Histogram | Risk calculation latency |
| `apex_snapshot_build_duration_ms` | Histogram | Snapshot build latency |
| `apex_risk_breach_total` | Counter | Total breaches by rule/level |
| `apex_last_snapshot_timestamp` | Gauge | Last snapshot Unix timestamp |

### Health Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `apex_broker_connected` | Gauge | 1=connected, 0=disconnected |
| `apex_adapter_health` | Gauge | 1=healthy, 0=unhealthy, -1=degraded |
| `apex_market_data_coverage` | Gauge | Coverage ratio (0.0-1.0) |
| `apex_last_tick_timestamp` | Gauge | Last tick time per symbol |
| `apex_last_any_tick_timestamp` | Gauge | Last tick time (any symbol) |
| `apex_event_bus_queue_size` | Gauge | Event queue depth by lane |
| `apex_system_ready` | Gauge | 1=ready, 0=not ready |
| `apex_startup_duration_seconds` | Gauge | Time to first snapshot |
| `apex_tick_to_store_ms` | Histogram | Tick processing latency |
| `apex_event_process_duration_ms` | Histogram | Event processing latency |
| `apex_eventbus_slow_max_gap_ms` | Gauge | Max slow lane dispatch gap |
| `apex_connection_attempts_total` | Counter | Connection attempts |
| `apex_connection_failures_total` | Counter | Connection failures |
| `apex_reconnection_total` | Counter | Reconnection events |

---

## Grafana Dashboards

### Add Prometheus Data Source

1. Open Grafana: http://localhost:3000
2. Go to **Configuration** → **Data Sources**
3. Click **Add data source**
4. Select **Prometheus**
5. Set URL: `http://prometheus:9090` (Docker) or `http://localhost:9090` (local)
6. Click **Save & Test**

### Create Risk Overview Dashboard

1. Click **+** → **Dashboard** → **Add visualization**
2. Select **Prometheus** data source
3. Add panels using the queries below

### Example Panel Queries

**Portfolio P&L (Time Series):**
```promql
apex_unrealized_pnl
apex_daily_pnl
```

**Portfolio Greeks (Stat Panel):**
```promql
apex_portfolio_delta
apex_portfolio_gamma
apex_portfolio_vega
apex_portfolio_theta
```

**Breach Level (Gauge):**
```promql
max(apex_risk_breach_level)
```
- Thresholds: 0=green, 1=yellow, 2=red

**Broker Status (Stat):**
```promql
apex_broker_connected{broker="ib"}
apex_broker_connected{broker="futu"}
```

**Market Data Coverage (Gauge):**
```promql
apex_market_data_coverage * 100
```
- Thresholds: 0-70=red, 70-90=yellow, 90-100=green

**Snapshot Latency (Time Series):**
```promql
histogram_quantile(0.50, rate(apex_snapshot_build_duration_ms_bucket[5m]))
histogram_quantile(0.95, rate(apex_snapshot_build_duration_ms_bucket[5m]))
histogram_quantile(0.99, rate(apex_snapshot_build_duration_ms_bucket[5m]))
```

**Position Count (Stat):**
```promql
apex_total_positions
```

**Margin Utilization (Gauge):**
```promql
apex_margin_utilization
```
- Thresholds: 0-60=green, 60-80=yellow, 80-100=red

### Import Pre-built Dashboard

Create a file `apex_dashboard.json` and import it:

```json
{
  "dashboard": {
    "title": "APEX Risk Monitor",
    "panels": [
      {
        "title": "Portfolio P&L",
        "type": "timeseries",
        "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
        "targets": [
          {"expr": "apex_unrealized_pnl", "legendFormat": "Unrealized"},
          {"expr": "apex_daily_pnl", "legendFormat": "Daily"}
        ]
      },
      {
        "title": "Delta",
        "type": "stat",
        "gridPos": {"x": 12, "y": 0, "w": 3, "h": 4},
        "targets": [{"expr": "apex_portfolio_delta"}]
      },
      {
        "title": "Gamma",
        "type": "stat",
        "gridPos": {"x": 15, "y": 0, "w": 3, "h": 4},
        "targets": [{"expr": "apex_portfolio_gamma"}]
      },
      {
        "title": "Vega",
        "type": "stat",
        "gridPos": {"x": 18, "y": 0, "w": 3, "h": 4},
        "targets": [{"expr": "apex_portfolio_vega"}]
      },
      {
        "title": "Theta",
        "type": "stat",
        "gridPos": {"x": 21, "y": 0, "w": 3, "h": 4},
        "targets": [{"expr": "apex_portfolio_theta"}]
      },
      {
        "title": "Breach Level",
        "type": "gauge",
        "gridPos": {"x": 12, "y": 4, "w": 6, "h": 4},
        "targets": [{"expr": "max(apex_risk_breach_level)"}],
        "fieldConfig": {
          "defaults": {
            "max": 2,
            "thresholds": {
              "steps": [
                {"color": "green", "value": 0},
                {"color": "yellow", "value": 1},
                {"color": "red", "value": 2}
              ]
            }
          }
        }
      },
      {
        "title": "Market Data Coverage",
        "type": "gauge",
        "gridPos": {"x": 18, "y": 4, "w": 6, "h": 4},
        "targets": [{"expr": "apex_market_data_coverage * 100"}],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "max": 100,
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "yellow", "value": 70},
                {"color": "green", "value": 90}
              ]
            }
          }
        }
      }
    ]
  }
}
```

To import:
1. Go to **Dashboards** → **Import**
2. Paste the JSON or upload the file
3. Select Prometheus data source
4. Click **Import**

---

## Alert Configuration

### Built-in Alert Rules

The project includes pre-configured alerts in `config/prometheus/alerts/apex_risk.yml`:

| Alert | Severity | Condition |
|-------|----------|-----------|
| `RiskHardBreach` | critical | `apex_risk_breach_level == 2` (immediate) |
| `BrokerDisconnected` | critical | `apex_broker_connected == 0` for 10s |
| `RiskSoftBreach` | warning | `apex_risk_breach_level == 1` for 1m |
| `MarketDataStale` | warning | No tick for 30s |
| `LowMarketDataCoverage` | warning | Coverage < 80% for 1m |
| `SystemNotReady` | warning | Not ready for 2m |
| `HighMarginUtilization` | warning | Margin > 80% for 5m |
| `SlowRiskCalculation` | warning | p95 > 500ms |
| `HighEventQueueDepth` | warning | Queue > 1000 |
| `ManyPositionsMissingData` | info | > 5 missing data |
| `HighConcentration` | info | > 25% single name |

### Configure Slack Notifications

Edit `config/prometheus/alertmanager.yml`:

```yaml
global:
  slack_api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'

receivers:
  - name: 'critical'
    slack_configs:
      - channel: '#apex-alerts'
        title: '{{ .CommonAnnotations.summary }}'
        text: '{{ .CommonAnnotations.description }}'
        send_resolved: true
        color: '{{ if eq .Status "firing" }}danger{{ else }}good{{ end }}'

  - name: 'warning'
    slack_configs:
      - channel: '#apex-alerts'
        title: '{{ .CommonAnnotations.summary }}'
        text: '{{ .CommonAnnotations.description }}'
        send_resolved: true
        color: '{{ if eq .Status "firing" }}warning{{ else }}good{{ end }}'
```

### Configure Email Notifications

```yaml
global:
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'apex-alerts@yourdomain.com'
  smtp_auth_username: 'your-email@gmail.com'
  smtp_auth_password: 'your-app-password'

receivers:
  - name: 'critical'
    email_configs:
      - to: 'team@yourdomain.com'
        send_resolved: true
```

### Configure PagerDuty

```yaml
receivers:
  - name: 'critical'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
        severity: critical
```

### Add Custom Alerts

Create new rules in `config/prometheus/alerts/custom.yml`:

```yaml
groups:
  - name: custom_alerts
    rules:
      # Alert when daily loss exceeds threshold
      - alert: DailyLossThreshold
        expr: apex_daily_pnl < -10000
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Daily loss exceeds $10,000"
          description: "Daily P&L is {{ $value | printf \"$%.2f\" }}. Review positions."

      # Alert when delta exceeds limit
      - alert: DeltaLimitWarning
        expr: abs(apex_portfolio_delta) > 40000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Portfolio delta approaching limit"
          description: "Delta is {{ $value }}. Limit is 50,000."
```

---

## Troubleshooting

### Metrics Not Showing

**Problem:** `http://localhost:8000/metrics` returns error

**Solutions:**
1. Check APEX is running with metrics enabled:
   ```bash
   python orchestrator.py --metrics-port 8000
   ```
2. Check observability is installed:
   ```bash
   pip install -e ".[observability]"
   ```
3. Check port is not in use:
   ```bash
   lsof -i :8000
   ```

### Prometheus Can't Scrape

**Problem:** Target shows "DOWN" in Prometheus

**Solutions:**
1. Check network connectivity:
   ```bash
   curl http://localhost:8000/metrics
   ```
2. For Docker, use `host.docker.internal:8000` in prometheus.yml
3. Check firewall settings

### Grafana No Data

**Problem:** Panels show "No data"

**Solutions:**
1. Verify Prometheus data source is configured correctly
2. Test query in Prometheus UI first: http://localhost:9090
3. Check time range in Grafana (default may be too narrow)
4. Verify APEX has been running and producing metrics

### Alerts Not Firing

**Problem:** Alerts not triggering despite conditions met

**Solutions:**
1. Check Prometheus alert status: http://localhost:9090/alerts
2. Verify alert rules loaded: http://localhost:9090/rules
3. Check Alertmanager status: http://localhost:9093/#/alerts
4. Review alertmanager.yml for routing issues

### High Memory Usage

**Problem:** Prometheus using too much memory

**Solutions:**
1. Reduce retention period in prometheus.yml:
   ```yaml
   command:
     - '--storage.tsdb.retention.time=7d'
   ```
2. Reduce scrape frequency for less critical metrics
3. Add recording rules to pre-aggregate common queries

---

## Quick Reference

### URLs

| Service | URL | Default Credentials |
|---------|-----|---------------------|
| APEX Metrics | http://localhost:8000/metrics | - |
| Prometheus | http://localhost:9090 | - |
| Alertmanager | http://localhost:9093 | - |
| Grafana | http://localhost:3000 | admin/admin |

### Docker Commands

```bash
# Start stack
docker-compose -f docker-compose.observability.yml up -d

# View logs
docker-compose -f docker-compose.observability.yml logs -f prometheus
docker-compose -f docker-compose.observability.yml logs -f grafana

# Restart single service
docker-compose -f docker-compose.observability.yml restart prometheus

# Stop stack
docker-compose -f docker-compose.observability.yml down

# Remove all data
docker-compose -f docker-compose.observability.yml down -v
```

### Useful PromQL Queries

```promql
# Current breach status
max(apex_risk_breach_level)

# Positions by status
apex_total_positions - apex_positions_missing_md

# 5-minute average P&L change
rate(apex_daily_pnl[5m])

# Broker uptime (last hour)
avg_over_time(apex_broker_connected{broker="ib"}[1h]) * 100

# Snapshot latency percentiles
histogram_quantile(0.95, rate(apex_snapshot_build_duration_ms_bucket[5m]))

# Events per second
rate(apex_connection_attempts_total[1m])
```

---

## Related Documentation

- [USER_MANUAL.md](USER_MANUAL.md) - Main user guide
- [PERSISTENCE_LAYER.md](PERSISTENCE_LAYER.md) - Database setup
- [README.md](../README.md) - Project overview
