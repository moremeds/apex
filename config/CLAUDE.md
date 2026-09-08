# config/ — Configuration files

Root `CLAUDE.md` is authoritative for policy.

`config/` is an importable package (`config/__init__.py`), not just a data directory.

## Config manager

`config/config_manager.py` — **not** `src/config/`, which does not exist. It loads `base.yaml`, `universe.yaml` and `risk_config.yaml` into a typed `Config` dataclass (`config/models.py`). Import it as `from config.config_manager import ConfigManager`. Never read these YAMLs directly in application code.

## Key files

| File                          | Purpose                                                        |
| ----------------------------- | -------------------------------------------------------------- |
| `base.yaml`                   | Broker ports, risk limits, MDQC thresholds                     |
| `universe.yaml`               | All symbols, sectors, subsets                                  |
| `risk_config.yaml`            | Stop loss, earnings risk, correlations                         |
| `signals/*.yaml`              | Per-rule definitions for the RuleEngine                        |
| `strategy/{name}.yaml`        | Params + history per strategy (frozen subsystem)               |
| `strategy/regime_policy.yaml` | Per-strategy regime gating thresholds                          |
| `secrets.yaml`                | FMP API key, R2 credentials, SMTP — **gitignored**             |
| `backtest/`                   | Optuna search spaces and experiment specs — NOT param defaults |

Also present and self-explanatory: `demo.yaml`, `regime_weights.yaml`, `gate_policy_clusters.yaml`, `momentum_screener.yaml`, `pead_screener.yaml`, and the `validation/`, `verification/`, `grafana/`, `prometheus/` subdirectories.

## Adding a universe subset

Add it under the `subsets:` key in `config/universe.yaml`. Do not create a new YAML file for a subset — that breaks `config_manager.py`'s universe loading.

## Strategy params — single source of truth

One YAML per strategy in `config/strategy/`. All code reads params via `get_strategy_params("name")` from `src/domain/strategy/param_loader.py`; never hardcode values in runner dicts, `__init__` defaults, or `.get()` fallbacks. When changing params, update `params:` and push the old values to `history:`.

The YAML does **not** register a strategy — `@register_strategy` in `src/domain/strategy/playbook/` does. `pead.yaml` exists without a playbook class, so pead is unreachable from the strategy registry (see `src/backtest/CLAUDE.md`).
