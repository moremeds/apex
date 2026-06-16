# config/ — Configuration files

Root `CLAUDE.md` is authoritative for policy.

## Single source of truth rule

**One YAML per strategy** — `config/strategy/{name}.yaml`. All code reads params via `get_strategy_params("name")` from `src/domain/strategy/param_loader.py`. Never hardcode param values in:
- Runner dicts
- `__init__` defaults
- `.get()` fallbacks

When changing params: update `params:` in YAML and push old values to `history:`. The loader serves both signal generators and the runner's strategy registry.

## Key config files

| File | Purpose |
|------|---------|
| `base.yaml` | Broker ports, risk limits, MDQC thresholds |
| `universe.yaml` | All symbols, sectors, subsets — add subsets here, never create new files |
| `risk_config.yaml` | Stop loss, earnings risk, correlations |
| `signals/*.yaml` | Per-rule YAML definitions for the RuleEngine |
| `strategy/{name}.yaml` | Canonical params + history for each strategy |
| `strategy/regime_policy.yaml` | Per-strategy regime gating thresholds |
| `secrets.yaml` | FMP API key, SMTP — **gitignored** |
| `backtest/` | Optuna search spaces and spec YAMLs — NOT param defaults |

## Adding a new config subset (universe)

Add the subset directly to `config/universe.yaml` under the `subsets:` key. Do not create a new YAML file for a subset — that violates the single-source rule and breaks `config_manager.py`'s universe loading.

## Config manager

`src/config/config_manager.py` loads `base.yaml`, `universe.yaml`, and `risk_config.yaml` into a typed `Config` dataclass. Access via `get_config()` — never read YAML files directly in application code.
