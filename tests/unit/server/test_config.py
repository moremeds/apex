"""Tests for server config loader."""

from src.server.config import ProviderConfig, ServerConfig, load_server_config


def test_load_default_config():
    cfg = load_server_config("config/server.yaml")
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 8080
    assert cfg.duckdb_path == "data/server.duckdb"
    assert cfg.r2_flush_interval_sec == 300
    assert cfg.max_symbols == 200


def test_config_timeframes():
    cfg = load_server_config("config/server.yaml")
    assert "1d" in cfg.timeframes
    assert "1h" in cfg.timeframes


def test_config_providers():
    cfg = load_server_config("config/server.yaml")
    assert "longbridge" in cfg.providers
    assert cfg.providers["longbridge"].enabled is True


def test_config_provider_sub_types():
    cfg = load_server_config("config/server.yaml")
    lb = cfg.providers["longbridge"]
    assert "quote" in lb.sub_types
    assert "depth" in lb.sub_types


def test_config_universe_path():
    cfg = load_server_config("config/server.yaml")
    assert cfg.universe_path == "config/universe.yaml"


def test_config_indicators():
    cfg = load_server_config("config/server.yaml")
    assert "rsi" in cfg.indicators
    assert "dual_macd" in cfg.indicators


def test_config_from_screener_flag():
    cfg = load_server_config("config/server.yaml")
    assert isinstance(cfg.from_screener, bool)
