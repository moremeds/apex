# Vulture Whitelist for APEX
# This file contains intentionally unused code that vulture should ignore.
# These are typically framework callbacks, abstract methods, and test fixtures.

# ═══════════════════════════════════════════════════════════════════════════════
# Strategy Framework Callbacks
# These methods are called by the backtesting/trading engine, not directly invoked
# ═══════════════════════════════════════════════════════════════════════════════

_.on_bar  # Called when a new bar closes (Strategy.on_bar)
_.on_tick  # Called on each market tick (Strategy.on_tick)
_.on_order  # Called when order state changes (Strategy.on_order)
_.on_fill  # Called when order is filled (Strategy.on_fill)
_.on_start  # Called when strategy starts (Strategy.on_start)
_.on_stop  # Called when strategy stops (Strategy.on_stop)
_.on_data  # Called when new data arrives (Strategy.on_data)
_.on_signal  # Called when trading signal is generated
_.on_position_change  # Called when position changes

# ═══════════════════════════════════════════════════════════════════════════════
# Event Bus Handlers
# These are registered as callbacks and invoked by the event bus
# ═══════════════════════════════════════════════════════════════════════════════

_.handle  # Generic event handler method
_.on_event  # Event handler callback
_.on_market_data  # Market data event handler
_.on_position_update  # Position update handler
_.on_risk_signal  # Risk signal handler

# ═══════════════════════════════════════════════════════════════════════════════
# Abstract/Interface Methods
# Defined in base classes, implemented in subclasses
# ═══════════════════════════════════════════════════════════════════════════════

_.calculate  # Indicator calculation method
_.evaluate  # Rule evaluation method
_.execute  # Command execution method
_.validate  # Validation method
_.transform  # Data transformation method

# ═══════════════════════════════════════════════════════════════════════════════
# Testing Framework
# unittest/pytest fixtures and hooks
# ═══════════════════════════════════════════════════════════════════════════════

_.setUp  # unittest setup
_.tearDown  # unittest teardown
_.setUpClass  # unittest class setup
_.tearDownClass  # unittest class teardown
_.setUpModule  # unittest module setup
_.tearDownModule  # unittest module teardown

# Pytest fixtures (decorated functions that appear unused)
_.pytest_configure  # pytest hook
_.pytest_collection_modifyitems  # pytest hook
_.conftest  # pytest configuration

# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic/Dataclass Magic Methods
# Auto-generated methods that vulture doesn't understand
# ═══════════════════════════════════════════════════════════════════════════════

_.model_validate  # Pydantic v2 validation
_.model_dump  # Pydantic v2 serialization
_.model_json_schema  # Pydantic v2 schema generation
_.__post_init__  # dataclass post-init hook

# ═══════════════════════════════════════════════════════════════════════════════
# Textual TUI Framework
# Widget methods called by the framework
# Note: Textual uses watch_<attr>, action_<name>, key_<key> naming conventions
# ═══════════════════════════════════════════════════════════════════════════════

_.compose  # Widget composition
_.on_mount  # Mount lifecycle hook
_.on_unmount  # Unmount lifecycle hook
_.on_focus  # Focus event handler
_.on_blur  # Blur event handler
_.on_key  # Key event handler
_.on_click  # Click event handler
_.on_resize  # Resize event handler
_.render  # Widget render method
_.refresh  # Widget refresh method

# ═══════════════════════════════════════════════════════════════════════════════
# SQLAlchemy / Database
# ORM methods and hooks
# ═══════════════════════════════════════════════════════════════════════════════

_.__tablename__  # Table name declaration
_.__table_args__  # Table arguments

# ═══════════════════════════════════════════════════════════════════════════════
# Decorator Patterns
# Functions used as decorators appear unused
# ═══════════════════════════════════════════════════════════════════════════════

register_strategy  # Strategy registration decorator
register_indicator  # Indicator registration decorator
register_rule  # Rule registration decorator

# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entry Points
# Main functions called by entry points, not imported directly
# ═══════════════════════════════════════════════════════════════════════════════

main  # CLI entry point
cli  # Click CLI group

# ═══════════════════════════════════════════════════════════════════════════════
# Required-but-Unused Arguments
# These are required by protocols/signatures but intentionally not used
# ═══════════════════════════════════════════════════════════════════════════════

frame  # Signal handler frame argument (required by signal.signal)
exc_val  # Context manager __exit__ argument
exc_tb  # Context manager __exit__ traceback argument
exc_type  # Context manager __exit__ exception type
