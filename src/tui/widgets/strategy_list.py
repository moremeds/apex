"""
Strategy list widget for the Lab view.

Uses StrategyViewModel for business logic and incremental updates.
Displays available strategies from the StrategyRegistry with backtest results.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from textual.widgets import DataTable
from textual.reactive import reactive
from textual.message import Message
from textual import work

from ..viewmodels.strategy_vm import StrategyDisplayState, StrategyViewModel


class StrategyList(DataTable):
    """
    Strategy list display with selection support.

    Shows registered strategies with their last backtest results.
    Uses StrategyViewModel for incremental updates.
    """

    class StrategySelected(Message):
        """Message sent when a strategy is selected."""

        def __init__(self, strategy_name: str, strategy_info: dict) -> None:
            self.strategy_name = strategy_name
            self.strategy_info = strategy_info
            super().__init__()

    class StrategyActivated(Message):
        """Message sent when a strategy is activated (run requested)."""

        def __init__(self, strategy_name: str, strategy_info: dict) -> None:
            self.strategy_name = strategy_name
            self.strategy_info = strategy_info
            super().__init__()

    # Column definitions matching original Rich layout
    COLUMNS = [
        ("", 2),  # Selection indicator
        ("Strategy", 18),
        ("Description", 25),
        ("Return", 8),
        ("Sharpe", 7),
        ("Max DD", 8),
        ("Trades", 7),
        ("Win%", 6),
        ("Status", 10),
    ]

    # Single combined state reactive - avoids triple rebuild on related changes
    state: reactive[StrategyDisplayState] = reactive(
        StrategyDisplayState, init=False
    )

    def __init__(self, **kwargs) -> None:
        super().__init__(cursor_type="row", zebra_stripes=False, **kwargs)
        self._view_model = StrategyViewModel()
        self._strategy_map: Dict[str, dict] = {}
        self._strategy_list: List[str] = []
        self._column_keys: List[str] = []
        self._row_keys: List[str] = []
        self._selected_strategy: Optional[str] = None

    def on_mount(self) -> None:
        """Set up columns with keys for incremental updates."""
        self._column_keys.clear()
        for idx, (name, width) in enumerate(self.COLUMNS):
            col_key = f"col-{idx}"
            self.add_column(name, width=width, key=col_key)
            self._column_keys.append(col_key)

        # Show loading state
        self._show_loading_state()

        # Load strategies in background worker to avoid blocking main thread
        self._load_strategies_worker()

    def _show_loading_state(self) -> None:
        """Show loading state while strategies are being loaded."""
        self.clear()
        self._row_keys.clear()
        self.add_row(
            "",
            "[dim]Loading...[/]",
            "[dim]Fetching strategies[/]",
            "-",
            "-",
            "-",
            "-",
            "-",
            "",
            key="__loading__",
        )
        self._row_keys.append("__loading__")

    @work(thread=True)
    def _load_strategies_worker(self) -> None:
        """Load strategies in background thread to avoid blocking UI."""
        try:
            # Import example strategies to ensure they're registered
            # This can be slow due to module loading
            from ...domain.strategy import examples  # noqa: F401
            from ...domain.strategy.registry import StrategyRegistry, get_strategy_info

            strategy_list = sorted(StrategyRegistry.list_strategies())
            strategy_map = {}

            for name in strategy_list:
                info = get_strategy_info(name)
                if info:
                    strategy_map[name] = {
                        "name": name,
                        "description": info.get("description", ""),
                        "version": info.get("version", "1.0"),
                        "author": info.get("author", ""),
                    }
                else:
                    strategy_map[name] = {
                        "name": name,
                        "description": "",
                        "version": "1.0",
                        "author": "",
                    }

            # Marshal back to main thread for UI updates
            self.app.call_from_thread(self._on_strategies_loaded, strategy_list, strategy_map)
        except Exception as e:
            self.app.call_from_thread(self._on_strategies_error, str(e))

    def _on_strategies_loaded(
        self, strategy_list: List[str], strategy_map: Dict[str, dict]
    ) -> None:
        """Handle successful strategy load (called from main thread)."""
        self._strategy_list = strategy_list
        self._strategy_map = strategy_map

        # Update ViewModel with strategy info
        self._view_model.set_strategies(self._strategy_map)
        self._full_rebuild()

    def _on_strategies_error(self, error: str) -> None:
        """Handle strategy load error (called from main thread)."""
        self.log.error(f"Failed to load strategies: {error}")
        self._show_empty_state()

    def _show_empty_state(self) -> None:
        """Show empty state when no strategies available."""
        self.clear()
        self._row_keys.clear()
        self.add_row(
            "",
            "[dim]No strategies[/]",
            "[dim]Import strategy modules[/]",
            "-",
            "-",
            "-",
            "-",
            "-",
            "",
            key="__empty__",
        )
        self._row_keys.append("__empty__")

    def watch_state(self, state: StrategyDisplayState) -> None:
        """Single watcher for all state changes."""
        self._view_model.set_state(state)

        if self._view_model.full_refresh_needed(self._strategy_list):
            self._full_rebuild()
        else:
            self._incremental_update()

    def _full_rebuild(self) -> None:
        """Full table rebuild - used for structural changes."""
        # Preserve cursor position
        selected_name = self._selected_strategy
        if not selected_name and self.cursor_row is not None and self._strategy_list:
            if 0 <= self.cursor_row < len(self._strategy_list):
                selected_name = self._strategy_list[self.cursor_row]

        self._view_model.set_selected(selected_name)

        self.clear()
        self._row_keys.clear()

        display_data = self._view_model.compute_display_data(self._strategy_list)
        row_order = self._view_model.get_row_order(self._strategy_list)

        for row_key in row_order:
            if row_key in display_data:
                self.add_row(*display_data[row_key], key=row_key)
                self._row_keys.append(row_key)

        # Restore cursor
        if self.row_count > 0 and selected_name:
            try:
                selected_idx = self._strategy_list.index(selected_name)
                selected_idx = min(selected_idx, self.row_count - 1)
                self.move_cursor(row=selected_idx, column=0, scroll=False)
            except (ValueError, IndexError):
                pass

    def _incremental_update(self) -> None:
        """Incremental cell-level updates - used for value changes."""
        # Update selected state in ViewModel
        selected_name = self._selected_strategy
        if not selected_name and self.cursor_row is not None and self._strategy_list:
            if 0 <= self.cursor_row < len(self._strategy_list):
                selected_name = self._strategy_list[self.cursor_row]

        self._view_model.set_selected(selected_name)

        row_ops, cell_updates, new_order = self._view_model.compute_updates(
            self._strategy_list
        )

        # Process row removals first
        for row_op in row_ops:
            if row_op.action == "remove":
                try:
                    self.remove_row(row_op.row_key)
                    if row_op.row_key in self._row_keys:
                        self._row_keys.remove(row_op.row_key)
                except Exception as e:
                    self.log.error(f"Failed to remove row {row_op.row_key}: {e}")

        # Process row additions
        for row_op in row_ops:
            if row_op.action == "add" and row_op.values:
                try:
                    self.add_row(*row_op.values, key=row_op.row_key)
                    self._row_keys.append(row_op.row_key)
                except Exception as e:
                    self.log.error(f"Failed to add row {row_op.row_key}: {e}")

        # Process cell updates
        for cell in cell_updates:
            try:
                if cell.column_index < len(self._column_keys):
                    col_key = self._column_keys[cell.column_index]
                    self.update_cell(cell.row_key, col_key, cell.value)
            except Exception as e:
                self.log.error(f"Failed to update cell: {e}")

        # NOTE: new_order from compute_updates() is intentionally not applied.
        # Textual DataTable doesn't support row reordering natively.
        # Strategy list order is stable (alphabetical), so this is acceptable.

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection (Enter/click)."""
        if event.row_key is not None:
            key = str(event.row_key.value)
            if key != "__empty__" and key in self._strategy_map:
                self._selected_strategy = key
                self.post_message(
                    self.StrategyActivated(key, self._strategy_map[key])
                )

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Handle row highlight (cursor movement)."""
        if event.row_key is not None:
            key = str(event.row_key.value)
            if key != "__empty__" and key in self._strategy_map:
                self._selected_strategy = key
                self.post_message(
                    self.StrategySelected(key, self._strategy_map[key])
                )

    def get_selected_strategy(self) -> Optional[str]:
        """Get the currently selected strategy name."""
        if self.cursor_row is not None and self._strategy_list:
            try:
                if self.cursor_row < len(self._strategy_list):
                    return self._strategy_list[self.cursor_row]
            except Exception as e:
                self.log.error(f"Failed to get selected strategy: {e}")
        return None

    def refresh_strategies(self) -> None:
        """Reload strategies from registry."""
        self._show_loading_state()
        self._load_strategies_worker()

    def set_backtest_result(self, strategy_name: str, result: Any) -> None:
        """Set backtest result for a strategy."""
        current_state = self.state
        new_results = dict(current_state.results)
        new_results[strategy_name] = result
        self.state = StrategyDisplayState(
            results=new_results,
            failures=current_state.failures,
            running=current_state.running,
        )

    def set_backtest_failure(self, strategy_name: str, error: str) -> None:
        """Set backtest failure for a strategy."""
        current_state = self.state
        new_failures = dict(current_state.failures)
        new_failures[strategy_name] = error
        self.state = StrategyDisplayState(
            results=current_state.results,
            failures=new_failures,
            running=current_state.running,
        )

    def set_running_strategy(self, strategy_name: Optional[str]) -> None:
        """Set currently running strategy."""
        current_state = self.state
        self.state = StrategyDisplayState(
            results=current_state.results,
            failures=current_state.failures,
            running=strategy_name,
        )
