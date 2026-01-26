"""Unit tests for HealthBar widget."""

import pytest
from textual.app import App, ComposeResult

from src.infrastructure.monitoring.health_monitor import ComponentHealth, HealthStatus
from src.tui.widgets.health_bar import HealthBar


class HealthBarTestApp(App):
    """Test app for HealthBar widget."""

    CSS = """
    HealthBar {
        height: 5;
    }
    HealthBar .health-component {
        height: 3;
        width: auto;
    }
    """

    def compose(self) -> ComposeResult:
        yield HealthBar(id="test-health")


class TestHealthBar:
    """Tests for HealthBar widget."""

    @pytest.mark.asyncio
    async def test_health_bar_renders_empty_on_mount(self) -> None:
        """HealthBar should render with hidden widgets when no health data."""
        async with HealthBarTestApp().run_test() as pilot:
            app = pilot.app
            health_bar = app.query_one("#test-health", HealthBar)

            # All 10 widgets should exist
            for i in range(10):
                widget = health_bar.query_one(f"#health-{i}")
                assert widget is not None
                # Initially hidden (display=False after on_mount renders empty list)
                assert widget.display is False

    @pytest.mark.asyncio
    async def test_health_bar_renders_health_data(self) -> None:
        """HealthBar should display health components when data is set."""
        async with HealthBarTestApp().run_test() as pilot:
            app = pilot.app
            health_bar = app.query_one("#test-health", HealthBar)

            # Set health data
            health_data = [
                ComponentHealth(
                    component_name="test_component",
                    status=HealthStatus.HEALTHY,
                    message="All good",
                ),
                ComponentHealth(
                    component_name="degraded_component",
                    status=HealthStatus.DEGRADED,
                    message="Needs attention",
                ),
            ]
            health_bar.health = health_data

            # Wait for reactive update to process
            await pilot.pause()

            # First two widgets should be visible
            widget0 = health_bar.query_one("#health-0")
            widget1 = health_bar.query_one("#health-1")
            widget2 = health_bar.query_one("#health-2")

            assert widget0.display is True
            assert widget1.display is True
            assert widget2.display is False  # Unused

    @pytest.mark.asyncio
    async def test_health_bar_formats_status_correctly(self) -> None:
        """HealthBar should format status icons correctly."""
        async with HealthBarTestApp().run_test() as pilot:
            app = pilot.app
            health_bar = app.query_one("#test-health", HealthBar)

            # Test all status types
            health_data = [
                ComponentHealth("healthy", HealthStatus.HEALTHY, "OK"),
                ComponentHealth("degraded", HealthStatus.DEGRADED, "Warn"),
                ComponentHealth("unhealthy", HealthStatus.UNHEALTHY, "Down"),
                ComponentHealth("unknown", HealthStatus.UNKNOWN, "?"),
            ]
            health_bar.health = health_data
            await pilot.pause()

            # Verify all 4 widgets are visible
            widget0 = health_bar.query_one("#health-0")
            widget1 = health_bar.query_one("#health-1")
            widget2 = health_bar.query_one("#health-2")
            widget3 = health_bar.query_one("#health-3")

            assert widget0.display is True
            assert widget1.display is True
            assert widget2.display is True
            assert widget3.display is True

            # Test the format function directly
            formatted = health_bar._format_component(health_data[0])
            assert "[OK]" in formatted
            formatted = health_bar._format_component(health_data[1])
            assert "[W]" in formatted
            formatted = health_bar._format_component(health_data[2])
            assert "[X]" in formatted
            formatted = health_bar._format_component(health_data[3])
            assert "[?]" in formatted

    @pytest.mark.asyncio
    async def test_health_bar_market_data_coverage_metadata(self) -> None:
        """HealthBar should handle market_data_coverage metadata correctly."""
        async with HealthBarTestApp().run_test() as pilot:
            app = pilot.app
            health_bar = app.query_one("#test-health", HealthBar)

            # Test market_data_coverage with missing count
            health_data = [
                ComponentHealth(
                    component_name="market_data_coverage",
                    status=HealthStatus.DEGRADED,
                    message="Coverage issue",
                    metadata={"missing_count": 3, "total": 10},
                ),
            ]
            health_bar.health = health_data
            await pilot.pause()

            widget0 = health_bar.query_one("#health-0")
            assert widget0.display is True

            # Test the format function directly to verify metadata handling
            formatted = health_bar._format_component(health_data[0])
            assert "3/10 missing MD" in formatted

    @pytest.mark.asyncio
    async def test_health_bar_clears_unused_widgets(self) -> None:
        """HealthBar should hide widgets when health list shrinks."""
        async with HealthBarTestApp().run_test() as pilot:
            app = pilot.app
            health_bar = app.query_one("#test-health", HealthBar)

            # Set 3 health components
            health_bar.health = [
                ComponentHealth("c1", HealthStatus.HEALTHY, ""),
                ComponentHealth("c2", HealthStatus.HEALTHY, ""),
                ComponentHealth("c3", HealthStatus.HEALTHY, ""),
            ]
            await pilot.pause()

            # Verify 3 visible
            assert health_bar.query_one("#health-0").display is True
            assert health_bar.query_one("#health-1").display is True
            assert health_bar.query_one("#health-2").display is True
            assert health_bar.query_one("#health-3").display is False

            # Reduce to 1 component
            health_bar.health = [
                ComponentHealth("c1", HealthStatus.HEALTHY, ""),
            ]
            await pilot.pause()

            # Verify only 1 visible now
            assert health_bar.query_one("#health-0").display is True
            assert health_bar.query_one("#health-1").display is False
            assert health_bar.query_one("#health-2").display is False
