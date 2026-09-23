"""Calendar policy for gap diagnosis: exchange closures, weekday approximation, runs."""

from __future__ import annotations

from datetime import date

from src.application.lake.gaps import _runs, expected_sessions


def test_xnys_skips_the_2018_12_05_and_2025_01_09_closures() -> None:
    sessions, label = expected_sessions("xnys", date(2018, 12, 3), date(2018, 12, 7))
    assert date(2018, 12, 5) not in sessions and len(sessions) == 4
    sessions, _ = expected_sessions("xnys", date(2025, 1, 6), date(2025, 1, 10))
    assert date(2025, 1, 9) not in sessions and len(sessions) == 4
    assert label["name"] == "XNYS" and label["version"].startswith("pandas-market-calendars")


def test_weekday_policy_keeps_us_holidays_and_labels_itself() -> None:
    sessions, label = expected_sessions("weekdays", date(2025, 1, 6), date(2025, 1, 12))
    assert date(2025, 1, 9) in sessions and len(sessions) == 5
    assert label == {"name": "weekdays", "certainty": "approximate"}


def test_runs_follow_expected_sessions_not_calendar_days() -> None:
    expected, _ = expected_sessions("xnys", date(2025, 1, 6), date(2025, 1, 17))
    order = {day: i for i, day in enumerate(expected)}
    # 01-08 and 01-10 straddle the 01-09 closure: consecutive sessions, one run.
    runs = _runs([date(2025, 1, 8), date(2025, 1, 10), date(2025, 1, 15)], order)
    assert [(r.start, r.end, r.sessions) for r in runs] == [
        (date(2025, 1, 8), date(2025, 1, 10), 2),
        (date(2025, 1, 15), date(2025, 1, 15), 1),
    ]
