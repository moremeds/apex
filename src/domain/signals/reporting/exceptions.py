"""
Reporting Exceptions.

Custom exceptions for signal report generation, including budget enforcement.
"""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class BudgetContributor:
    """Identifies a contributor to budget overflow."""

    key: str
    size_bytes: int
    pct_of_section: float


class SizeBudgetExceeded(Exception):
    """
    Exception raised when a report section exceeds its size budget.

    Used by PackageBuilder to enforce size limits for summary.json
    and other report components to ensure fast load times.
    """

    def __init__(
        self,
        section: str,
        actual_kb: float,
        budget_kb: float,
        top_contributors: List[BudgetContributor] | None = None,
    ) -> None:
        """
        Initialize SizeBudgetExceeded exception.

        Args:
            section: Name of the section that exceeded budget
            actual_kb: Actual size in KB
            budget_kb: Budget limit in KB
            top_contributors: Top contributors to the overflow
        """
        self.section = section
        self.actual_kb = actual_kb
        self.budget_kb = budget_kb
        self.top_contributors = top_contributors or []

        # Build message
        overflow_pct = ((actual_kb - budget_kb) / budget_kb) * 100 if budget_kb > 0 else float("inf")
        msg = (
            f"Section '{section}' exceeds budget: "
            f"{actual_kb:.1f}KB > {budget_kb}KB (overflow: +{overflow_pct:.0f}%)"
        )

        if top_contributors:
            msg += "\n  Top contributors:"
            for c in top_contributors[:5]:
                msg += f"\n    - {c.key}: {c.size_bytes / 1024:.1f}KB ({c.pct_of_section:.0f}%)"

        super().__init__(msg)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "section": self.section,
            "actual_kb": self.actual_kb,
            "budget_kb": self.budget_kb,
            "overflow_kb": self.actual_kb - self.budget_kb,
            "top_contributors": [
                {
                    "key": c.key,
                    "size_bytes": c.size_bytes,
                    "pct_of_section": c.pct_of_section,
                }
                for c in self.top_contributors
            ],
        }


class ReportGenerationError(Exception):
    """General error during report generation."""

    pass


class RegimeDataMissing(Exception):
    """Raised when required regime data is not available."""

    pass
