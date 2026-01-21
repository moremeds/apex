"""
Result type for explicit error handling.

Provides a Result[T] type that represents either a successful value (Ok)
or an error (Err). This makes error handling explicit and type-safe.

Usage:
    from src.utils.result import Result, Ok, Err

    def divide(a: float, b: float) -> Result[float, str]:
        if b == 0:
            return Err("Division by zero")
        return Ok(a / b)

    result = divide(10, 2)
    if result.is_ok():
        print(f"Result: {result.unwrap()}")
    else:
        print(f"Error: {result.error}")

    # Or use match pattern (Python 3.10+)
    match result:
        case Ok(value):
            print(f"Result: {value}")
        case Err(error):
            print(f"Error: {error}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar, Union

T = TypeVar("T")  # Success type
E = TypeVar("E")  # Error type
U = TypeVar("U")  # Mapped type


@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    """Represents a successful result."""

    value: T

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False

    def unwrap(self) -> T:
        """Get the value. Safe to call after checking is_ok()."""
        return self.value

    def unwrap_or(self, default: T) -> T:
        """Get the value or a default."""
        return self.value

    def unwrap_or_else(self, f: Callable[[], T]) -> T:
        """Get the value or compute a default."""
        return self.value

    @property
    def error(self) -> None:
        """Error is None for Ok results."""
        return None

    def map(self, f: Callable[[T], U]) -> "Result[U, E]":
        """Apply a function to the value if Ok."""
        return Ok(f(self.value))

    def map_err(self, f: Callable[[E], U]) -> "Result[T, U]":
        """Apply a function to the error if Err. No-op for Ok."""
        return self  # type: ignore

    def and_then(self, f: Callable[[T], "Result[U, E]"]) -> "Result[U, E]":
        """Chain another Result-returning function."""
        return f(self.value)


@dataclass(frozen=True, slots=True)
class Err(Generic[E]):
    """Represents a failed result."""

    error: E

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    def unwrap(self) -> None:
        """Raises an error. Use is_ok() to check first."""
        raise ValueError(f"Called unwrap() on Err: {self.error}")

    def unwrap_or(self, default: T) -> T:
        """Get a default value for Err."""
        return default

    def unwrap_or_else(self, f: Callable[[], T]) -> T:
        """Compute a default value for Err."""
        return f()

    @property
    def value(self) -> None:
        """Value is None for Err results."""
        return None

    def map(self, f: Callable[[T], U]) -> "Result[U, E]":
        """No-op for Err."""
        return self  # type: ignore

    def map_err(self, f: Callable[[E], U]) -> "Result[T, U]":
        """Apply a function to the error."""
        return Err(f(self.error))

    def and_then(self, f: Callable[[T], "Result[U, E]"]) -> "Result[U, E]":
        """No-op for Err."""
        return self  # type: ignore


# Result is a union of Ok and Err
Result = Union[Ok[T], Err[E]]


def try_result(f: Callable[[], T], error_type: type[BaseException] = Exception) -> Result[T, str]:
    """
    Execute a function and capture exceptions as Err.

    Args:
        f: Function to execute.
        error_type: Exception type to catch (default: Exception).

    Returns:
        Ok with result or Err with error message.

    Example:
        result = try_result(lambda: int("not a number"))
        # Returns Err("invalid literal for int() with base 10: 'not a number'")
    """
    try:
        return Ok(f())
    except error_type as e:
        return Err(str(e))


def collect_results(results: list[Result[T, E]]) -> Result[list[T], E]:
    """
    Collect a list of Results into a Result of list.

    Returns the first error if any result is Err.

    Args:
        results: List of Result objects.

    Returns:
        Ok with list of values or first Err encountered.
    """
    values: list[T] = []
    for result in results:
        if result.is_err():
            return result  # type: ignore
        values.append(result.unwrap())  # type: ignore[arg-type]
    return Ok(values)
