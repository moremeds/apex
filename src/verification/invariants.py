"""
Constrained DSL for semantic invariants.

Provides typed invariant definitions with AST-based expression evaluation.
NO EVAL() - uses strict whitelist of allowed AST nodes for security.

Usage:
    from src.verification.invariants import BoundsInvariant, evaluate_expression

    # Bounds check
    bounds = BoundsInvariant(field="rsi", min_val=0, max_val=100)
    passed = check_bounds(bounds, {"rsi": 45.5})

    # Identity expression (MACD_hist = MACD_line - Signal_line)
    result = evaluate_expression(
        "macd_line - signal_line",
        {"macd_line": 1.5, "signal_line": 1.2}
    )  # Returns 0.3
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Type, Union

import numpy as np


class InvariantType(Enum):
    """Type of invariant check."""

    BOUNDS = "bounds"
    IDENTITY = "identity"
    NO_NAN = "no_nan"
    ALIGNMENT = "alignment"
    CAUSALITY = "causality"
    CUSTOM = "custom"


# ═══════════════════════════════════════════════════════════════
# AST Whitelist Expression Parser (NO EVAL)
# ═══════════════════════════════════════════════════════════════

# Allowed binary operators
ALLOWED_BIN_OPS: Set[Type[ast.operator]] = {
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
}

# Allowed unary operators
ALLOWED_UNARY_OPS: Set[Type[ast.unaryop]] = {
    ast.USub,  # Negation: -x
    ast.UAdd,  # Positive: +x (rarely used)
}

# Allowed comparison operators (for boolean expressions)
ALLOWED_CMP_OPS: Set[Type[ast.cmpop]] = {
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
}

# Allowed AST node types (strict whitelist)
# Note: Includes operator types (Add, Sub, etc.) which are visited by ast.walk()
ALLOWED_NODES: Set[Type[ast.AST]] = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Constant,  # Python 3.8+ uses Constant for numbers
    ast.Name,
    ast.Load,
    ast.BoolOp,
    ast.And,
    ast.Or,
    # Binary operators (visited as separate nodes)
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    # Unary operators
    ast.USub,
    ast.UAdd,
    # Comparison operators
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
}


class ExpressionError(ValueError):
    """Raised when expression parsing/evaluation fails."""


def parse_expression(expr: str, allowed_fields: Set[str]) -> ast.AST:
    """
    Parse expression with strict whitelist validation.

    Only allows:
    - Binary operators: + - * / // % **
    - Unary operators: - +
    - Comparison operators: == != < <= > >=
    - Boolean operators: and or
    - Field names (from allowed_fields)
    - Numeric constants

    NO function calls, NO attribute access, NO imports.

    Args:
        expr: Expression string (e.g., "macd_line - signal_line")
        allowed_fields: Set of allowed field names

    Returns:
        Validated AST tree

    Raises:
        ExpressionError: If expression contains disallowed constructs
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ExpressionError(f"Syntax error in expression: {e}")

    # Walk the AST and validate every node
    for node in ast.walk(tree):
        node_type = type(node)

        # Check node type is allowed
        if node_type not in ALLOWED_NODES:
            raise ExpressionError(
                f"Disallowed AST node type: {node_type.__name__}. "
                f"Expression must only contain arithmetic operations and field names."
            )

        # Check binary operator is allowed
        if isinstance(node, ast.BinOp):
            if type(node.op) not in ALLOWED_BIN_OPS:
                raise ExpressionError(f"Disallowed operator: {type(node.op).__name__}")

        # Check unary operator is allowed
        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in ALLOWED_UNARY_OPS:
                raise ExpressionError(f"Disallowed unary operator: {type(node.op).__name__}")

        # Check comparison operator is allowed
        if isinstance(node, ast.Compare):
            for op in node.ops:
                if type(op) not in ALLOWED_CMP_OPS:
                    raise ExpressionError(f"Disallowed comparison operator: {type(op).__name__}")

        # Check that Name nodes reference allowed fields only
        if isinstance(node, ast.Name):
            if node.id not in allowed_fields:
                raise ExpressionError(
                    f"Unknown field: '{node.id}'. " f"Allowed fields: {sorted(allowed_fields)}"
                )

        # Check that constants are numeric
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float, bool)):
                raise ExpressionError(
                    f"Only numeric constants allowed, got: {type(node.value).__name__}"
                )

    return tree


def _eval_node(node: ast.AST, context: Dict[str, float]) -> Union[float, bool]:
    """
    Recursively evaluate AST node with given context.

    This is a safe evaluator that only handles whitelisted node types.

    Args:
        node: AST node to evaluate
        context: Mapping of field names to values

    Returns:
        Evaluated numeric result
    """
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, context)

    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, (int, float, bool)):
            return value
        raise ExpressionError(f"Unexpected constant type: {type(value).__name__}")

    if isinstance(node, ast.Name):
        return context[node.id]

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, context)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        raise ExpressionError(f"Unexpected unary op: {type(node.op)}")

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, context)
        right = _eval_node(node.right, context)

        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                return float("nan")
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            if right == 0:
                return float("nan")
            return left // right
        if isinstance(node.op, ast.Mod):
            if right == 0:
                return float("nan")
            return left % right
        if isinstance(node.op, ast.Pow):
            return left**right

        raise ExpressionError(f"Unexpected binary op: {type(node.op)}")

    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, context)
        result = True
        current = left

        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_node(comparator, context)

            if isinstance(op, ast.Eq):
                result = result and (current == right)
            elif isinstance(op, ast.NotEq):
                result = result and (current != right)
            elif isinstance(op, ast.Lt):
                result = result and (current < right)
            elif isinstance(op, ast.LtE):
                result = result and (current <= right)
            elif isinstance(op, ast.Gt):
                result = result and (current > right)
            elif isinstance(op, ast.GtE):
                result = result and (current >= right)
            else:
                raise ExpressionError(f"Unexpected comparison op: {type(op)}")

            current = right

        return result

    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return all(_eval_node(v, context) for v in node.values)
        if isinstance(node.op, ast.Or):
            return any(_eval_node(v, context) for v in node.values)
        raise ExpressionError(f"Unexpected bool op: {type(node.op)}")

    raise ExpressionError(f"Cannot evaluate node type: {type(node).__name__}")


def evaluate_expression(expr: str, context: Dict[str, float]) -> Union[float, bool]:
    """
    Safely evaluate expression with given field values.

    Args:
        expr: Expression string (e.g., "macd_line - signal_line")
        context: Mapping of field names to values

    Returns:
        Evaluated result

    Raises:
        ExpressionError: If expression is invalid or references unknown fields
    """
    tree = parse_expression(expr, allowed_fields=set(context.keys()))
    return _eval_node(tree, context)


# ═══════════════════════════════════════════════════════════════
# Typed Invariant Definitions
# ═══════════════════════════════════════════════════════════════


@dataclass
class BoundsInvariant:
    """
    Bounds check invariant.

    e.g., RSI must be in [0, 100]
    """

    field: str
    min_val: float
    max_val: float
    after_warmup: bool = True
    description: str = ""

    def check(self, value: float) -> bool:
        """Check if value is within bounds."""
        if math.isnan(value) or math.isinf(value):
            return False
        return self.min_val <= value <= self.max_val


@dataclass
class IdentityInvariant:
    """
    Identity relationship invariant.

    e.g., MACD_histogram = MACD_line - Signal_line
    """

    result_field: str
    expression: str  # Parsed via AST whitelist, not eval()
    tolerance: float = 1e-9
    description: str = ""

    def check(self, context: Dict[str, float]) -> bool:
        """
        Check if result_field equals the expression evaluation.

        Args:
            context: Dict with all required fields including result_field
        """
        if self.result_field not in context:
            return False

        actual = context[self.result_field]
        if math.isnan(actual):
            return False

        try:
            # Remove result_field from context for expression evaluation
            expr_context = {k: v for k, v in context.items() if k != self.result_field}
            expected = evaluate_expression(self.expression, expr_context)
            return abs(actual - expected) <= self.tolerance
        except ExpressionError:
            return False


@dataclass
class NoNaNInvariant:
    """
    No NaN/Inf invariant after warmup.

    Ensures indicator outputs are valid numbers.
    """

    fields: List[str]  # Use ["*"] for all fields
    after_warmup: bool = True
    description: str = ""

    def check(self, row: Dict[str, Any], all_fields: Optional[List[str]] = None) -> bool:
        """
        Check no NaN/Inf values in specified fields.

        Args:
            row: Data row as dict
            all_fields: All available fields (used when fields=["*"])
        """
        fields_to_check = self.fields
        if self.fields == ["*"] and all_fields:
            fields_to_check = all_fields

        for field_name in fields_to_check:
            if field_name not in row:
                continue
            value = row[field_name]
            if isinstance(value, (int, float)):
                if math.isnan(value) or math.isinf(value):
                    return False
        return True


@dataclass
class AlignmentInvariant:
    """
    Output index alignment invariant.

    Ensures indicator output index matches input bar index.
    """

    timeframes: List[str]
    description: str = ""

    def check(self, input_index: Any, output_index: Any) -> bool:
        """Check that output index matches input index."""
        # Compare as lists for pandas compatibility
        if hasattr(input_index, "tolist"):
            input_list = input_index.tolist()
        else:
            input_list = list(input_index)

        if hasattr(output_index, "tolist"):
            output_list = output_index.tolist()
        else:
            output_list = list(output_index)

        return bool(input_list == output_list)


@dataclass
class CausalityInvariant:
    """
    No-lookahead causality invariant (CRITICAL for trading systems).

    Verifies that perturbing future data does not change past outputs.

    Method:
    1. Calculate output Y from OHLCV
    2. Perturb bar at t+offset (e.g., close *= 1.5)
    3. Recalculate output Y'
    4. Assert: Y[<=t] == Y'[<=t] (byte-equal or within tolerance)
    """

    indicators: List[str] = field(default_factory=list)  # Indicators to test
    perturb_offset: int = 10  # Which future bar to perturb
    perturb_field: str = "close"
    perturb_factor: float = 1.5  # Multiply field by this
    tolerance: float = 0.0  # 0.0 = byte-equal required
    description: str = ""

    def check_outputs_unchanged(
        self,
        original_output: np.ndarray,
        perturbed_output: np.ndarray,
        perturb_index: int,
    ) -> bool:
        """
        Check that outputs before perturb_index are unchanged.

        Args:
            original_output: Output from original data
            perturbed_output: Output from perturbed data
            perturb_index: Index of the perturbed bar

        Returns:
            True if past outputs are unchanged
        """
        # Only compare rows up to (not including) the perturbed index
        past_original = original_output[:perturb_index]
        past_perturbed = perturbed_output[:perturb_index]

        if len(past_original) == 0:
            return True

        if self.tolerance == 0.0:
            # Byte-equal comparison
            return bool(np.array_equal(past_original, past_perturbed))
        else:
            # Tolerance-based comparison
            diff = np.abs(past_original - past_perturbed)
            return bool(np.all(diff <= self.tolerance))


@dataclass
class CustomInvariant:
    """
    Custom invariant with callable check function.

    Use for complex invariants that don't fit other types.
    """

    name: str
    check_fn: Any  # Callable[[Dict], bool]
    description: str = ""

    def check(self, context: Dict[str, Any]) -> bool:
        """Run the custom check function."""
        return bool(self.check_fn(context))


# ═══════════════════════════════════════════════════════════════
# Invariant Registry
# ═══════════════════════════════════════════════════════════════

INVARIANT_TYPES: Dict[str, Type] = {
    "bounds": BoundsInvariant,
    "identity": IdentityInvariant,
    "no_nan": NoNaNInvariant,
    "alignment": AlignmentInvariant,
    "causality": CausalityInvariant,
    "custom": CustomInvariant,
}


def create_invariant(config: Dict[str, Any]) -> Any:
    """
    Create an invariant from configuration dict.

    Args:
        config: Dict with 'type' key and type-specific parameters

    Returns:
        Invariant instance
    """
    inv_type = config.get("type")
    if inv_type not in INVARIANT_TYPES:
        raise ValueError(f"Unknown invariant type: {inv_type}")

    cls = INVARIANT_TYPES[inv_type]

    # Extract type-specific params
    params = {k: v for k, v in config.items() if k not in ("type", "id", "indicator")}

    return cls(**params)


# ═══════════════════════════════════════════════════════════════
# Common Invariant Checks
# ═══════════════════════════════════════════════════════════════


def check_bounds(value: float, min_val: float, max_val: float) -> bool:
    """Simple bounds check utility."""
    if math.isnan(value) or math.isinf(value):
        return False
    return min_val <= value <= max_val


def check_no_nan(values: List[float]) -> bool:
    """Check no NaN/Inf in list."""
    return all(not (math.isnan(v) or math.isinf(v)) for v in values if isinstance(v, (int, float)))


def check_identity(
    actual: float,
    expr: str,
    context: Dict[str, float],
    tolerance: float = 1e-9,
) -> bool:
    """
    Check identity relationship.

    Args:
        actual: Actual value to check
        expr: Expression that should equal actual
        context: Field values for expression evaluation
        tolerance: Acceptable difference
    """
    try:
        expected = evaluate_expression(expr, context)
        return abs(actual - expected) <= tolerance
    except ExpressionError:
        return False
