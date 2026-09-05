"""A safe arithmetic-expression evaluator."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Safely evaluate an arithmetic expression, including common math functions.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Arithmetic expression to evaluate"}
            },
            "required": ["expression"],
        },
    },
}

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
}
_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(float(right)) > 100:
            raise ValueError("exponent is too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function = _FUNCTIONS.get(node.func.id)
        if function is None or node.keywords:
            raise ValueError("function is not allowed")
        return function(*[_evaluate(arg) for arg in node.args])
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


def run(args: dict[str, Any]) -> str:
    expression = str(args.get("expression", "")).strip()
    if not expression or len(expression) > 300:
        raise ValueError("expression must contain 1-300 characters")
    try:
        value = _evaluate(ast.parse(expression, mode="eval"))
    except (SyntaxError, ArithmeticError, TypeError, ValueError) as exc:
        raise ValueError(f"calculation failed: {exc}") from exc
    if isinstance(value, float):
        rendered = f"{value:.12g}"
        details = f"\nrounded_6={value:.6f}"
    else:
        rendered = str(value)
        digit_count = len(str(abs(value))) if isinstance(value, int) else 0
        details = f"\ndigits={digit_count} ({digit_count} 位)" if digit_count else ""
    return f"expression={expression}\nresult={rendered}{details}"
