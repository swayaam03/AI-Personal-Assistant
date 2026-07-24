import ast
import operator as op
from langchain_core.tools import tool

# Supported safe operators for calculation
_SAFE_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _safe_eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        op_type = type(node.op)
        if op_type in _SAFE_OPERATORS:
            return _SAFE_OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
    elif isinstance(node, ast.UnaryOp):
        operand = _safe_eval_node(node.operand)
        op_type = type(node.op)
        if op_type in _SAFE_OPERATORS:
            return _SAFE_OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
    else:
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")


@tool
def calculate(expression: str) -> str:
    """
    Evaluates a mathematical expression and returns the numerical result.
    
    Args:
        expression: A math string expression, e.g. "25 * 4 + 10" or "100 / 4".
        
    Returns:
        The calculated numerical result as a string.
    """
    try:
        # Parse expression into an AST tree safely without eval() security vulnerabilities
        parsed = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval_node(parsed.body)
        return f"Calculation Result: {expression} = {result}"
    except Exception as e:
        return f"Error evaluating expression '{expression}': {str(e)}"
