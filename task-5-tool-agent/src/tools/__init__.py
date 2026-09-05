"""Built-in tools exposed to the ReAct agent."""

from . import calculator, file_search, python_sandbox, wiki

TOOLS = {
    "calculator": calculator.run,
    "python_sandbox": python_sandbox.run,
    "file_search": file_search.run,
    "wiki": wiki.run,
}

TOOL_SCHEMAS = [
    calculator.TOOL_SCHEMA,
    python_sandbox.TOOL_SCHEMA,
    file_search.TOOL_SCHEMA,
    wiki.TOOL_SCHEMA,
]

__all__ = ["TOOLS", "TOOL_SCHEMAS"]
