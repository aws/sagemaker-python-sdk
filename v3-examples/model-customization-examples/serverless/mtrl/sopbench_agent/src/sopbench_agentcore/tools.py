"""SOP-Bench tool adapters: OpenAI function-calling schemas and an executor.

Both helpers take a loaded Benchmark (amazon_sop_bench.benchmarks.load_benchmark). Its `tools`
is a ToolManager exposing get_tool_specs() (Bedrock format) and
execute_tool(name, args) -> ToolCall(.result, .success, .error).
"""

import json
from typing import Any, Callable, Dict, List


def create_openai_tools(benchmark) -> List[Dict[str, Any]]:
    """Convert SOP-Bench's Bedrock-format tool specs to the OpenAI function-calling schema."""
    tools = []
    for spec in benchmark.tools.get_tool_specs():
        ts = spec.get("toolSpec", spec)
        schema = ts.get("inputSchema", {})
        params = schema.get("json", schema) if isinstance(schema, dict) else {}
        if not isinstance(params, dict) or "type" not in params:
            params = {"type": "object", "properties": {}}
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": ts["name"],
                    "description": ts.get("description", ""),
                    "parameters": params,
                },
            }
        )
    return tools


def make_tool_executor(benchmark) -> Callable[[str, Dict[str, Any]], str]:
    """Return a closure(tool_name, args) -> result string over the benchmark's ToolManager."""
    tm = benchmark.tools

    def _execute(tool_name: str, arguments: Dict[str, Any]) -> str:
        result = tm.execute_tool(tool_name, arguments)
        if result.success:
            if isinstance(result.result, str):
                return result.result
            return json.dumps(result.result, default=str)
        return f"Tool execution failed. Error: {result.error}"

    return _execute
