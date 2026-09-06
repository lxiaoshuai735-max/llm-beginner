"""Regression tests for general, model-directed Task 5 tool calls."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import ReActAgent  # noqa: E402


class ScriptedAgent(ReActAgent):
    """Use deterministic model responses while exercising the real agent loop."""

    def __init__(self, responses: list[str], max_steps: int = 6) -> None:
        super().__init__(api_url="http://unused.invalid", model="scripted", max_steps=max_steps)
        self.responses = iter(responses)
        self.message_history: list[list[dict[str, str]]] = []

    def _chat(self, messages: list[dict[str, str]]) -> str:
        self.message_history.append([dict(message) for message in messages])
        return next(self.responses)


class AgentArgumentRegressionTests(unittest.TestCase):
    def run_scripted(self, task: str, action: str, final: str) -> dict:
        agent = ScriptedAgent([action, f"Thought: 已依据工具结果作答。\nFinal Answer: {final}"])
        result = agent.run(task)
        self.assertTrue(result["success"], result)
        self.assertEqual(len(result["steps"]), 1)
        return result

    def test_sqrt_81_is_not_rewritten(self) -> None:
        result = self.run_scripted(
            "请使用计算器计算 sqrt(81)。",
            'Thought: 计算平方根。\nAction: calculator\nAction Input: {"expression": "sqrt(81)"}',
            "sqrt(81) = 9。",
        )
        step = result["steps"][0]
        self.assertEqual(step["tool_input"], {"expression": "sqrt(81)"})
        self.assertIn("result=9", step["observation"])
        self.assertNotIn("2026", str(step))

    def test_system_prompt_exposes_tool_parameter_schemas(self) -> None:
        agent = ScriptedAgent([])
        self.assertIn('"expression"', agent.system_prompt)
        self.assertIn('"code"', agent.system_prompt)
        self.assertIn('"pattern"', agent.system_prompt)
        self.assertIn('"query"', agent.system_prompt)

    def test_primes_below_10_are_not_rewritten(self) -> None:
        code = "print(sum(n for n in range(2, 10) if all(n % d for d in range(2, int(n ** 0.5) + 1))))"
        result = self.run_scripted(
            "请用 Python 求 10 以内质数之和。",
            f'Thought: 编写通用质数判断。\nAction: python_sandbox\nAction Input: {{"code": "{code}"}}',
            "10 以内质数之和是 17。",
        )
        step = result["steps"][0]
        self.assertEqual(step["tool_input"], {"code": code})
        self.assertIn("17", step["observation"])
        self.assertNotIn("range(2,100)", str(step))

    def test_model_selected_file_directory_is_preserved(self) -> None:
        arguments = {
            "pattern": "REGRESSION_PATH_MARKER",
            "dir": "eval/fixtures/custom_location",
        }
        result = self.run_scripted(
            "在 eval/fixtures/custom_location 中搜索 REGRESSION_PATH_MARKER。",
            "Thought: 按用户给定目录搜索。\nAction: file_search\n"
            'Action Input: {"pattern": "REGRESSION_PATH_MARKER", "dir": "eval/fixtures/custom_location"}',
            "已在指定目录找到标记。",
        )
        step = result["steps"][0]
        self.assertEqual(step["tool_input"], arguments)
        self.assertIn("eval/fixtures/custom_location/notes.txt", step["observation"])
        self.assertNotIn("data/agent-fixtures", str(step))

    def test_wiki_query_is_not_rewritten_to_an_evaluation_topic(self) -> None:
        query = "Ada Lovelace analytical engine"
        action = (
            "Thought: 查询指定主题。\nAction: wiki\n"
            'Action Input: {"query": "Ada Lovelace analytical engine"}'
        )
        with patch.dict("src.agent.TOOLS", {"wiki": lambda args: f"query={args['query']}"}):
            result = self.run_scripted("请查询 Ada Lovelace 与分析机。", action, "查询已完成。")
        step = result["steps"][0]
        self.assertEqual(step["tool_input"], {"query": query})
        self.assertEqual(step["observation"], f"query={query}")

    def test_malformed_response_does_not_trigger_a_hidden_tool_call(self) -> None:
        agent = ScriptedAgent(
            [
                "我认为答案是 9。",
                'Thought: 按原题重试。\nAction: calculator\nAction Input: {"expression": "sqrt(49)"}',
                "Thought: 已获得工具结果。\nFinal Answer: sqrt(49) = 7。",
            ]
        )
        result = agent.run("请计算 sqrt(49)。")
        self.assertTrue(result["success"], result)
        self.assertNotIn("tool", result["steps"][0])
        self.assertEqual(result["steps"][1]["tool_input"], {"expression": "sqrt(49)"})
        self.assertIn("result=7", result["steps"][1]["observation"])

    def test_premature_final_requires_model_to_choose_the_tool(self) -> None:
        agent = ScriptedAgent(
            [
                "Thought: 直接回答。\nFinal Answer: 12。",
                'Thought: 应先计算。\nAction: calculator\nAction Input: {"expression": "5+8"}',
                "Thought: 使用计算结果。\nFinal Answer: 5+8=13。",
            ]
        )
        result = agent.run("请计算 5+8。")
        self.assertTrue(result["success"], result)
        self.assertNotIn("tool", result["steps"][0])
        self.assertEqual(result["steps"][1]["tool_input"], {"expression": "5+8"})
        self.assertEqual(result["final_answer"], "5+8=13。")


if __name__ == "__main__":
    unittest.main()
