"""A compact handwritten ReAct agent backed by the local Qwen service."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from .tools import TOOLS, TOOL_SCHEMAS

_SYSTEM_PROMPT = """你是一个严谨的 ReAct 工具智能体。必须先用工具获得证据，不能凭记忆直接作答。
每次响应只能使用以下两种格式之一：

Thought: 简短说明下一步
Action: calculator|python_sandbox|file_search|wiki
Action Input: 单行 JSON 对象

或在证据充分时：
Thought: 简短总结
Final Answer: 给用户的完整答案

工具说明：
{tool_text}

规则：
1. 数值计算使用 calculator；算法、函数测试或指定 Python 时使用 python_sandbox。
2. 项目文件问题使用 file_search；百科事实使用 wiki。
3. 复合题按顺序调用多个工具。Python 沙箱不允许 import，代码必须 print 结果。
4. 仔细读取 Observation；工具报错时修正输入再试，不要重复完全相同的失败调用。
5. 最终答案必须包含题目要求的数字、名称、布尔值或文件名，并使用中文简洁回答。
6. 不要输出 Markdown 代码围栏，不要伪造 Observation。"""


class ReActAgent:
    """Run a Thought → Action → Observation loop until a final answer is produced."""

    def __init__(
        self,
        api_url: str | None = None,
        model: str | None = None,
        max_steps: int = 8,
        timeout: int = 120,
    ) -> None:
        self.api_url = api_url or os.getenv("QWEN_API_URL", "http://127.0.0.1:11434/api/chat")
        self.model = model or os.getenv("QWEN_MODEL", "qwen2.5:7b-instruct")
        self.max_steps = max_steps
        self.timeout = timeout
        tool_text = "\n".join(
            (
                f"- {item['function']['name']}: {item['function']['description']}; "
                f"Action Input JSON Schema="
                f"{json.dumps(item['function']['parameters'], ensure_ascii=False)}"
            )
            for item in TOOL_SCHEMAS
        )
        self.system_prompt = _SYSTEM_PROMPT.format(tool_text=tool_text)

    def _chat(self, messages: list[dict[str, str]]) -> str:
        import requests

        response = requests.post(
            self.api_url,
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.05, "num_predict": 320},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        content = payload.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"model response has no message.content: {payload}")
        return content.strip()

    @staticmethod
    def _parse(text: str) -> dict[str, Any]:
        final = re.search(r"Final\s*Answer\s*[:：]\s*(.+)", text, flags=re.I | re.S)
        thought_match = re.search(
            r"Thought\s*[:：]\s*(.*?)(?=\n\s*(?:Action|Final\s*Answer)\s*[:：])",
            text,
            flags=re.I | re.S,
        )
        thought = thought_match.group(1).strip() if thought_match else ""
        if final:
            return {"kind": "final", "thought": thought, "answer": final.group(1).strip()}
        action = re.search(r"Action\s*[:：]\s*([A-Za-z_][\w-]*)", text, flags=re.I)
        action_input = re.search(r"Action\s*Input\s*[:：]\s*", text, flags=re.I)
        if not action or not action_input:
            raise ValueError("response must contain Action and Action Input, or Final Answer")
        raw_json = text[action_input.end():].strip()
        decoder = json.JSONDecoder()
        try:
            arguments, end = decoder.raw_decode(raw_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Action Input is not valid JSON: {exc.msg}") from exc
        if raw_json[end:].strip():
            raise ValueError("Action Input must contain exactly one JSON object")
        if not isinstance(arguments, dict):
            raise ValueError("Action Input must be a JSON object")
        return {
            "kind": "action",
            "thought": thought,
            "tool": action.group(1).lower(),
            "arguments": arguments,
        }

    @staticmethod
    def _has_successful_tool_call(steps: list[dict[str, Any]]) -> bool:
        """Return whether at least one model-requested tool call succeeded."""
        return any(
            step.get("tool")
            and not str(step.get("observation", "")).startswith("ToolError")
            for step in steps
        )

    def run(self, task: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
        steps: list[dict[str, Any]] = []
        repeated: dict[str, int] = {}
        last_error = ""

        for index in range(1, self.max_steps + 1):
            try:
                model_text = self._chat(messages)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                messages.append(
                    {
                        "role": "user",
                        "content": f"Observation: 模型服务调用失败：{last_error}\n请在服务恢复后继续。",
                    }
                )
                steps.append({"step": index, "thought": "模型服务调用失败", "observation": last_error})
                continue
            try:
                parsed = self._parse(model_text)
            except (TypeError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                messages.extend(
                    [
                        {"role": "assistant", "content": model_text},
                        {
                            "role": "user",
                            "content": (
                                f"Observation: {last_error}\n"
                                "上一条响应无法执行。请重新阅读原始问题，严格按格式给出工具及其参数；"
                                "不要复用其他问题中的数字、范围或路径。"
                            ),
                        },
                    ]
                )
                steps.append({"step": index, "thought": "等待模型修复输出", "observation": last_error})
                continue

            if parsed["kind"] == "final":
                answer = str(parsed["answer"]).strip()
                if not self._has_successful_tool_call(steps):
                    messages.extend(
                        [
                            {"role": "assistant", "content": model_text},
                            {
                                "role": "user",
                                "content": (
                                    "Observation: 尚无成功的工具调用，不能直接给出 Final Answer。"
                                    "请根据原始问题自行选择工具并提供准确的 Action Input。"
                                ),
                            },
                        ]
                    )
                    steps.append(
                        {
                            "step": index,
                            "thought": parsed.get("thought", ""),
                            "observation": "ToolError: final answer requires successful tool evidence",
                        }
                    )
                    continue
                return {"steps": steps, "final_answer": answer, "success": bool(answer)}

            tool_name = str(parsed["tool"])
            # Preserve Action Input exactly as the model supplied it.  Tool
            # errors are returned as observations so the model, rather than
            # task-specific host logic, decides how to correct an argument.
            arguments = parsed["arguments"]
            signature = json.dumps([tool_name, arguments], ensure_ascii=False, sort_keys=True)
            repeated[signature] = repeated.get(signature, 0) + 1
            if tool_name not in TOOLS:
                observation = f"ToolError: unknown tool {tool_name!r}; choose one of {', '.join(TOOLS)}"
            elif repeated[signature] > 1:
                observation = "ToolError: this exact call was already made; use its prior result and finish or choose a different action."
            else:
                try:
                    observation = TOOLS[tool_name](arguments)
                except Exception as exc:
                    observation = f"ToolError: {type(exc).__name__}: {exc}"
            steps.append(
                {
                    "step": index,
                    "thought": parsed.get("thought", ""),
                    "tool": tool_name,
                    "tool_input": arguments,
                    "observation": observation,
                }
            )
            messages.extend(
                [
                    {"role": "assistant", "content": model_text},
                    {"role": "user", "content": f"Observation: {observation}"},
                ]
            )

        try:
            messages.append({"role": "user", "content": "已到步骤上限。只输出 Thought 和 Final Answer，并综合已有 Observation。"})
            text = self._chat(messages)
            parsed = self._parse(text)
            if parsed["kind"] == "final" and self._has_successful_tool_call(steps):
                answer = str(parsed["answer"]).strip()
                return {"steps": steps, "final_answer": answer, "success": bool(answer)}
            last_error = "model did not provide a final answer at the step limit"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        return {"steps": steps, "final_answer": f"任务未完成：{last_error}", "success": False}
