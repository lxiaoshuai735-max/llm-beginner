# Task 6 实验记录

## 修订后的 DoD checklist

- [x] `MCPStdioClient` 启动独立的 `src.mcp_server` 子进程并完成 MCP 初始化
- [x] 主 Agent 从 MCP `list_tools` 获取工具定义，所有主循环工具调用均通过 MCP `call_tool`
- [x] 主 Agent 不再导入或调用 `TOOL_FUNCTIONS`，服务端也不再提供该捷径映射
- [x] 工具参数中的仓库根目录由宿主注入，不交给模型控制
- [x] 文件工具执行路径规整和仓库边界校验，命令不使用 `shell=True`
- [x] SkillLoader 渐进加载含 YAML front-matter 的 Skill
- [x] 代码搜索与测试诊断 Subagent 使用独立消息上下文，仓库访问也复用 MCP stdio 会话
- [x] 运行前记录基线 diff，完成条件要求最终 diff 与基线不同
- [x] 完成顺序严格要求：真实改动 → 改动后的测试通过 → 测试后的非空 diff
- [x] 模型不可用、协议持续错误或达到步数上限时，Trace 明确返回 `status=failed`
- [x] 无效 JSON 会收到严格格式重试提示；连续重复的只读动作有有限容错，超过上限仍明确失败
- [x] 删除面向固定函数和固定错误语句的字符串替换及最终兜底修复
- [x] 增加变化 bug、禁用模型和真实 MCP stdio 往返测试
- [x] MCP 服务端启动的 Git/pytest 子进程不继承协议 stdin，避免子进程阻塞 stdio 通道
- [x] 每次测试使用独立 Python 字节码缓存，避免同秒同长度源码改动误用旧 `.pyc`

## 调用链

```text
CodingAgent
  └─ MCPStdioClient（单次会话）
       └─ stdio JSON-RPC
            └─ FastMCP server
                 ├─ 只读 Subagent：搜索 / 读取 / 基线测试
                 └─ 主循环：文件、pytest 与 git 工具
```

主模型只能看到不含 `repo_path` 的工具参数；客户端在发送 MCP 请求时注入已经解析过的仓库绝对路径。主循环与两个 Subagent 的目标仓库读取、测试和 Git 操作均通过同一 stdio 会话。Trace 记录实际客户端的 `transport_name`，生产运行应为 `mcp-stdio`，注入测试替身时则如实记录为 `mcp-test-double`。

## 可验证失败语义

返回值新增 `status`、`success`、`failure_reason`、`baseline_patch` 和 `tool_transport`。一次普通的 `write_file` 返回并不足以视为修改：写入内容与原文件相同时服务端返回 `unchanged`。系统记录最后一次真实修改、最后一次成功测试和最后一次 diff 的步骤号，只有满足 `mutation < passing_test < final_diff`，且最终 diff 非空并区别于运行前基线，任务才是 `succeeded`。因此预存脏改动、写入后还原、在测试前读取 diff 均不能拼成成功结果。模型异常时允许的自动恢复仅限读文件、运行测试和记录 diff，不包含任何猜测性编辑；没有模型生成的修复时任务必然失败。

## 回归检查

依赖较少的检查：

```bash
python eval/regression_self_check.py
```

AutoDL 执行结果：

```text
REGRESSION_SELF_CHECK_OK
```

完整检查（安装 `requirements.txt` 后）：

```bash
python -m pytest -q tests
```

完整回归测试连续运行两次，均为 `9 passed`。测试覆盖：

1. 将缺陷变化为 `multiply` 错用加法，确认闭环不依赖原 toy 缺陷的固定字符串。
2. 禁用模型客户端，确认返回失败、patch 为空且源文件完全不变。
3. 使用真实 `MCPStdioClient` 启动服务器，校验工具发现、隐藏宿主参数和 stdio 读文件往返。
4. 在一次无效 JSON 和一次重复只读动作之后继续完成变化 bug，确认协议容错不会引入确定性代码修复。
5. 使用真实 MCP stdio 跑通 `read → write → test → diff` 完整闭环。
6. 使用变化后的乘法缺陷完整跑过真实 MCP stdio Agent 闭环，并核对 Trace 中每个主工具步骤的传输标记。
7. 在仓库已有脏 diff 时先写入再还原，确认最终状态与基线相同会失败。
8. 故意先取 diff 再跑测试，确认 Agent 会在测试后重新取 diff 才允许成功。

## 真实 Qwen 闭环

2026-09-06 在 AutoDL 启动 `qwen2.5-coder:7b-instruct` 后运行 `python run_toy_agent.py`。结果为 `status=succeeded`、`tests_passed=true`、`tool_transport=mcp-stdio`，Trace 共 6 步：两个只读 Subagent、`read_file`、模型生成的 `write_file`、`run_tests`、`git_diff`。最终 patch 非空，只修改实现文件，语义改动为修正加法错误；`eval/toy_repo_trace.json` 保存了完整模型与 MCP 证据。

`python eval/run.py` 默认把这份真实 Trace 的 patch 重新应用到干净 toy 仓库并运行测试，因此评测可稳定复现且不会受模型采样波动影响；设置 `TASK6_REGENERATE_TRACE=1` 时则会重新调用当前 Qwen 服务并覆盖 Trace。最终自检中 Trace 重放、3 项 toy 测试、MCP 传输和非空 patch 检查全部通过；SWE-bench Lite 样本因数据文件未下载而明确跳过。

## 实验观察

初版虽然表面上能够修复 toy-repo，却把服务端函数直接导入主循环，而且在模型输出失败后按题目文本执行固定字符串替换。这意味着即使 Qwen 没有给出有效工具决策，结果仍可能显示测试通过，无法证明 Coding Agent 的模型与 MCP 链路有效。本次修订让主循环和 Subagent 共用真实 stdio MCP 会话，并取消所有确定性代码修复，仅保留不会改变仓库的证据收集动作。新的停机条件不仅检查测试和 diff，还保存初始工作树状态并校验严格步骤顺序。变化缺陷自检能够成功，说明状态机不再绑定原题；禁用模型自检明确失败且文件未变化；脏工作树和乱序检查也无法再构造伪成功。远端复跑还发现并修复了 pytest 可能复用同秒生成旧字节码的竞态。最终真实 Qwen 闭环成功，证明模型决策、MCP 调用、测试和 diff 验收链路均实际参与，而非兜底代码代做。
