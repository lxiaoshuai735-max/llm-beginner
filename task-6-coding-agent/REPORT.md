# Task 6 实验记录

## DoD checklist

- [x] MCP server 可独立通过 stdio 启动，并暴露 6 个工具
- [x] 所有文件工具执行路径规整和仓库边界校验，命令不使用 `shell=True`
- [x] SkillLoader 渐进加载 3 个含 YAML front-matter 的 Skill
- [x] 代码搜索与测试诊断两个 Subagent 使用独立消息上下文
- [x] CodingAgent 返回含 `steps`、`patch`、`tests_passed` 的 Trace
- [x] Qwen2.5-Coder-7B-Instruct 自动修复 toy-repo，pytest 3/3 通过

## 实验观察（200–500 字）

本实验部署了 Qwen2.5-Coder-7B-Instruct 的 Q4_K_M 量化版本，并在独立的 11435 端口提供本地接口，避免影响 Task 4 服务。底层 MCP server 暴露读写文件、运行测试、搜索、查看和应用补丁等六个工具；中层 SkillLoader 只先读取元数据，命中后再加载正文；顶层两个 Subagent 分别进行代码定位和测试诊断，只把摘要交给主智能体。真实运行中，Coder 准确发现 `calculator.add` 使用了减法，但两次输出的 JSON 存在格式错误。主循环把解析异常转为 Observation，并按“读代码—跑基线测试—最小修改—再次测试—检查 diff”的恢复策略继续执行，最终七步完成修复，三个测试全部通过。这个现象说明模型的代码理解能力和工具协议稳定性是两件事；明确停机条件、路径保护及确定性恢复逻辑，能显著提高 Coding Agent 的可复现性。
