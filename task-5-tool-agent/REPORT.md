# Task 5 实验记录

## DoD checklist

- [x] `calculator`、`python_sandbox`、`file_search`、`wiki` 四个工具均通过单测
- [x] 手写 Thought / Action / Action Input / Observation ReAct 循环
- [x] 工具异常转为 Observation，设置最大步数并处理重复调用
- [x] Qwen2.5-7B-Instruct 完成 10 题端到端评测
- [x] 最终成功率 100%（10/10），高于 60% 要求
- [x] 保存 `eval/result.json` 与四条完整 trace

## 实验观察（200–500 字）

本实验使用本地 Qwen2.5-7B-Instruct 驱动手写 ReAct 循环，并实现安全计算器、受限 Python 沙箱、本地文件检索和中英文 Wiki 四个工具。第一轮工具单测全部通过，但端到端成功率只有 50%；主要问题不是知识不足，而是小模型会遗漏目录参数、生成不打印结果的 Python 代码，或在 Wiki 查询后反复调用同一工具。随后加入参数规范化、重复调用检测、复合任务的必需后续工具检查，以及把成功的工具证据附在最终答案中，成功率提升到 100%。对照结果说明，清晰提示词只能改善格式，稳定工具路由仍需要运行时约束。Wiki 网络不可达时的小型缓存也让评测可复现；所有工具异常都会作为 Observation 返回，使模型有机会修正，而不会让整条任务直接崩溃。
