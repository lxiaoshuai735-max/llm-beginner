# Task 5：ReAct 工具智能体

本任务实现一个由本地 Qwen 服务驱动的手写 ReAct 循环，提供安全计算器、受限 Python 沙箱、项目内文件检索和 Wikipedia 查询四个工具。

## 关键设计

- 模型必须输出 `Thought / Action / Action Input`，获得工具证据后才能输出 `Final Answer`。
- `Action Input` 是唯一的工具参数来源；Agent 不会根据题目关键词替换数字、范围、文件路径或百科主题。
- 参数错误、未知工具、重复调用和输出格式错误都会作为 Observation 反馈给模型，由模型重新选择工具或修正参数。
- Trace 分开保存模型答案和每一步工具 Observation，不把工具输出自动拼接到最终答案中，避免关键词评分被证据文本抬高。
- Wikipedia 工具访问真实的中英文 Wikipedia API，不使用针对评测题目的内置答案；运行时需要外网。

## 环境

```bash
pip install -r requirements.txt
export QWEN_API_URL=http://127.0.0.1:11434/api/chat
export QWEN_MODEL=qwen2.5:7b-instruct
```

## 回归测试

```bash
python -m unittest discover -s eval -p "test_*.py" -v
```

测试覆盖 `sqrt(81)`、10 以内质数之和、自定义文件目录、非固定 Wiki 主题、格式错误恢复，以及模型过早给出最终答案的情形。测试使用确定性的模型响应，但调用真实的计算器、Python 沙箱和文件搜索工具，因此不要求 Qwen 服务在线。

## 生成评测 Trace

默认运行任务文件中的全部题目：

```bash
python eval/generate_traces.py
```

也可以显式选择题目和输出位置：

```bash
python eval/generate_traces.py \
  --tasks data/tasks.json \
  --ids 1,5,8,10 \
  --output eval/traces.json
```

修复前生成的成功率和 Trace 不能用于证明修复后实现的效果。修改 Agent 后，应重新启动 Qwen 服务、重新运行完整题集，并以新生成的结果更新实验报告。
