# Task 2 实验观察

本实验实现了字节级 BPE、RoPE、带 KV cache 的 decoder-only Transformer，以及 greedy、temperature、top-k、top-p 四种采样。最终词表大小为 320，其中包含 64 条实际 merge 规则；模型采用 8 层、8 头、隐藏维度 256，约 639 万参数。验证集困惑度在第 16 轮达到最低值约 28.1，随后训练损失继续下降而验证困惑度回升，显示唐诗小语料上出现过拟合。自检测得困惑度 28.16，低于 50 的通过线；KV cache 与全量前向的最大 logits 误差为 2.38e-6。生成结果中 greedy 重复性更强，temperature、top-k 和 top-p 增加了用词变化，但受限于约 49KB 语料，较长文本的语义连贯性仍有限。可通过扩大语料、增加正则化并延长有效上下文进一步改善。
