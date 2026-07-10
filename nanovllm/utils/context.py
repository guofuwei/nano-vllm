from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    """一次模型前向传播共享的注意力元数据。

    Attention 和 LM head 不将这些随 batch 变化的参数逐层传递，而是在前向期间
    从该对象读取。这个设计依赖单个 ModelRunner 串行执行前向；它不是线程安全的。
    """

    # True 为提示词预填充；False 为每条序列生成一个 token 的 decode 阶段。
    is_prefill: bool = False
    # 变长 prefill 中每条序列在拼接后的 Q/K 张量里的起止偏移量（前缀和）。
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    # 本 batch 中 Q/K 的最大序列长度，供 FlashAttention 选择执行配置。
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    # 当前 token 的 K/V 应写入分页 KV cache 的物理槽位。
    slot_mapping: torch.Tensor | None = None
    # decode 时每条序列包含当前 token 在内的上下文长度。
    context_lens: torch.Tensor | None = None
    # 每条序列的逻辑块号到 KV cache 物理块号的映射；也用于 prefix cache。
    block_tables: torch.Tensor | None = None

# 模块级状态使各层能在不改变模型 forward 签名的情况下共享 batch 元数据。
_CONTEXT = Context()


def get_context():
    # 返回当前前向传播共享的注意力上下文。
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    # 在 ModelRunner 准备好 batch 元数据后，将其写入全局上下文供 Attention 读取。
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    # 清空本轮推理上下文，避免下一次前向传播读到旧的 batch 元数据。
    global _CONTEXT
    _CONTEXT = Context()
