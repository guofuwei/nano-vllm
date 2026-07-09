import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    # 本地 HuggingFace 模型目录，里面应包含 config.json、tokenizer 和权重文件。
    model: str
    # 一个调度 step 中最多处理的 token 数，主要限制 prefill 阶段的 batch 总长度。
    max_num_batched_tokens: int = 16384
    # 一个调度 step 中最多同时处理的序列数，影响并发度和 decode batch size。
    max_num_seqs: int = 512
    # 单条序列允许的最大上下文长度，会被模型自身的 max_position_embeddings 截断。
    max_model_len: int = 4096
    # 用于 KV cache 的显存比例，ModelRunner 会据此估算可分配的 cache block 数。
    gpu_memory_utilization: float = 0.2
    # 张量并行进程数，通常等于参与推理的 GPU 数量。
    tensor_parallel_size: int = 1
    # 是否强制使用 eager 执行；为 False 时 decode 可使用 CUDA Graph 加速。
    enforce_eager: bool = False
    # HuggingFace AutoConfig 对象，在 __post_init__ 中从 model 目录加载。
    # 对 Qwen3 模型来说，常见字段含义如下：
    # - architectures: HuggingFace 模型类名，例如 Qwen3ForCausalLM。
    # - model_type: 模型类型标识，AutoConfig 用它选择具体 Config 类，例如 qwen3。
    # - vocab_size: tokenizer 词表大小，也是 embedding 和 lm_head 的输出维度。
    # - hidden_size: 每个 token 的隐藏状态维度。
    # - intermediate_size: MLP 中间层维度，Qwen3MLP 的 gate/up 投影会用到。
    # - num_hidden_layers: decoder layer 层数，也决定每个 KV cache block 要保存多少层。
    # - num_attention_heads: query attention head 总数，tensor parallel 时按 rank 切分。
    # - num_key_value_heads: key/value head 总数，GQA/MQA 和 KV cache 形状会用到。
    # - head_dim: 单个 attention head 的维度；缺省时通常等于 hidden_size // num_attention_heads。
    # - hidden_act: MLP 激活函数名，本实现要求为 silu。
    # - rms_norm_eps: RMSNorm 的 epsilon，避免归一化时除零。
    # - attention_bias: Q/K/V 线性层是否带 bias；Qwen3 无 bias 时会额外使用 q_norm/k_norm。
    # - attention_dropout: attention dropout 概率；推理时通常为 0。
    # - max_position_embeddings: 模型训练支持的最大位置长度，max_model_len 不能超过它。
    # - rope_theta: RoPE 的 base/theta，旧版 transformers 可能直接暴露这个字段。
    # - rope_scaling/rope_parameters: RoPE 参数字典，新版 transformers 会把 rope_theta 放在这里。
    # - sliding_window/use_sliding_window/max_window_layers: 滑动窗口注意力设置，本实现按全量注意力处理。
    # - tie_word_embeddings: 是否共享 token embedding 和 lm_head 权重。
    # - torch_dtype/dtype: 权重和默认计算精度，AutoConfig 会转成 torch.dtype，如 torch.bfloat16。
    # - bos_token_id/eos_token_id/pad_token_id: tokenizer 特殊 token id；eos 另由 tokenizer 写入本 Config。
    # - use_cache: HuggingFace 生成时是否使用 KV cache；本项目自己管理分页 KV cache。
    # - initializer_range/transformers_version: 训练初始化和配置来源版本信息，推理路径通常不直接使用。
    hf_config: AutoConfig | None = None
    # tokenizer 的 EOS token id，在 LLMEngine 初始化 tokenizer 后写入。
    eos: int = -1
    # KV cache 的分页大小；每个 Sequence 的 token 会按这个大小切成 block。
    kvcache_block_size: int = 256
    # 实际可用的 KV cache block 数，在 ModelRunner.allocate_kv_cache 中根据显存计算。
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        # 该实现只支持从本地目录加载模型。
        assert os.path.isdir(self.model)
        # 当前 attention/KV cache 实现要求 block size 与底层分页粒度对齐。
        assert self.kvcache_block_size % 256 == 0
        # 简单限制 tensor parallel 规模，避免启动超出预期数量的进程。
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # 不能超过模型训练时支持的位置编码长度。
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
