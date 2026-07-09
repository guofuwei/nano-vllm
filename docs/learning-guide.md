# Nano-vLLM 项目学习指南

这份指南面向第一次阅读 nano-vLLM 的同学，目标是用较短路径理解它如何把一个 HuggingFace Qwen3 模型变成支持连续批处理、分页 KV cache、prefix cache、张量并行和 CUDA Graph 的离线推理引擎。

## 先建立整体地图

建议先从 `README.md` 和 `example.py` 开始，确认项目的使用方式：

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/path/to/Qwen3-0.6B", enforce_eager=True)
outputs = llm.generate(["Hello, Nano-vLLM."], SamplingParams(max_tokens=64))
```

这段 API 背后有四条主线：

1. `nanovllm/engine/llm_engine.py`：对外入口，负责 tokenizer、请求添加、推理循环和多进程初始化。
2. `nanovllm/engine/scheduler.py`：调度器，决定每一步做 prefill 还是 decode，以及哪些序列进入 batch。
3. `nanovllm/engine/model_runner.py`：把调度结果转换成 CUDA 张量，运行模型、采样 token，并管理 KV cache 与 CUDA Graph。
4. `nanovllm/models/qwen3.py` 与 `nanovllm/layers/`：模型结构、注意力、并行线性层、采样器等底层算子。

## 推荐阅读顺序

### 1. 从用户接口读到推理循环

先看 `nanovllm/llm.py`，它只是继承 `LLMEngine`。真正的入口在 `nanovllm/engine/llm_engine.py`：

- `__init__`：读取配置、创建 tensor parallel 子进程、加载 tokenizer、创建 scheduler 和 model runner。
- `add_request`：把字符串 prompt 编码成 token，并包装成 `Sequence`。
- `step`：调用 scheduler 选 batch，交给 model runner 执行，再让 scheduler 更新序列状态。
- `generate`：循环执行 `step`，直到所有请求完成。

读完这里，你应该能回答：一个 prompt 是怎样进入系统、怎样被不断生成 token、最后怎样返回文本的。

### 2. 理解 Sequence 是调度的最小单位

接着看 `nanovllm/engine/sequence.py`。`Sequence` 同时保存 token、采样参数、状态和 KV cache block 表。

重点字段：

- `token_ids`：prompt 和已生成 completion 的完整 token 序列。
- `num_prompt_tokens`：用于区分 prompt 与 completion。
- `num_cached_tokens`：已经写入或命中 KV cache 的 token 数。
- `num_scheduled_tokens`：本轮被调度执行的 token 数。
- `block_table`：逻辑 block 到物理 KV cache block 的映射。

这里的关键思想是：调度器不直接操作张量，而是修改 `Sequence` 的状态；`ModelRunner` 再根据这些状态构造真实的 GPU 输入。

### 3. 看调度器如何区分 prefill 和 decode

打开 `nanovllm/engine/scheduler.py`，重点读 `schedule` 和 `postprocess`。

prefill 阶段：

- 从 waiting 队列取序列。
- 尝试通过 `BlockManager.can_allocate` 命中 prefix cache。
- 给序列分配 KV cache block。
- 受 `max_num_batched_tokens` 限制时，可能只处理 prompt 的一部分，也就是 chunked prefill。

decode 阶段：

- 从 running 队列取序列。
- 每条序列本轮只生成一个 token。
- 如果 KV block 不够，会抢占部分序列，把它们放回 waiting。

`postprocess` 会把新写满的 block 登记到 prefix cache，追加采样出来的新 token，并在遇到 EOS 或达到 `max_tokens` 时释放 KV cache。

### 4. 读 BlockManager 理解分页 KV cache

`nanovllm/engine/block_manager.py` 是理解 vLLM 类系统的核心文件之一。

你可以把 KV cache 看成一批固定大小的物理页：

- `free_block_ids` 保存空闲页。
- `used_block_ids` 保存正在使用的页。
- `block_table` 存在每条 `Sequence` 里，记录这条序列用了哪些物理页。
- `hash_to_block_id` 用于 prefix cache，用 token block 的链式 hash 找到可复用的缓存页。
- `ref_count` 支持多条序列共享同一个 prefix block。

建议跟着一个例子手推：两条 prompt 共享相同前缀时，第一条 prefill 后会登记 block hash，第二条 prefill 时会复用这些 block，只计算没有命中的后续 token。

### 5. 进入 ModelRunner 看张量如何准备

`nanovllm/engine/model_runner.py` 是连接调度层和模型层的桥。

重点函数：

- `prepare_prefill`：为 prompt token 构造 `input_ids`、`positions`、变长 attention 的 `cu_seqlens`，以及写 KV cache 的 `slot_mapping`。
- `prepare_decode`：每条 running 序列只输入最后一个 token，并准备 `context_lens` 和 `block_tables`。
- `run_model`：prefill 通常走 eager，decode 在条件允许时走 CUDA Graph。
- `capture_cudagraph`：提前捕获多个 batch size 的 decode 图，减少每步 decode 的调度开销。
- `allocate_kv_cache`：根据显存预算计算可分配的 KV cache block 数。

读这里时要和 `nanovllm/utils/context.py` 一起看。项目通过一个全局 context 把 `slot_mapping`、`block_tables`、`cu_seqlens` 等信息传给 attention 层，避免把这些参数层层传递。

### 6. 读 Attention 和 Qwen3 模型

`nanovllm/layers/attention.py` 做了两件关键事：

- 用 Triton kernel 把本轮新产生的 K/V 写入分页 KV cache。
- 根据 prefill/decode 选择 `flash_attn_varlen_func` 或 `flash_attn_with_kvcache`。

`nanovllm/models/qwen3.py` 实现了 Qwen3 的主体结构：

- `Qwen3Attention`：QKV 投影、RoPE、attention、输出投影。
- `Qwen3MLP`：合并 gate/up 投影，再做 SiLU and multiply。
- `Qwen3DecoderLayer`：RMSNorm、残差、attention、MLP。
- `Qwen3ForCausalLM`：模型主体加并行 LM head。

再配合 `nanovllm/layers/linear.py` 阅读，可以理解 tensor parallel 是如何切分权重的：

- Column parallel 按输出维度切分。
- Row parallel 按输入维度切分，并在输出后 all-reduce。
- QKV 和 gate/up 权重会在加载时合并到项目自定义层中。

## 建议的调试路线

第一次运行建议设置：

```python
llm = LLM("/path/to/Qwen3-0.6B", enforce_eager=True, tensor_parallel_size=1)
```

这样可以先避开 CUDA Graph 和多进程张量并行，把注意力放在主流程上。等单卡 eager 跑通后，再尝试：

```python
llm = LLM("/path/to/Qwen3-0.6B", enforce_eager=False, tensor_parallel_size=1)
```

最后再研究 `tensor_parallel_size > 1` 的共享内存广播、NCCL 初始化和并行线性层权重切分。

## 可以动手做的小练习

1. 在 `Scheduler.schedule` 中打印每轮是 prefill 还是 decode，以及被调度的 `seq_id`。
2. 构造两条有相同长前缀的 prompt，观察 `BlockManager.can_allocate` 返回的 cached block 数量。
3. 把 `kvcache_block_size` 调小，观察 block 分配、hash 登记和抢占更频繁时的行为。
4. 关闭 CUDA Graph，对比 `bench.py` 的吞吐变化。
5. 在 `linear.py` 中打印每个 rank 加载的权重 shard shape，理解张量并行切分。

## 读代码时最重要的心智模型

Nano-vLLM 的核心不是“逐条请求跑模型”，而是维护一批 `Sequence`，不断在 prefill 和 decode 之间调度，并把每条序列的上下文放进分页 KV cache。调度器只负责决定做什么，`BlockManager` 负责 KV cache 页的生命周期，`ModelRunner` 负责把决定变成 GPU 张量，模型层只关心如何用这些张量完成一次前向计算。

只要抓住这条链路：

```text
LLM.generate -> Scheduler.schedule -> ModelRunner.run -> Attention/KV cache -> Scheduler.postprocess
```

再回头看 prefix cache、chunked prefill、CUDA Graph 和 tensor parallel，这些优化都会变成主流程上的局部增强，而不是彼此割裂的概念。
