import pickle
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event

import torch
import torch.distributed as dist

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.context import get_context, reset_context, set_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        # 初始化单个 tensor-parallel rank 的模型、采样器、KV cache 和可选 CUDA Graph。
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # 每个 rank 绑定一张 GPU，并通过 NCCL 组成张量并行进程组。
        dist.init_process_group(
            "nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank
        )
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        # warmup_model 会触发一次压力 prefill,估算KV cache可用显存
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                # rank 0 用共享内存广播方法名和参数，其他 rank 收到 event 后同步执行。
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        # 释放共享内存、CUDA Graph 和分布式进程组等运行时资源。
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        # 非 rank 0 worker 持续等待共享内存里的方法调用并同步执行。
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        # 从 rank 0 写入的共享内存中读取方法名和序列化参数。
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        # rank 0 将要广播的方法调用写入共享内存，并唤醒其他 rank。
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        # 在 rank 0 触发跨进程同步调用，然后在当前 rank 执行同名方法。
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        # 用最大规格附近的 fake batch 预热模型，并记录后续估算 KV cache 可用显存所需的峰值。
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = (
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        # warmup_model 已经测出了模型执行一次大规格 prefill 时的显存峰值。
        # 这里先从显存预算中扣除当前占用和推理时的临时显存，再把剩余空间全部
        # 换算成分页 KV cache block。该函数会在每个 tensor-parallel rank 上执行，
        # 因而下面的容量和张量形状都是“单张 GPU / 单个 rank”的数据。
        config = self.config
        hf_config = config.hf_config

        # mem_get_info 返回当前 GPU 的物理空闲显存和总显存。used 不只包含模型权重，
        # 也可能包含 CUDA context、通信库以及当前 GPU 上的其他显存占用。
        free, total = torch.cuda.mem_get_info()
        used = total - free

        # peak/current 是 PyTorch allocator 自 warmup 前 reset_peak_memory_stats() 以来的
        # 峰值/当前已分配显存。因此 peak - current 近似表示一次真实模型前向额外需要的
        # 临时显存，例如 attention、MLP、logits 等中间张量所占的空间。
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        # Tensor parallel 会把 KV heads 切到各个 rank；每张 GPU 只需要保存自己的部分。
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        )

        # 一个“物理 block”包含该 block 内所有 token、所有模型层的 K 和 V：
        #   [2(K/V), num_layers, block_size, num_kv_heads_per_rank, head_dim]
        # 所以单个 block 的字节数是以上各维度乘积，再乘每个元素的字节数。
        block_bytes = (
            2
            * hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * hf_config.dtype.itemsize
        )

        # KV cache 可使用的显存为：
        #   总显存预算 - 当前实际占用 - warmup 测出的额外临时显存
        # = total * utilization - used - (peak - current)
        # 最后除以单个 block 的大小并向下取整，避免实际分配超过预算。
        config.num_kvcache_blocks = (
            int(total * config.gpu_memory_utilization - used - peak + current)
            // block_bytes
        )
        # 小于等于 0 表示仅模型权重和运行时临时显存就已经超过配置的显存预算。
        assert config.num_kvcache_blocks > 0

        # 整块连续分配 KV cache，形状为：
        #   [K/V, layer, physical_block, token_in_block, kv_head, head_dim]
        # __init__ 中已将默认 device/dtype 设为 cuda 和模型 dtype，因此这里虽然没有
        # 显式传入 device/dtype，得到的仍是当前 rank GPU 上、模型精度的张量。
        self.kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
        )

        # 每个 Attention 层只持有总 cache 中属于自己的视图：
        #   k_cache/v_cache.shape =
        #   [num_blocks, block_size, num_kv_heads_per_rank, head_dim]
        # 这些切片与 self.kv_cache 共享底层存储，不会再次申请显存。
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        # 将变长的 block_table padding 成张量，供 flash-attn 的分页 KV cache 接口读取。
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        # 将 scheduler 选中的多条变长序列整理成一次 prefill 前向需要的扁平张量。
        # 本函数不执行模型，只负责准备三类数据：
        # 1. input_ids / positions：本轮真正需要计算的新 token；
        # 2. cu_seqlens_* / max_seqlen_*：FlashAttention 划分变长序列的元数据；
        # 3. slot_mapping / block_tables：新 K/V 的写入位置及已缓存前缀的读取位置。
        #
        # 多条序列不会 padding 成 [batch, max_len]，而是首尾拼成一维 token 流。
        # cu_seqlens_q/k 保存每条序列在拼接张量中的边界，避免 padding 浪费计算。
        input_ids = []
        positions = []

        # cumulative sequence lengths，必须以 0 开头。例如 Q 长度为 [3, 2] 时，
        # cu_seqlens_q = [0, 3, 5]，表示两条序列分别位于 [0:3] 和 [3:5]。
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0

        # slot_mapping 与拼接后的新 token 一一对应，元素值是 KV cache 展平后的
        # 物理 token 槽位：physical_block_id * block_size + offset_in_block。
        slot_mapping = []

        # 没有已缓存前缀时，attention 可以直接使用本轮计算出的连续 K/V；只有存在
        # prefix cache 或 chunked prefill 的历史 K/V 时，才需要分页 block table。
        block_tables = None

        for seq in seqs:
            # start 之前的 token 已经写入或命中 KV cache，本轮不重复送入模型。
            # scheduler 用 num_scheduled_tokens 控制本轮处理多少新 token，因此同一条
            # 长 prompt 可以被拆成多轮 chunked prefill。
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q

            # Q 只包含本轮新 token，长度为 seqlen_q；这些 Q 做注意力时能够看到的
            # K/V 包含 [0:end) 的完整上下文，所以逻辑 K 长度是 end。
            # 当 start > 0 时，前 start 个 K/V 从分页 KV cache 中读取。
            seqlen_k = end

            # input_ids 按序列顺序拼接；positions 使用原序列中的绝对位置，供 RoPE 使用。
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))

            # 记录当前序列结束边界。Q/K 的边界分别基于“本轮新 token 数”和
            # “包含缓存前缀的完整上下文长度”累加，因此两者可能不同。
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

            # FlashAttention 需要 batch 内最大 Q/K 长度来选择 kernel 配置和工作区大小。
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            # warmup_model 构造的假序列没有分配 KV block；它只测量模型前向峰值，
            # Attention 会直接使用本轮产生的 K/V，所以无需生成 cache 写入位置。
            if not seq.block_table:
                continue

            # 找出本轮 [start, end) 跨越的逻辑 block 范围。end_block 使用向上取整，
            # 并作为开区间终点，例如 block_size=4、[3, 7) 会涉及 block 0 和 1。
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size

            # seq.block_table[i] 给出逻辑 block i 对应的物理 block id。下面逐 block
            # 展开成本轮每个新 token 的物理槽位，顺序必须与 input_ids 完全一致。
            for i in range(start_block, end_block):
                # 当前物理 block 展平后的起始槽位。
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    # 第一块可能已有缓存 token，只从 start 在块内的偏移处开始写。
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    # 中间完整 block 一直写到该物理 block 的末尾。
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    # 最后一块通常只写一部分；end - i * block_size 是块内结束偏移。
                    slot_end = (
                        seq.block_table[i] * self.block_size + end - i * self.block_size
                    )
                slot_mapping.extend(range(slot_start, slot_end))

        # sum(K lengths) - sum(Q lengths) 等于所有序列已缓存前缀长度之和。
        # 只要大于 0，Attention 就必须通过 block_tables 从分页 KV cache 读取旧 K/V；
        # 否则 K/V 全由本轮前向产生，直接使用连续张量更简单。
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)

        # 先在 pinned CPU memory 中构造张量，再异步复制到当前 GPU。pinned memory 配合
        # non_blocking=True 才能真正进行异步 H2D copy；int32 是 FlashAttention 元数据
        # 接口要求的类型，token id 和 position 使用 int64。
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        # 模型各层的 Attention 和 LM head 会从全局 Context 读取这些 batch 元数据：
        # Attention 用它完成变长因果注意力及 KV cache 读写；LM head 使用
        # cu_seqlens_q 只选取每条序列本轮最后一个 Q 的 hidden state。若 chunked
        # prefill 尚未完成，虽然会得到 logits，但 scheduler 不会追加采样结果。
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
        )

        # input_ids/positions 作为模型 forward 的显式输入，其余数据通过 Context 传递。
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        # 将 scheduler 选中的多条运行中序列整理成一次 decode 前向需要的张量。
        # Decode 与 prefill 的主要区别是：每条序列本轮都只计算一个 token，因此
        # input_ids 的长度就是 decode batch size，不需要 cu_seqlens 等变长元数据。
        #
        # 进入这里时，seq.last_token 是上一轮刚采样并追加到 Sequence 的 token，
        # 但它自己的 K/V 还没有写入 cache。本轮将该 token 输入模型，先把新 K/V
        # 写入 slot_mapping 指定的位置，再让它关注包含自身在内的完整历史上下文，
        # 最终产生“下一个 token”的 logits。
        input_ids = []
        positions = []

        # slot_mapping[i]：第 i 条序列当前 token 的 K/V 应写入哪个物理 cache 槽位。
        slot_mapping = []

        # context_lens[i]：第 i 条序列当前完整上下文长度，包含本轮输入的 last_token。
        # FlashAttention 据此只读取该序列已经有效的 KV cache 范围。
        context_lens = []

        for seq in seqs:
            # 每条序列只输入最近采样出的一个 token，所以 input_ids.shape 最终为 [bs]。
            input_ids.append(seq.last_token)

            # token 的位置从 0 开始；len(seq) 已包含 last_token，因此绝对位置是 len-1。
            # 该位置会用于 RoPE，不能简单地对所有序列都设置为 0。
            positions.append(len(seq) - 1)

            # 当前 token 写入 cache 后，有效上下文就是 Sequence 的全部 token。
            context_lens.append(len(seq))

            # block_table[-1] 是当前 token 所在逻辑 block 对应的物理 block id；
            # last_block_num_tokens 是最后一块已有的 token 数，减 1 后得到当前 token
            # 在块内的零基偏移。因此物理槽位为：
            #   physical_block_id * block_size + offset_in_block
            # scheduler.may_append() 已在当前 token 开启新 block 时提前完成分配。
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )

        # 先在 pinned CPU memory 中构造，再异步复制到当前 GPU。token id 和 position
        # 使用 int64；FlashAttention / KV cache 元数据使用 int32。
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        # Decode 的历史 K/V 全部存放在分页 KV cache 中，所以始终需要 block_tables
        # 告诉 FlashAttention 每条序列的逻辑 block 映射到了哪些物理 block。
        # prepare_block_tables 会把不同长度的映射表用 -1 padding 成二维 int32 张量。
        block_tables = self.prepare_block_tables(seqs)

        # is_prefill=False 让 Attention 走 flash_attn_with_kvcache 路径：
        # 1. 根据 slot_mapping 写入当前 token 的 K/V；
        # 2. 根据 context_lens 和 block_tables 读取完整上下文并计算当前 Query 的输出。
        # input_ids/positions 显式传给模型，其余元数据通过全局 Context 供各层读取。
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        # 提取每条序列的采样温度并搬到 GPU 上。
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=True
        ).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(
        self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool
    ):
        # 根据阶段和 batch size 选择 eager 前向或 CUDA Graph replay，并返回 logits。
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, : context.block_tables.size(1)] = (
                context.block_tables
            )
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        # 执行 scheduler 已选中 batch 的一次完整推理：
        #   Sequence 状态 -> GPU 输入/Attention 元数据 -> 模型 logits -> 采样 token。
        # 本函数不会修改 Sequence 的 token/cache 计数；rank 0 返回 token_ids 后，
        # Scheduler.postprocess 才负责更新 num_cached_tokens、追加 token 和结束请求。
        #
        # Tensor parallel 时，rank 0 通过 call("run", ...) 将同一批 seqs 广播给 worker。
        # 所有 rank 都会进入本函数并计算各自的权重分片，且必须以相同顺序参与
        # all-reduce/gather 等 collective；只有 rank 0 负责最终采样并返回结果。

        # Prefill 和 decode 需要的输入形式不同：
        # - prefill：拼接本轮所有新 token，并准备 cu_seqlens、slot_mapping 等元数据；
        # - decode：每条序列只输入 last_token，并准备 context_lens、block_tables。
        # 两个 prepare_* 都会把 Attention 所需数据写入当前进程的全局 Context，
        # 而 input_ids/positions 作为 self.model 的显式参数返回。
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        )

        # 每条序列的 temperature 只在采样阶段使用。多卡时 ParallelLMHead 会把
        # 各 rank 的局部词表 logits gather 到 rank 0，所以 worker 无需准备温度张量。
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

        # 执行模型 backbone 和 LM head：prefill 通常走 eager；decode 在允许时 replay
        # CUDA Graph。Tensor parallel 的跨卡通信发生在模型层和 ParallelLMHead 内部。
        # 多卡情况下只有 rank 0 得到拼接后的完整词表 logits，worker 的 logits 为 None。
        logits = self.run_model(input_ids, positions, is_prefill)

        # rank 0 根据 temperature 对每条序列的 logits 采样一个 token。tolist() 会把
        # GPU 结果取回 CPU，并形成与 seqs 顺序一一对应的 Python token id 列表。
        # worker 不采样，返回 None；其工作只是在模型前向中贡献自己的张量并行分片。
        token_ids = (
            self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        )

        # Context 是模块级临时状态，必须在本轮前向结束后清空，避免下一轮或不同阶段
        # 的 Attention 误用旧的 slot_mapping、cu_seqlens、block_tables 等元数据。
        reset_context()

        # rank 0 的结果随后交给 Scheduler.postprocess；worker 返回值不会被主进程使用。
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        # Decode 每步只处理每条序列的一个 token，模型由大量短 CUDA kernel 组成，
        # CPU 逐个 launch kernel 的开销会比较明显。CUDA Graph 把一次固定形状的
        # 模型前向记录为 GPU 操作图，后续同形状 batch 可用 graph.replay() 一次提交。
        #
        # Graph 要求捕获期间使用的输入/输出地址和张量形状保持不变。因此这里先分配
        # 最大 batch 的静态缓冲区；运行时只覆盖其中的前 bs 项，不能把新建张量直接
        # 传给 replay。
        config = self.config
        hf_config = config.hf_config

        # 本实现最多为 512 条 decode 序列捕图。更大的 batch 在 run_model 中走 eager，
        # 避免为很少使用的大 shape 保存更多 CUDA Graph 及其专属显存。
        # decode的batch size就是seq的数量，因为每条seq只生成一个token
        max_bs = min(self.config.max_num_seqs, 512)

        # block_tables 的列数必须固定；一条长度为 max_model_len 的序列最多需要
        # ceil(max_model_len / block_size) 个逻辑 KV block。
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        # 以下均为会被 graph 捕获、且运行时反复覆盖内容的静态 GPU 缓冲区：
        # - input_ids / positions：本轮每条序列要 decode 的 token 及其位置；
        # - slot_mapping：新 K/V 应写入分页 KV cache 的物理槽位；
        # - context_lens / block_tables：attention 从 KV cache 读取上下文所需的元数据；
        # - outputs：模型 backbone 的 hidden states，由图写入、图外计算 logits。
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        # 预先覆盖常见 decode batch：小 batch 用 1/2/4/8，其他 batch 向上取整到
        # 16 的倍数。例如真实 bs=19 会 replay bs=32 的图，只有前 19 行是有效请求。
        # 这在图数量、显存占用和形状匹配率之间做折中。
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # 从最大 batch 往小捕获：先让共享 pool 容纳最大的中间张量，后续较小图通常
        # 能复用它的显存。捕获的图本身仍分别保存在 self.graphs[bs] 中。
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()

            # Attention 不通过 self.model 的参数接收这些元数据，而是从全局 Context
            # 读取。捕获时把 Context 绑定到静态缓冲区的 bs 行切片，确保 replay 时
            # attention 继续读取同一地址、但内容已被 run_model 更新。
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )

            # 先普通执行一次，使 Triton/FlashAttention 的懒初始化、内存分配和 autotune
            # 在 capture 之外完成；否则这些动态操作可能无法被安全地记录进 CUDA Graph。
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                # 只记录 backbone 前向，不记录 lm_head 和 sampler。运行时 replay 后，
                # run_model 会从 outputs[:实际 bs] 计算 logits 并交给 sampler。
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture

            # 第一个图创建 graph memory pool；其余图共用该 pool，以复用捕获期间的
            # 中间张量显存。图的这部分内存必须一直保留到 ModelRunner.exit() 才能释放。
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph

            # 确保当前图的捕获和 warmup 完成后再重置 Context、继续捕获下一个形状。
            torch.cuda.synchronize()
            reset_context()

        # run_model 的 replay 路径通过这个字典取得静态缓冲区并写入真实 batch 数据。
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
