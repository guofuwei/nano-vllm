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
        # 为 prefill batch 拼接输入 token、位置编码和写 cache 的 slot 映射。
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            # 只把尚未缓存的 token 放进本轮 prefill；已命中的 prefix 通过 block_table 参与注意力。
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:  # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            # slot_mapping 描述每个新 token 应写入 KV cache 的物理位置。
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = (
                        seq.block_table[i] * self.block_size + end - i * self.block_size
                    )
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # prefix cache
            block_tables = self.prepare_block_tables(seqs)
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
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        # 为 decode batch 准备每条序列的最新 token、当前位置和 KV cache 访问元数据。
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
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
        block_tables = self.prepare_block_tables(seqs)
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
        # prepare_* 会把调度结果转换成模型需要的张量，并通过全局 context 传给 Attention。
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        )
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = (
            self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        )
        reset_context()
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
