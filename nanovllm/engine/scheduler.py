from collections import deque

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus


class Scheduler:
    def __init__(self, config: Config):
        # 单次调度最多允许进入 batch 的序列数，限制并发 prefill/decode 的请求数量。
        self.max_num_seqs = config.max_num_seqs
        # 单次调度最多处理的 token 总数，主要用于控制 prefill batch 的计算量和显存占用。
        self.max_num_batched_tokens = config.max_num_batched_tokens
        # 结束 token id；postprocess 中用它判断一条序列是否生成结束。
        self.eos = config.eos
        # KV cache block 大小，用于计算 prompt 命中的 prefix block 和剩余 token 数。
        self.block_size = config.kvcache_block_size
        # KV cache 物理 block 管理器，负责分配、释放、复用和 prefix cache 命中。
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size
        )
        # 等待队列：新请求、尚未完成 prefill 的请求，以及被抢占后需要重新 prefill 的请求。
        self.waiting: deque[Sequence] = deque()
        # 运行队列：已完成 prefill、正在 decode 阶段逐 token 生成的请求。
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        # waiting 和 running 队列都为空时，说明所有请求已经结束。
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        # 新请求先进入 waiting 队列，等待 prefill 调度。
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # 本轮真正会交给 ModelRunner 执行的序列列表。
        scheduled_seqs = []
        # prefill 阶段累计的 token 数，用来和 max_num_batched_tokens 做预算比较。
        num_batched_tokens = 0

        # prefill 阶段负责把 prompt 写入 KV cache；这里优先利用 prefix cache，必要时做 chunked prefill。
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            # waiting 队列按 FIFO 处理；这里只查看队首，只有完整进入 running 后才 popleft。
            seq = self.waiting[0]
            # 当前 batch 还剩多少 token 预算。
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                # 新进入 prefill 的序列还没有分配 KV block；先尝试命中 prefix cache 并检查容量。-1表示无法分配，>=0表示命中 prefix block 的数量。
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    # 连这条队首序列都无法分配 KV cache，本轮不能继续 prefill。
                    break
                # 已命中的 prefix block 不需要重复计算，只调度未缓存的 token。
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # 这条序列之前做过 chunked prefill，本轮从上次缓存进度继续。
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            # 为了避免一个 batch 中混入太多不完整 prompt，只允许 batch 的第一条序列被切块。
            if remaining < num_tokens and scheduled_seqs:
                break
            if not seq.block_table:
                # 真正把逻辑 block 映射到物理 KV cache block；命中的 prefix block 会被复用。
                self.block_manager.allocate(seq, num_cached_blocks)
            # 如果剩余预算不够处理完整 prompt，就只调度 remaining 个 token，形成 chunked prefill。
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                # prompt 已经全部 prefill 完，下一步开始进入 decode 阶段。
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            # 即使是未完成的 chunked prefill，也会加入本轮 batch；下轮继续留在 waiting 队首。
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            # 只要本轮安排了 prefill，就优先返回 prefill batch，不与 decode 混跑。
            return scheduled_seqs, True

        # decode 阶段每条 running 序列追加一个 token；KV block 不够时会抢占尾部序列并回到 waiting。
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            # 从 running 队首取出一条序列尝试 decode，稍后会把成功调度的序列放回队列头部。
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                # 如果当前序列追加新 token 需要新 block，但没有空闲 block，就先抢占队尾序列释放 cache。
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    # 如果只剩当前序列也无法追加，就抢占自己，本轮不调度它。
                    self.preempt(seq)
                    break
            else:
                # decode 每轮每条序列只处理一个 token。
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                # 如果这个 token 会开启新的 KV block，这里先分配出来。
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        # 把本轮成功 decode 的序列放回 running 队首，保持轮转顺序。
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        # 抢占会释放这条序列的 KV cache。之后重新 prefill 时可通过 prefix cache 复用已哈希的整块。
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        # 消化模型输出：更新 cache 进度、追加 token，并释放已结束序列的资源。
        for seq, token_id in zip(seqs, token_ids):
            # prefill/decode 完成后，先把新写满的 KV block 记录到 prefix cache 索引中。
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (
                not seq.ignore_eos and token_id == self.eos
            ) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
