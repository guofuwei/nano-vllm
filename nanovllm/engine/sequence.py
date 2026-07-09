from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    # 等待调度，通常还没有完成 prefill，或被抢占后回到等待队列。
    WAITING = auto()
    # 已完成 prefill，正在 decode 阶段逐 token 生成。
    RUNNING = auto()
    # 已达到 EOS 或 max_tokens，生成结束。
    FINISHED = auto()


class Sequence:
    # KV cache 的逻辑分块大小，由 LLMEngine 根据 Config.kvcache_block_size 统一覆盖。
    block_size = 256
    # 全局自增计数器，用来给每条请求分配稳定的 seq_id。
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        # 当前序列的唯一编号，用于恢复输出顺序和区分不同请求。
        self.seq_id = next(Sequence.counter)
        # 当前调度状态：等待、运行中或已结束。
        self.status = SequenceStatus.WAITING
        # 完整 token 序列，初始为 prompt tokens，后续会不断追加生成 token。
        self.token_ids = copy(token_ids)
        # 最近一个 token；decode 阶段每次只需要把这个 token 输入模型。
        self.last_token = token_ids[-1]
        # 当前总 token 数，等于 prompt tokens + 已生成 tokens。
        self.num_tokens = len(self.token_ids)
        # prompt 的 token 数，用于从完整 token 序列中切分出 completion。
        self.num_prompt_tokens = len(token_ids)
        # 已经写入或命中 KV cache 的 token 数。
        self.num_cached_tokens = 0
        # 本轮 scheduler 安排给模型执行的 token 数。
        self.num_scheduled_tokens = 0
        # 是否处于 prefill 语义；被抢占后会重新置为 True。
        self.is_prefill = True
        # 逻辑 block 到物理 KV cache block id 的映射表。
        self.block_table = []
        # 采样温度，值越大随机性越强。
        self.temperature = sampling_params.temperature
        # 最多生成多少个 completion token。
        self.max_tokens = sampling_params.max_tokens
        # 是否忽略 EOS；benchmark 中常用于强制生成到 max_tokens。
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        # 让 len(seq) 直接返回当前 token 总数。
        return self.num_tokens

    def __getitem__(self, key):
        # 允许像列表一样用 seq[i] 或 seq[start:end] 读取 token。
        return self.token_ids[key]

    @property
    def is_finished(self):
        # 便捷判断当前序列是否已经完成。
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        # 已生成 token 数，不包含 prompt。
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        # 原始 prompt token 切片。
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        # 生成结果 token 切片。
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        # 当前序列按 block_size 切分后需要的逻辑 block 数，向上取整。
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        # 最后一个逻辑 block 中已有多少 token，用于计算新 token 写入 KV cache 的位置。
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        # 返回第 i 个逻辑 block 对应的 token 切片。
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        # 将新采样出的 token 追加到序列末尾，并同步更新 last_token 和计数。
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        # 多进程传输时压缩序列状态：decode 阶段只传 last_token，避免重复拷贝完整历史。
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        # 与 __getstate__ 配套，还原发送到 worker rank 的轻量序列状态。
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
