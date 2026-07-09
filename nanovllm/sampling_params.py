from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        # 当前采样器使用温度采样，不支持把 temperature 设为近似 0 的贪心模式。
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
