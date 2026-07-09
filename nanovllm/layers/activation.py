import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 将合并的 gate/up 张量一分为二，并执行 SwiGLU 的 silu(gate) * up。
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
