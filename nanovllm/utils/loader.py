import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    # 默认权重加载逻辑：完整张量直接复制到目标参数。
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    # 遍历 safetensors 权重文件，并按模型层自定义的 weight_loader 完成切片或合并加载。
    # 有些 checkpoint 中分开的权重，在本项目模型里会被合并成一个参数，
    # 例如 q/k/v projection 会被加载到同一个 qkv_proj 参数中。
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        # 先把 checkpoint tensor 读取到 CPU，再由 copy_ 写入模型参数所在设备。
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # 优先处理需要“改名 + 打包”的权重。
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        # 将 checkpoint 中的原始参数名映射到模型里的合并参数名。
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        # 打包参数必须提供自定义 loader，用 shard_id 指明写入哪一段。
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # for...else：只有没有命中任何 packed_modules_mapping 时才会走这里。
                    # 普通参数按同名加载；若参数自带 weight_loader，则由它完成 TP 切片。
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
