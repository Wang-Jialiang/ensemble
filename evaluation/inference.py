"""
================================================================================
模型推理模块
================================================================================

包含: get_models_from_source, get_all_models_logits
"""

from typing import Any, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 模型提取与推理                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def get_models_from_source(source: Any) -> Tuple[List[nn.Module], torch.device]:
    """多态探测: 从 Trainer, Worker 或 List[Module] 中提取资源"""
    # 1. 解析模型列表
    if hasattr(source, "get_models"):
        models = source.get_models()
    elif hasattr(source, "workers"):
        models = [m for w in source.workers for m in w.models]
    else:
        models = source # 假定为 List[nn.Module]
    
    # 2. 解析计算设备
    def _probe_device(objs):
        for o in objs: 
            if hasattr(o, "device"): return o.device
            p = next(o.parameters(), None)
            if p is not None: return p.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return models, _probe_device(source.workers if hasattr(source, "workers") else models)


def get_all_models_logits(models: List[nn.Module], loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """全量数据集推理门面 (大纲化)"""
    from tqdm import tqdm
    all_l, all_t = [], []
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Inference", leave=False):
            # 将 [N_models, Batch, D] 暂存
            batch_l = _infer_models_on_batch(models, x.to(device))
            all_l.append(batch_l)
            all_t.append(y)

    if not all_l: return torch.tensor([]), torch.tensor([])
    return torch.cat(all_l, dim=1), torch.cat(all_t)

def _infer_models_on_batch(models, x):
    """单批次多模型推理核心"""
    batch_res = []
    for m in models:
        m.eval()
        batch_res.append(m(x).unsqueeze(0).cpu())
    return torch.cat(batch_res, dim=0) # [num_models, batch_size, num_classes]
