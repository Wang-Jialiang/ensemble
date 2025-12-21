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


def get_models_from_source(
    trainer_or_models: Any,
) -> Tuple[List[nn.Module], torch.device]:
    """
    从 Trainer 或模型列表中提取模型和设备

    Args:
        trainer_or_models: Trainer 实例或 List[nn.Module]

    Returns:
        (models, device): 模型列表和计算设备
    """
    # 优先使用统一的 get_models() 接口
    if hasattr(trainer_or_models, "get_models"):
        models = trainer_or_models.get_models()
        # 从第一个模型获取设备
        if models:
            device = next(models[0].parameters()).device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 后备: 直接访问 workers 属性 (向后兼容)
    elif hasattr(trainer_or_models, "workers"):
        models = [
            model for worker in trainer_or_models.workers for model in worker.models
        ]
        device = trainer_or_models.workers[0].device
    else:  # 是模型列表
        models = trainer_or_models
        try:
            device = next(models[0].parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return models, device


def get_all_models_logits(
    models: List[nn.Module], loader: DataLoader, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    获取所有模型在数据集上的 logits

    Args:
        models: 模型列表 List[nn.Module]
        loader: 数据加载器
        device: 计算设备

    Returns:
        all_logits: (num_models, num_samples, num_classes)
        targets: (num_samples,)
    """
    from tqdm import tqdm

    all_logits_list = []
    all_targets_list = []

    iterator = tqdm(loader, desc="Evaluating Models", leave=False)

    with torch.no_grad():
        for inputs, targets in iterator:
            inputs = inputs.to(device)
            batch_logits = []

            for model in models:
                model.eval()
                logits = model(inputs)  # (batch_size, num_classes)
                batch_logits.append(logits.unsqueeze(0).cpu())

            # combined: (num_models, batch_size, num_classes)
            if batch_logits:
                combined = torch.cat(batch_logits, dim=0)
                all_logits_list.append(combined)
                all_targets_list.append(targets.cpu())

    if not all_logits_list:
        return torch.tensor([]), torch.tensor([])

    # 沿着 batch 维度 (dim=1) 拼接
    all_logits = torch.cat(all_logits_list, dim=1)
    all_targets = torch.cat(all_targets_list)

    return all_logits, all_targets
