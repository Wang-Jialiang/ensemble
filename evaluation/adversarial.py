"""
================================================================================
对抗鲁棒性评估模块
================================================================================

包含: FGSM 攻击、PGD 攻击、对抗鲁棒性评估
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..utils import get_logger
from .inference import get_models_from_source

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 对抗攻击方法                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def _get_norm_params(eps, alpha, mean, std):
    """计算标准化空间下的扰动边界与裁剪范围"""
    return (eps / std, alpha / std if alpha else None, (0 - mean) / std, (1 - mean) / std)

def fgsm_attack(model, images, labels, eps, mean, std) -> torch.Tensor:
    """FGSM 攻击: std 空间变换 -> 符号梯度 -> 裁剪"""
    e_n, _, lower, upper = _get_norm_params(eps, None, mean, std)
    images = images.clone().detach().requires_grad_(True)
    
    loss = F.cross_entropy(model(images), labels)
    loss.backward()
    
    adv = images + e_n * images.grad.sign()
    return torch.max(torch.min(adv, upper), lower).detach()

def pgd_attack(model, images, labels, eps, alpha, steps, mean, std) -> torch.Tensor:
    """PGD 攻击: 随机初始化 -> 迭代更新 -> 投影 -> 裁剪"""
    e_n, a_n, lower, upper = _get_norm_params(eps, alpha, mean, std)
    adv = (images + torch.empty_like(images).uniform_(-1, 1) * e_n).clamp(lower, upper)

    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), labels)
        model.zero_grad(); loss.backward()
        
        # 迭代更新与投影
        adv = images + (adv + a_n * adv.grad.sign() - images).clamp(-e_n, e_n)
        adv = adv.clamp(lower, upper).detach()
    return adv


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 对抗鲁棒性评估                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def evaluate_adversarial(trainer_or_models, loader, eps=8/255, alpha=2/255, steps=10, dataset="cifar10", logger=None) -> Dict:
    """集成对抗鲁棒性评估 (大纲化)"""
    from tqdm import tqdm
    log = logger or get_logger()
    log.info(f"\n🗡️ Adversarial Eval (ε={eps*255:.1f}/255, Steps={steps})")

    models, device = get_models_from_source(trainer_or_models)
    mean, std = _get_dataset_norm(dataset, device)
    
    # 建立集成攻击外壳
    ens_model = _EnsembleProxy(models).to(device).eval()
    stats = {"total": 0, "clean": 0, "fgsm": 0, "pgd": 0}

    pbar = tqdm(loader, desc="Adversarial", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        stats["total"] += x.size(0)
        
        # 1. 干净精度
        with torch.no_grad(): stats["clean"] += (ens_model(x).argmax(1) == y).sum().item()
        
        # 2. 对抗攻击 (FGSM/PGD)
        stats["fgsm"] += _run_and_eval_attack(ens_model, fgsm_attack, x, y, eps, mean, std)
        stats["pgd"] += _run_and_eval_attack(ens_model, pgd_attack, x, y, eps, alpha, steps, mean, std)
        
        pbar.set_postfix({k: f"{100*v/stats['total']:.1f}%" for k, v in stats.items() if k != "total"})

    return _summarize_adv_results(stats, eps, alpha, steps, log)

class _EnsembleProxy(nn.Module):
    def __init__(self, models): super().__init__(); self.models = nn.ModuleList(models)
    def forward(self, x): return torch.stack([m(x) for m in self.models]).mean(0)

def _get_dataset_norm(name, device):
    from ..datasets import DATASET_REGISTRY
    cls = DATASET_REGISTRY.get(name.lower())
    m = cls.MEAN if cls else [0.485, 0.456, 0.406]
    s = cls.STD if cls else [0.229, 0.224, 0.225]
    return torch.tensor(m).view(1,3,1,1).to(device), torch.tensor(s).view(1,3,1,1).to(device)

def _run_and_eval_attack(model, attack_fn, x, y, *args):
    """封装 攻击 -> 推理 -> 计数 逻辑"""
    prev_training = model.training
    model.train() # 确保允许梯度计算
    for m in model.models: m.eval() # BN 维持 eval
    
    adv_x = attack_fn(model, x, y, *args)
    
    model.train(prev_training) 
    with torch.no_grad(): return (model(adv_x).argmax(1) == y).sum().item()

def _summarize_adv_results(s, eps, alpha, steps, log):
    t = s["total"]
    res = { "clean_acc": 100*s["clean"]/t, "fgsm_acc": 100*s["fgsm"]/t, "pgd_acc": 100*s["pgd"]/t,
            "eps": eps, "eps_255": eps*255, "alpha": alpha, "pgd_steps": steps, "num_samples": t }
    log.info(f"   ✅ Clean: {res['clean_acc']:.2f}% | FGSM: {res['fgsm_acc']:.2f}% | PGD-{steps}: {res['pgd_acc']:.2f}%")
    return res
