"""
================================================================================
OOD (Out-of-Distribution) 检测评估模块
================================================================================
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..utils import get_logger
from .inference import get_all_models_logits, get_models_from_source

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ OOD 检测评估                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def evaluate_ood(trainer_or_models, id_loader, ood_loader, ood_name="OOD", logger=None) -> Dict:
    """OOD 检测评估 (大纲化)"""
    log = logger or get_logger()
    log.info(f"\n🔍 OOD Eval ({ood_name})")

    models, device = get_models_from_source(trainer_or_models)
    
    # 提取 ID 与 OOD 的分数
    def _extract_scores(loader, desc):
        log.info(f"   📊 Inference on {desc}...")
        all_l, _ = get_all_models_logits(models, loader, device)
        probs = F.softmax(all_l.mean(0), dim=1) # [N, C]
        msp = probs.max(1)[0].cpu().numpy()
        ent = -(probs * torch.log(probs + 1e-10)).sum(1).cpu().numpy()
        return msp, ent

    id_msp, id_ent = _extract_scores(id_loader, "ID")
    ood_msp, ood_ent = _extract_scores(ood_loader, ood_name)

    if not len(id_msp) or not len(ood_msp): return {"error": "Empty data"}

    # 计算指标
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([np.ones(len(id_msp)), np.zeros(len(ood_msp))])
    
    auroc_msp = roc_auc_score(y, np.concatenate([id_msp, ood_msp])) * 100
    auroc_ent = roc_auc_score(y, -np.concatenate([id_ent, ood_ent])) * 100
    
    fpr95_msp = _compute_fpr_at_95tpr(np.concatenate([id_msp, ood_msp]), y)
    fpr95_ent = _compute_fpr_at_95tpr(-np.concatenate([id_ent, ood_ent]), y)

    res = { "ood_dataset": ood_name, "ood_auroc_msp": auroc_msp, "ood_auroc_entropy": auroc_ent,
            "ood_fpr95_msp": fpr95_msp, "ood_fpr95_entropy": fpr95_ent }
    
    log.info(f"   ✅ AUROC (MSP): {auroc_msp:.2f}% | FPR95: {fpr95_msp:.2f}%")
    return res

def _compute_fpr_at_95tpr(scores, labels):
    """计算 95% TPR 下的 FPR"""
    id_scores, ood_scores = scores[labels==1], scores[labels==0]
    thresh = np.sort(id_scores)[int(len(id_scores) * 0.05)]
    return (ood_scores >= thresh).mean() * 100 if len(ood_scores) else 0.0
