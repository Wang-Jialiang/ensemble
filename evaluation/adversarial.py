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
from .core import extract_models

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 对抗攻击方法                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """FGSM (Fast Gradient Sign Method) 对抗攻击

    单步攻击，沿损失梯度符号方向添加扰动。

    Args:
        model: 目标模型
        images: 输入图像 (已标准化)
        labels: 真实标签
        eps: 扰动强度 ε (在原始像素空间, 如 8/255)
        mean: 标准化均值
        std: 标准化标准差

    Returns:
        对抗样本 (已标准化)
    """
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    loss.backward()

    # 在原始像素空间计算扰动，然后转换回标准化空间
    eps_normalized = eps / std

    perturbation = eps_normalized * images.grad.sign()
    adv_images = images + perturbation

    # 裁剪到有效范围
    lower_bound = (0 - mean) / std
    upper_bound = (1 - mean) / std
    adv_images = torch.max(torch.min(adv_images, upper_bound), lower_bound)

    return adv_images.detach()


def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """PGD (Projected Gradient Descent) 对抗攻击

    多步迭代攻击，是 FGSM 的增强版。

    Args:
        model: 目标模型
        images: 输入图像 (已标准化)
        labels: 真实标签
        eps: 最大扰动强度 ε (在原始像素空间)
        alpha: 每步扰动大小 α (在原始像素空间)
        steps: 迭代步数
        mean: 标准化均值
        std: 标准化标准差

    Returns:
        对抗样本 (已标准化)
    """
    # 转换到标准化空间
    eps_normalized = eps / std
    alpha_normalized = alpha / std

    # 有效范围
    lower_bound = (0 - mean) / std
    upper_bound = (1 - mean) / std

    # 随机初始化扰动
    adv_images = images.clone().detach()
    random_noise = torch.empty_like(adv_images).uniform_(-1, 1) * eps_normalized
    adv_images = adv_images + random_noise
    adv_images = torch.max(torch.min(adv_images, upper_bound), lower_bound)

    for _ in range(steps):
        adv_images.requires_grad_(True)
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)

        model.zero_grad()
        loss.backward()

        # 沿梯度方向更新
        grad_sign = adv_images.grad.sign()
        adv_images = adv_images.detach() + alpha_normalized * grad_sign

        # 投影到 ε-球内
        delta = adv_images - images
        delta = torch.max(torch.min(delta, eps_normalized), -eps_normalized)
        adv_images = images + delta

        # 裁剪到有效范围
        adv_images = torch.max(torch.min(adv_images, upper_bound), lower_bound)

    return adv_images.detach()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 对抗鲁棒性评估                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def evaluate_adversarial(
    trainer_or_models: Any,
    test_loader: DataLoader,
    eps: float = 8 / 255,
    alpha: float = 2 / 255,
    pgd_steps: int = 10,
    dataset_name: str = "cifar10",
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """对抗鲁棒性评估 (FGSM/PGD 实时攻击)

    使用 FGSM 和 PGD 攻击评估集成模型的对抗鲁棒性。
    攻击针对集成模型的平均 logits 进行。

    Args:
        trainer_or_models: StagedEnsembleTrainer 实例或 List[nn.Module]
        test_loader: 测试数据加载器
        eps: 扰动强度 ε (默认 8/255 ≈ 0.031)
        alpha: PGD 步长 α (默认 2/255 ≈ 0.008)
        pgd_steps: PGD 迭代步数 (默认 10)
        dataset_name: 数据集名称 (用于获取标准化参数)
        logger: 日志记录器

    Returns:
        包含对抗鲁棒性指标的字典
    """
    from tqdm import tqdm

    from ..datasets import DATASET_REGISTRY

    logger = logger or get_logger()
    logger.info("\n🗡️ Running Adversarial Robustness Evaluation")
    logger.info(f"   ε = {eps:.4f} ({eps * 255:.1f}/255)")
    logger.info(f"   PGD: α = {alpha:.4f}, steps = {pgd_steps}")

    models, device = extract_models(trainer_or_models)

    # 获取数据集的标准化参数
    if dataset_name.lower() in DATASET_REGISTRY:
        DatasetClass = DATASET_REGISTRY[dataset_name.lower()]
        mean = torch.tensor(DatasetClass.MEAN).view(1, 3, 1, 1).to(device)
        std = torch.tensor(DatasetClass.STD).view(1, 3, 1, 1).to(device)
    else:
        # 默认使用 ImageNet 标准化参数
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    # 创建一个包装模型，输出集成的平均 logits
    class EnsembleWrapper(nn.Module):
        def __init__(self, models_list):
            super().__init__()
            self.models = nn.ModuleList(models_list)

        def forward(self, x):
            logits_list = [m(x) for m in self.models]
            return torch.stack(logits_list).mean(dim=0)

    ensemble_model = EnsembleWrapper(models).to(device)
    ensemble_model.eval()

    # 统计变量
    clean_correct = 0
    fgsm_correct = 0
    pgd_correct = 0
    total = 0

    pbar = tqdm(test_loader, desc="Adversarial Eval", leave=False)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)
        total += batch_size

        # 干净样本预测
        with torch.no_grad():
            clean_outputs = ensemble_model(images)
            clean_preds = clean_outputs.argmax(dim=1)
            clean_correct += (clean_preds == labels).sum().item()

        # FGSM 攻击
        ensemble_model.train()  # 需要梯度
        for m in ensemble_model.models:
            m.eval()  # 但 BN 保持 eval 模式

        fgsm_images = fgsm_attack(ensemble_model, images, labels, eps, mean, std)

        with torch.no_grad():
            fgsm_outputs = ensemble_model(fgsm_images)
            fgsm_preds = fgsm_outputs.argmax(dim=1)
            fgsm_correct += (fgsm_preds == labels).sum().item()

        # PGD 攻击
        pgd_images = pgd_attack(
            ensemble_model, images, labels, eps, alpha, pgd_steps, mean, std
        )

        with torch.no_grad():
            pgd_outputs = ensemble_model(pgd_images)
            pgd_preds = pgd_outputs.argmax(dim=1)
            pgd_correct += (pgd_preds == labels).sum().item()

        # 更新进度条
        pbar.set_postfix(
            {
                "clean": f"{100 * clean_correct / total:.1f}%",
                "fgsm": f"{100 * fgsm_correct / total:.1f}%",
                "pgd": f"{100 * pgd_correct / total:.1f}%",
            }
        )

    # 恢复 eval 模式
    ensemble_model.eval()

    # 计算指标
    clean_acc = 100.0 * clean_correct / total
    fgsm_acc = 100.0 * fgsm_correct / total
    pgd_acc = 100.0 * pgd_correct / total

    results = {
        "clean_acc": clean_acc,
        "fgsm_acc": fgsm_acc,
        "pgd_acc": pgd_acc,
        "fgsm_attack_success_rate": 100.0 - fgsm_acc,
        "pgd_attack_success_rate": 100.0 - pgd_acc,
        "fgsm_robustness_drop": clean_acc - fgsm_acc,
        "pgd_robustness_drop": clean_acc - pgd_acc,
        "eps": eps,
        "eps_255": eps * 255,
        "alpha": alpha,
        "pgd_steps": pgd_steps,
        "num_samples": total,
    }

    logger.info("   ✅ Adversarial Robustness Results:")
    logger.info(f"      Clean Accuracy: {clean_acc:.2f}%")
    logger.info(f"      FGSM Accuracy (ε={eps * 255:.0f}/255): {fgsm_acc:.2f}%")
    logger.info(
        f"      PGD-{pgd_steps} Accuracy (ε={eps * 255:.0f}/255): {pgd_acc:.2f}%"
    )
    logger.info(f"      FGSM Robustness Drop: {clean_acc - fgsm_acc:.2f}%")
    logger.info(f"      PGD Robustness Drop: {clean_acc - pgd_acc:.2f}%")

    return results
