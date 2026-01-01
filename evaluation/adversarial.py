"""
================================================================================
对抗鲁棒性评估模块
================================================================================

包含: FGSM 攻击、PGD 攻击、对抗鲁棒性评估
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import get_logger
from .inference import get_models_from_source

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 对抗攻击方法                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def _get_norm_params(eps, alpha, mean, std):
    """计算标准化空间下的扰动边界与裁剪范围"""
    return (
        eps / std,
        alpha / std if alpha else None,
        (0 - mean) / std,
        (1 - mean) / std,
    )


def fgsm_attack(model, images, labels, eps, mean, std, targeted=False) -> torch.Tensor:
    """
    FGSM 攻击
    
    Args:
        targeted: 若为 True，labels 应为目标标签，执行针对性攻击
    """
    e_n, _, lower, upper = _get_norm_params(eps, None, mean, std)
    images = images.clone().detach().requires_grad_(True)

    loss = F.cross_entropy(model(images), labels)
    loss.backward()

    # targeted: 梯度下降靠近目标; untargeted: 梯度上升远离真实标签
    sign = -1 if targeted else 1
    adv = images + sign * e_n * images.grad.sign()
    return torch.max(torch.min(adv, upper), lower).detach()



def pgd_attack(model, images, labels, eps, alpha, steps, mean, std, targeted=False) -> torch.Tensor:
    """
    PGD 攻击
    
    Args:
        targeted: 若为 True，labels 应为目标标签，执行针对性攻击
    """
    e_n, a_n, lower, upper = _get_norm_params(eps, alpha, mean, std)
    adv = (images + torch.empty_like(images).uniform_(-1, 1) * e_n).clamp(lower, upper)

    # targeted: 梯度下降靠近目标; untargeted: 梯度上升远离真实标签
    sign = -1 if targeted else 1
    
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), labels)
        model.zero_grad()
        loss.backward()

        # 迭代更新与投影
        adv = images + (adv + sign * a_n * adv.grad.sign() - images).clamp(-e_n, e_n)
        adv = adv.clamp(lower, upper).detach()
    return adv



# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 攻击方式扩展 (TODO)                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def cw_attack(
    model, images, labels, c: float = 1.0, kappa: float = 0.0, 
    steps: int = 1000, lr: float = 0.01, mean=None, std=None
) -> torch.Tensor:
    """
    C&W (Carlini & Wagner) L2 攻击 - TODO
    
    基于优化的攻击方法，最小化 L2 扰动同时使模型误分类。
    
    参考论文: "Towards Evaluating the Robustness of Neural Networks" (Carlini & Wagner, 2017)
    
    Args:
        model: 目标模型
        images: 输入图像 [B, C, H, W]
        labels: 真实标签 [B]
        c: 置信度权重
        kappa: 置信度边界
        steps: 优化迭代次数
        lr: 学习率
        mean, std: 数据集标准化参数
    
    Returns:
        对抗样本 [B, C, H, W]
    """
    raise NotImplementedError("C&W 攻击尚未实现，可使用 advertorch 或 foolbox 库")


def auto_attack(
    model, images, labels, eps: float, norm: str = "Linf",
    version: str = "standard", mean=None, std=None
) -> torch.Tensor:
    """
    AutoAttack - 当前最强对抗评估基准 - TODO
    
    组合多种攻击: APGD-CE, APGD-DLR, FAB, Square Attack
    
    参考论文: "Reliable evaluation of adversarial robustness with an ensemble of diverse 
              parameter-free attacks" (Croce & Hein, 2020)
    
    Args:
        model: 目标模型
        images: 输入图像 [B, C, H, W]
        labels: 真实标签 [B]
        eps: 扰动预算
        norm: 范数类型 ("Linf" 或 "L2")
        version: 版本 ("standard", "plus", "rand")
        mean, std: 数据集标准化参数
    
    Returns:
        对抗样本 [B, C, H, W]
    
    安装: pip install autoattack
    """
    raise NotImplementedError("AutoAttack 尚未实现，请安装 autoattack 库")


# 注意: 针对性攻击已通过 fgsm_attack/pgd_attack 的 targeted 参数支持
# 用法: pgd_attack(model, x, target_labels, ..., targeted=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 对抗鲁棒性评估                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def evaluate_adversarial(
    trainer_or_models, loader, 
    eps: Union[float, List[float]] = None, 
    alpha: float = None, 
    steps: int = None, 
    dataset: str = None, 
    cfg=None, 
    logger=None
) -> Dict:
    """
    集成对抗鲁棒性评估
    
    Args:
        trainer_or_models: Trainer 对象或模型列表
        loader: 测试数据 DataLoader
        eps: 扰动预算 (可选)，支持:
             - None: 从 cfg 自动读取 (adv_eps_list 或 adv_eps)
             - float: 单值评估，如 8/255
             - list: 多 ε 评估，如 [2/255, 4/255, 8/255, 16/255]
        alpha: PGD 步长 (可选，None 则从 cfg 读取)
        steps: PGD 迭代次数 (可选，None 则从 cfg 读取)
        dataset: 数据集名称 (可选，None 则从 cfg 读取)
        cfg: 配置对象
        logger: 日志记录器
    
    Returns:
        - 单 ε: {clean_acc, fgsm_acc, pgd_acc, ...}
        - 多 ε: {eps_value: {clean_acc, fgsm_acc, pgd_acc, ...}, ...}
    """
    # 1. 自动解析参数 (优先使用显式参数，否则使用 cfg)
    if cfg is not None:
        if eps is None:
            # 优先检查多 eps 列表
            eps = getattr(cfg.constants, 'adv_eps_list', None)
            if eps is None:
                eps = getattr(cfg.constants, 'adv_eps', 0.03137)
        
        alpha = alpha if alpha is not None else getattr(cfg.constants, 'adv_alpha', 0.00784)
        steps = steps if steps is not None else getattr(cfg.constants, 'adv_pgd_steps', 10)
        dataset = dataset if dataset is not None else getattr(cfg.base, 'dataset_name', 'cifar10')
    else:
        # 无配置时的默认兜底
        eps = eps if eps is not None else 0.03137
        alpha = alpha if alpha is not None else 0.00784
        steps = steps if steps is not None else 10
        if dataset is None:
            raise ValueError("未提供 cfg 时必须显式指定 dataset 名称")

    # 2. 多 ε 模式: 递归调用自身
    if isinstance(eps, (list, tuple)):
        log = logger or get_logger()
        log.info(f"\n🗡️ Multi-ε Adversarial Eval ({len(eps)} values)")
        return {
            e: evaluate_adversarial(
                trainer_or_models, loader, e, alpha, steps, dataset, cfg, logger
            ) for e in eps
        }
    
    # 3. 单 ε 模式: 核心评估逻辑
    return _evaluate_single_eps(
        trainer_or_models, loader, eps, alpha, steps, dataset, cfg, logger
    )


def _evaluate_single_eps(
    trainer_or_models, loader, eps: float, alpha, steps, dataset, cfg, logger
) -> Dict:
    """单 ε 对抗评估核心逻辑"""
    from tqdm import tqdm
    from .strategies import get_ensemble_fn

    log = logger or get_logger()
    log.info(f"\n🗡️ Adversarial Eval (ε={eps * 255:.1f}/255, Steps={steps})")

    models, device = get_models_from_source(trainer_or_models)
    mean, std = _get_dataset_norm(dataset, device)

    # 建立集成攻击外壳（使用配置的集成策略）
    ensemble_fn = get_ensemble_fn(cfg) if cfg else None
    ens_model = _EnsembleProxy(models, ensemble_fn).to(device).eval()
    stats = {"total": 0, "clean": 0, "fgsm": 0, "pgd": 0}
    
    # 从配置读取针对性攻击开关
    targeted = getattr(cfg.constants, 'adv_targeted', False) if cfg else False

    pbar = tqdm(loader, desc=f"Adv ε={eps*255:.0f}", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        stats["total"] += x.size(0)

        # 1. 干净精度
        with torch.no_grad():
            stats["clean"] += (ens_model(x).argmax(1) == y).sum().item()

        # 2. 针对性攻击: 生成目标标签 (随机选择非真实类别)
        if targeted:
            num_classes = ens_model(x[:1]).shape[-1]  # 获取类别数
            attack_labels = _generate_target_labels(y, num_classes, device)
        else:
            attack_labels = y

        # 3. 对抗攻击 (FGSM/PGD)
        stats["fgsm"] += _run_and_eval_attack(
            ens_model, fgsm_attack, x, attack_labels, eps, mean, std, targeted
        )
        stats["pgd"] += _run_and_eval_attack(
            ens_model, pgd_attack, x, attack_labels, eps, alpha, steps, mean, std, targeted
        )

        pbar.set_postfix(
            {
                k: f"{100 * v / stats['total']:.1f}%"
                for k, v in stats.items()
                if k != "total"
            }
        )

    return _summarize_adv_results(stats, eps, alpha, steps, log)


class _EnsembleProxy(nn.Module):
    def __init__(self, models, ensemble_fn=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self._ensemble_fn = ensemble_fn or (lambda x: x.mean(0))

    def forward(self, x):
        stacked = torch.stack([m(x) for m in self.models])
        return self._ensemble_fn(stacked)


def _get_dataset_norm(name, device):
    from ..datasets import DATASET_REGISTRY

    cls = DATASET_REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(
            f"不支持的数据集: {name}. 可用: {list(DATASET_REGISTRY.keys())}"
        )
    return torch.tensor(cls.MEAN).view(1, 3, 1, 1).to(device), torch.tensor(
        cls.STD
    ).view(1, 3, 1, 1).to(device)


def _generate_target_labels(true_labels: torch.Tensor, num_classes: int, device) -> torch.Tensor:
    """
    生成针对性攻击的目标标签
    
    随机选择一个不同于真实标签的类别作为攻击目标。
    
    Args:
        true_labels: 真实标签 [B]
        num_classes: 类别总数
        device: 设备
    
    Returns:
        目标标签 [B]，保证每个样本的目标类别 ≠ 真实类别
    """
    # 生成 [1, num_classes-1] 的随机偏移
    offsets = torch.randint(1, num_classes, true_labels.shape, device=device)
    # 目标 = (真实 + 偏移) % 类别数，保证不等于真实标签
    target_labels = (true_labels + offsets) % num_classes
    return target_labels


def _run_and_eval_attack(model, attack_fn, x, y, *args):
    """封装 攻击 -> 推理 -> 计数 逻辑"""
    prev_training = model.training
    model.train()  # 确保允许梯度计算
    for m in model.models:
        m.eval()  # BN 维持 eval

    adv_x = attack_fn(model, x, y, *args)

    model.train(prev_training)
    with torch.no_grad():
        return (model(adv_x).argmax(1) == y).sum().item()


def _summarize_adv_results(s, eps, alpha, steps, log):
    t = s["total"]
    res = {
        "clean_acc": 100 * s["clean"] / t,
        "fgsm_acc": 100 * s["fgsm"] / t,
        "pgd_acc": 100 * s["pgd"] / t,
        "eps": eps,
        "eps_255": eps * 255,
        "alpha": alpha,
        "pgd_steps": steps,
        "num_samples": t,
    }
    log.info(
        f"   ✅ Clean: {res['clean_acc']:.2f}% | FGSM: {res['fgsm_acc']:.2f}% | PGD-{steps}: {res['pgd_acc']:.2f}%"
    )
    return res
