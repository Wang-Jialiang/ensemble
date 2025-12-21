"""
================================================================================
新集成方法入口脚本
================================================================================

独立的入口脚本，用于运行新增的 4 种集成方法，不修改 main.py。

支持方法:
- batch: Batch Ensemble
- snapshot: Snapshot Ensemble
- distill: Knowledge Distillation (需要先训练教师模型)
- grouped: 使用 Grouped Ensemble 策略评估

使用示例:
    python run_methods.py --method batch --quick-test
    python run_methods.py --method snapshot --num-cycles 5
    python run_methods.py --method distill --teacher-path ./checkpoints/exp
"""

import argparse
from pathlib import Path

from ensemble.config import Config
from ensemble.datasets import load_dataset
from ensemble.utils import get_logger, set_seed

DEFAULT_CONFIG = Path(__file__).parent / "ensemble" / "config" / "default.yaml"


def run_batch_ensemble(cfg: Config, train_loader, val_loader, args):
    """运行 Batch Ensemble"""
    from ensemble.training.batch_trainer import train_batch_ensemble

    trainer, training_time = train_batch_ensemble(
        experiment_name="batch_ensemble",
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        num_members=args.num_members,
    )
    get_logger().info(f"✅ Batch Ensemble completed in {training_time:.1f}s")
    return trainer


def run_snapshot_ensemble(cfg: Config, train_loader, val_loader, args):
    """运行 Snapshot Ensemble"""
    from ensemble.training.snapshot_trainer import train_snapshot_ensemble

    trainer, training_time = train_snapshot_ensemble(
        experiment_name="snapshot_ensemble",
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        num_cycles=args.num_cycles,
    )
    get_logger().info(f"✅ Snapshot Ensemble completed in {training_time:.1f}s")
    return trainer


def run_distillation(cfg: Config, train_loader, val_loader, args):
    """运行 Knowledge Distillation"""
    import torch

    from ensemble.models import ModelFactory
    from ensemble.training.distillation import train_distillation

    # 加载教师模型
    if not args.teacher_path:
        get_logger().error("❌ --teacher-path is required for distillation")
        return None

    teacher_path = Path(args.teacher_path)
    if not teacher_path.exists():
        get_logger().error(f"❌ Teacher path not found: {teacher_path}")
        return None

    # 查找教师模型检查点
    teacher_models = []
    model_files = list(teacher_path.glob("**/model_*.pth"))
    if not model_files:
        model_files = list(teacher_path.glob("**/snapshot_*.pth"))

    if not model_files:
        get_logger().error(f"❌ No model files found in: {teacher_path}")
        return None

    device = torch.device(f"cuda:{cfg.gpu_ids[0]}" if cfg.gpu_ids else "cuda:0")

    for model_file in model_files[: args.num_teachers]:
        model = ModelFactory.create_model(cfg.model_name, num_classes=cfg.num_classes)
        model.load_state_dict(torch.load(model_file, weights_only=True))
        model.to(device).eval()
        teacher_models.append(model)

    get_logger().info(f"📚 Loaded {len(teacher_models)} teacher models")

    trainer, training_time = train_distillation(
        experiment_name="distilled_student",
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        teacher_models=teacher_models,
        temperature=args.temperature,
        alpha=args.alpha,
    )
    get_logger().info(f"✅ Distillation completed in {training_time:.1f}s")
    return trainer


def run_grouped_eval(cfg: Config, test_loader, args):
    """使用 Grouped 策略评估"""
    import torch

    # 导入 grouped 策略 (从包顶层导入)
    from ensemble.evaluation import ENSEMBLE_STRATEGIES
    from ensemble.models import ModelFactory

    if not args.checkpoint_path:
        get_logger().error("❌ --checkpoint-path is required for grouped evaluation")
        return

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        get_logger().error(f"❌ Checkpoint path not found: {checkpoint_path}")
        return

    # 加载模型
    device = torch.device(f"cuda:{cfg.gpu_ids[0]}" if cfg.gpu_ids else "cuda:0")
    model_files = list(checkpoint_path.glob("**/model_*.pth"))

    if not model_files:
        get_logger().error(f"❌ No model files found in: {checkpoint_path}")
        return

    models = []
    for model_file in model_files:
        model = ModelFactory.create_model(cfg.model_name, num_classes=cfg.num_classes)
        model.load_state_dict(torch.load(model_file, weights_only=True))
        model.to(device).eval()
        models.append(model)

    get_logger().info(f"📦 Loaded {len(models)} models")

    # 评估
    correct = 0
    total = 0

    strategy = args.strategy
    ensemble_fn = ENSEMBLE_STRATEGIES.get(strategy, ENSEMBLE_STRATEGIES["grouped"])
    get_logger().info(f"🎯 Using strategy: {strategy}")

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 收集所有模型预测
            all_logits = []
            for model in models:
                logits = model(inputs)
                all_logits.append(logits)

            all_logits = torch.stack(
                all_logits
            )  # [num_models, batch_size, num_classes]
            ensemble_logits = ensemble_fn(all_logits)

            preds = ensemble_logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    accuracy = 100.0 * correct / total
    get_logger().info(f"✅ Grouped Ensemble Accuracy: {accuracy:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="New Ensemble Methods")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["batch", "snapshot", "distill", "grouped"],
        help="Ensemble method to run",
    )
    parser.add_argument("--quick-test", action="store_true", help="Quick test mode")

    # Batch Ensemble
    parser.add_argument(
        "--num-members", type=int, default=4, help="Batch ensemble members"
    )

    # Snapshot Ensemble
    parser.add_argument("--num-cycles", type=int, default=5, help="Snapshot cycles")

    # Distillation
    parser.add_argument("--teacher-path", type=str, help="Path to teacher checkpoints")
    parser.add_argument(
        "--num-teachers", type=int, default=4, help="Max teachers to load"
    )
    parser.add_argument("--temperature", type=float, default=4.0, help="Distillation T")
    parser.add_argument("--alpha", type=float, default=0.7, help="Soft label weight")

    # Grouped Evaluation
    parser.add_argument("--checkpoint-path", type=str, help="Path to checkpoints")
    parser.add_argument(
        "--strategy",
        type=str,
        default="grouped",
        choices=[
            "grouped",
            "grouped_2",
            "grouped_3",
            "grouped_4",
            "hierarchical_voting",
        ],
        help="Grouping strategy",
    )

    args = parser.parse_args()

    # 加载配置
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = Path(__file__).parent / "ensemble" / "config" / "default.yaml"

    if not cfg_path.exists():
        print(f"❌ Config file not found: {cfg_path}")
        return

    base_cfg, _, _ = Config.load_yaml(str(cfg_path))

    if args.quick_test:
        base_cfg = base_cfg.apply_quick_test()

    set_seed(base_cfg.seed)

    get_logger().info(f"🚀 Running {args.method} method")
    get_logger().info(f"   Model: {base_cfg.model_name}")
    get_logger().info(f"   Dataset: {base_cfg.dataset_name}")

    # 方法路由
    if args.method == "grouped":
        _, _, test_loader, _ = load_dataset(base_cfg)
        run_grouped_eval(base_cfg, test_loader, args)
    else:
        train_loader, val_loader, _, _ = load_dataset(base_cfg)

        if args.method == "batch":
            run_batch_ensemble(base_cfg, train_loader, val_loader, args)
        elif args.method == "snapshot":
            run_snapshot_ensemble(base_cfg, train_loader, val_loader, args)
        elif args.method == "distill":
            run_distillation(base_cfg, train_loader, val_loader, args)


if __name__ == "__main__":
    main()
