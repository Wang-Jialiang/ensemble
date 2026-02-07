"""
================================================================================
主入口模块 - CLI解析和程序入口
================================================================================
"""

# ========== Windows 兼容性配置 ==========
import io
import sys

if sys.platform == "win32":
    import torch

    # 禁用 Dynamo 编译 (Windows 不支持 Triton)
    torch._dynamo.config.disable = True
    torch._inductor.config.compile_threads = 1
    # 关闭相关优化
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    # 修复终端输出中文/emoji 乱码
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


sys.dont_write_bytecode = True  # 禁用 __pycache__ 生成

import argparse
import datetime
from pathlib import Path

from .config import Config
from .datasets import configure_dataset_params, load_dataset
from .evaluation import ReportGenerator
from .training import train_experiment
from .utils import ensure_dir, get_logger, set_seed


def main():
    """NDE 系统主入口: 解析 -> 初始化 -> 分发"""
    args = _parse_args()
    cfg = _init_config(args)
    set_seed(cfg.seed)

    if args.eval:
        _run_evaluation(cfg, args)
    else:
        _run_training(cfg)


def _parse_args():
    p = argparse.ArgumentParser(description="NDE")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--quick-test", action="store_true")
    return p.parse_args()


def _init_config(args):
    config_name = "config/default.yaml"
    path = Path(config_name)
    if not path.exists():
        path = Path(__file__).parent / config_name

    base_cfg, experiments, eval_ckpts = Config.load_yaml(str(path))

    # 应用 quick_test 模式 (必须在 configure_dataset_params 之前)
    # 因为 apply_quick_test() 使用 replace() 会创建新对象，丢失 init=False 字段
    if args.quick_test:
        base_cfg = base_cfg.apply_quick_test()

    # [New] 手动触发数据集配置 (解耦合)
    configure_dataset_params(base_cfg)

    # 将实验列表挂载到配置对象上便于后续传递 (临时)
    base_cfg._experiments = experiments
    base_cfg._eval_ckpts = eval_ckpts
    return base_cfg


def _run_training(cfg):
    log = get_logger()
    log.info(
        f"🚀 NDE | Training Mode | Models: {cfg.total_models} | Experiments: {len(cfg._experiments)}"
    )

    train_loader, val_loader = load_dataset(cfg, mode="train")

    # 生成统一的训练批次目录 (所有实验共享)
    batch_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = Path(cfg.save_root) / "training" / batch_ts
    ensure_dir(batch_dir)
    log.info(f"📁 Training batch dir: {batch_dir}")

    for exp in cfg._experiments:
        log.info(f"\n🧪 Running: {exp.name}")

        # 🔑 关键：重置随机种子，确保每个实验从相同初始状态开始
        # 这保证了不同实验之间唯一的差异只有遮挡图案
        set_seed(cfg.seed)

        c = cfg.copy(experiment_name=exp.name, **exp.get_config_overrides())

        # 每个实验作为子目录
        c.training_base_dir = str(batch_dir)  # 共享时间戳目录 (日志/历史)
        c.save_dir = str(batch_dir / exp.name)  # 实验子目录 (检查点)
        ensure_dir(c.save_dir)

        train_experiment(cfg=c, train_loader=train_loader, val_loader=val_loader)


def _run_evaluation(cfg, args):
    log = get_logger()
    log.info(f"🚀 NDE | Evaluation Mode | Checkpoints: {len(cfg._eval_ckpts)}")

    if not cfg._eval_ckpts:
        get_logger().error("❌ No checkpoints for evaluation")
        return

    test_loader, c_ds, o_ds = load_dataset(cfg, mode="eval")
    ckpts = [c["path"] for c in cfg._eval_ckpts]

    ReportGenerator.evaluate_checkpoints(
        checkpoint_paths=ckpts,
        test_loader=test_loader,
        cfg=cfg,
        corruption_dataset=c_ds,
        ood_dataset=o_ds,
        run_gradcam=cfg.eval_run_gradcam,
        run_loss_landscape=cfg.eval_run_landscape,
    )


if __name__ == "__main__":
    main()
