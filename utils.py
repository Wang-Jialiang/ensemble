"""
================================================================================
工具模块 - 日志、通用工具函数
================================================================================
"""

import datetime
import logging
from pathlib import Path
from typing import Optional, Union

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 常量定义                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

DEFAULT_DATA_ROOT = str(Path(__file__).resolve().parent / "data")
DEFAULT_SAVE_ROOT = str(Path(__file__).resolve().parent / "checkpoints")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 日志系统                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def get_logger(
    log_level: int = logging.INFO, log_file: Optional[str] = None
) -> logging.Logger:
    """获取全局 logger 实例 (延迟初始化 + 单例模式)

    利用 logging.getLogger 的内置单例特性，相同名称的 logger 在整个进程中是同一个对象。

    Args:
        log_level: 日志级别 (默认 INFO，仅首次调用生效)
        log_file: 日志文件路径 (可选，仅首次调用生效)

    Returns:
        配置好的 logger 实例
    """
    logger = logging.getLogger("NDE")

    # 已配置过，直接返回 (单例)
    if logger.handlers:
        return logger

    logger.setLevel(log_level)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    # 文件处理器 (可选)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

    return logger


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 工具函数                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def set_seed(seed: int):
    """设置所有相关的随机种子"""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_duration(seconds: float) -> str:
    """将秒数格式化为 h:m:s 格式"""
    return str(datetime.timedelta(seconds=int(seconds)))


def ensure_dir(path: Union[str, Path]) -> Path:
    """确保目录存在"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
