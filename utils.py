"""
================================================================================
工具模块
================================================================================
"""

import datetime
import logging
from pathlib import Path
from typing import Union

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 日志系统                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
from rich.console import Console
from rich.traceback import install

# 全局 Console 对象
console = Console()
# 自动接管异常显示 (关闭局部变量显示以减少输出)
install(show_locals=False)


def get_logger(
    level: int = logging.INFO, file: str = None, console: bool = True
) -> logging.Logger:
    """获取单例 NDE Logger (强制集成 Rich)。

    Args:
        level: 日志级别，默认为 logging.INFO。
        file: 日志文件路径，若指定则同时输出到文件。
        console: 是否输出到控制台，默认 True。

    Returns:
        logging.Logger: 配置好的 NDE 日志记录器实例。
    """
    log = logging.getLogger("NDE")
    if log.handlers:
        return log

    log.setLevel(level)

    # 终端输出: 使用 RichHandler (可选)
    if console:
        from rich.console import Console
        from rich.logging import RichHandler

        rh = RichHandler(console=Console(), rich_tracebacks=True, show_path=True)
        log.addHandler(rh)

    # 文件输出 (保持标准格式)
    if file:
        fh = logging.FileHandler(file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        log.addHandler(fh)
    return log


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 工具函数                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def set_seed(seed: int):
    """设置所有相关的随机种子以确保可复现性。

    Args:
        seed: 随机种子值。
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_duration(seconds: float) -> str:
    """将秒数格式化为 h:m:s 格式。

    Args:
        seconds: 时间长度（秒）。

    Returns:
        str: 格式化后的时间字符串，如 "1:23:45"。
    """
    return str(datetime.timedelta(seconds=int(seconds)))


def ensure_dir(path: Union[str, Path]) -> Path:
    """确保目录存在，若不存在则创建。

    Args:
        path: 目录路径，可以是字符串或 Path 对象。

    Returns:
        Path: 确保存在的目录 Path 对象。
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
