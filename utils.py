"""
================================================================================
工具模块
================================================================================
"""

import datetime
import logging
from pathlib import Path
from typing import Optional, Union

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 日志系统                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

# 全局 Console 对象
console = Console()
# 自动接管异常显示 (带局部变量显示)
install(show_locals=True)


def get_logger(level: int = logging.INFO, file: str = None) -> logging.Logger:
    """获取单例 NDE Logger (强制集成 Rich)"""
    log = logging.getLogger("NDE")
    if log.handlers:
        return log

    log.setLevel(level)

    # 终端输出: 强制使用 RichHandler
    rh = RichHandler(console=console, rich_tracebacks=True, show_path=True)
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
