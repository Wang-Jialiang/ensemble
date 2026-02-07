# 🚀 Windows 环境部署指南

> 本指南确保所有内容（虚拟环境、数据、输出）都在 `ensemble` 文件夹内，删除此文件夹即可完全清理。

---

## 📁 最终目录结构

```
ensemble/                     # 项目根目录 - 删除此文件夹即可完全清理
├── venv/                     # Python 虚拟环境 (自动创建)
├── data/                     # 数据集存放位置 (自动创建)
├── output/                   # 训练/评估输出 (自动创建)
├── config/                   # 配置文件
├── datasets/                 # 数据集代码
├── evaluation/               # 评估代码
├── models/                   # 模型代码
├── training/                 # 训练代码
├── main.py                   # 主入口
├── requirements.txt          # 依赖列表
└── SETUP_GUIDE.md            # 本指南
```

---

## 🔧 安装步骤

### 步骤 1: 打开 PowerShell 并进入项目目录

```powershell
cd C:\Users\wangjialiang\Desktop\ensemble
```

### 步骤 2: 创建虚拟环境 (在项目内)

```powershell
python -m venv venv
```

### 步骤 3: 激活虚拟环境

```powershell
.\venv\Scripts\activate
```

> ⚠️ 激活后命令行前面会出现 `(venv)` 标识

### 步骤 4: 安装 PyTorch

根据你的显卡选择对应命令:

**NVIDIA 显卡 (CUDA 12.1):**

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 步骤 5: 安装其他依赖

```powershell
pip install -r requirements.txt
```

---

## ⚙️ 修改配置文件

编辑 `config/default.yaml`，修改第 60-61 行的路径为项目内相对路径:

```yaml
base:
  data_root: "./data"           # 数据存放在项目内的 data 文件夹
  save_root: "./output"         # 输出保存在项目内的 output 文件夹
```

> 💡 使用相对路径 `./` 可以确保数据和输出都在项目文件夹内

---

## ▶️ 运行项目

### 每次运行前先激活虚拟环境

```powershell
cd C:\Users\wangjialiang\Desktop\ensemble
.\venv\Scripts\activate
```

### 训练模式

```powershell
python -m ensemble
```

### 快速测试模式 (验证安装是否成功)

```powershell
python -m ensemble --quick-test
```

### 评估模式

```powershell
python -m ensemble --eval
```

---

## 🗑️ 完全卸载

只需删除整个 `ensemble` 文件夹即可，不会留下任何残留:

```powershell
# 先退出虚拟环境
deactivate

# 删除整个项目文件夹
Remove-Item -Recurse -Force C:\Users\wangjialiang\Desktop\ensemble
```

---

## ❓ 常见问题

### Q1: `python` 命令找不到

确保 Python 已安装并添加到 PATH。重新安装 Python 时勾选 "Add Python to PATH"。

### Q2: 激活虚拟环境报错 "无法加载脚本"

以管理员身份运行 PowerShell，执行:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Q3: CUDA 相关错误

1. 确认已安装 NVIDIA 显卡驱动
2. 运行 `nvidia-smi` 查看驱动版本
3. 选择兼容的 PyTorch CUDA 版本

### Q4: num_workers 相关警告

Windows 上建议在 `config/default.yaml` 中设置:

```yaml
num_workers: 0                # Windows 建议设为 0
```

---

## 📋 快速命令速查表

| 操作 | 命令 |
|------|------|
| 进入目录 | `cd C:\Users\wangjialiang\Desktop\ensemble` |
| 激活环境 | `.\venv\Scripts\activate` |
| 退出环境 | `deactivate` |
| 安装依赖 | `pip install -r requirements.txt` |
| 训练 | `python -m ensemble` |
| 快速测试 | `python -m ensemble --quick-test` |
| 评估 | `python -m ensemble --eval` |
