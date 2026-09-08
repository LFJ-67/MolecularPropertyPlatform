# MolecularPropertyPlatform

一个基于 Flask、RDKit 和 PyTorch Geometric 的分子性质预测 Web 平台，支持单分子预测和批量分子预测。

当前项目包含 5 类光化学/电化学性质预测模型：

- 三线态能量（Triplet Energy）
- 单线态能量（Singlet Energy）
- 氧化电势（Oxidation Potential）
- 还原电势（Reduction Potential）
- 激发波长（Absorption Wavelength）

平台支持直接输入 SMILES，也支持通过分子结构编辑器绘制分子并自动生成 SMILES。单分子预测页面还会同时展示分子的二维结构和若干基础分子描述符；批量预测支持 CSV、XLSX 和 XLS 文件上传、数据预览、性质选择以及结果文件下载。

> 本 README 基于当前项目代码、`requirements.txt` 和实际项目目录结构整理。未在代码中明确提供的训练数据、模型评价指标和部署配置没有在本文档中虚构。

---

## 1. 项目功能

### 1.1 单分子预测

在网页中输入一个合法的 SMILES 字符串，选择目标性质后进行预测。

支持：

| 预测目标 | 单位 |
|---|---|
| 三线态能量 | kcal/mol |
| 单线态能量 | kcal/mol |
| 氧化电势 | eV |
| 还原电势 | eV |
| 激发波长 | nm |

输入 SMILES 后，前端会：

1. 调用 `/structure` 接口生成分子二维结构图；
2. 获取分子基本信息；
3. 调用 `/predict` 接口执行目标性质预测；
4. 显示预测结果和推理时间。

### 1.2 分子结构编辑

前端集成 Kekule.js 分子编辑器，可点击“画分子”进入结构编辑界面。

绘制完成后，点击“使用该结构”，前端会将结构转换成 SMILES 并填入输入框。

> Kekule.js 当前通过 CDN 加载，因此使用结构编辑功能时需要能够访问相应的 CDN 资源。

### 1.3 分子基本信息

输入有效 SMILES 后，平台会通过 RDKit 计算并展示：

- 化学式
- 分子量（MolWt）
- TPSA
- LogP
- 氢键受体数（HBA）
- 氢键供体数（HBD）
- 可旋转键数
- 芳香环数

### 1.4 批量预测

批量预测支持上传：

- `.csv`
- `.xlsx`
- `.xls`

程序会自动识别常见的 SMILES 列名，包括：

```text
SMILE
smile
Smile
SMILES
smiles
Smiles
canonical_smiles
Canonical_SMILES
Structure
structure
```

上传成功后，页面会展示前 5 行数据预览，然后选择目标性质并开始批量预测。

预测完成后，结果文件会保存在 `results/` 目录中，并可从网页直接下载。

---

## 2. 项目结构

当前项目主要结构如下：

```text
MolecularPropertyPlatform/
│
├── .gitignore
├── app.py
├── config.py
├── predictor.py
├── 代码测试.py
├── 指令.txt
├── 预测_GAT_dropout_残差_改进.py
│
├── models/
│   ├── absorption/
│   │   ├── gnn_model.pth
│   │   └── scaler.pkl
│   │
│   ├── oxidation/
│   │   ├── gnn_model.pth
│   │   └── scaler.pkl
│   │
│   ├── reduction/
│   │   ├── gnn_model.pth
│   │   └── scaler.pkl
│   │
│   ├── singlet/
│   │   ├── gnn_model.pth
│   │   └── scaler.pkl
│   │
│   └── triplet/
│       ├── gnn_model.pth
│       ├── scaler.pkl        
│
├── services/
│   ├── batch_service.py
│   ├── prediction_service.py
│   ├── structure_service.py
│   └── __pycache__/
│
├── static/
│   ├── script.js
│   ├── style.css
│   └── images/
│       └── background.jpg
│
├── templates/
│   └── index.html
│
├── uploads/
│
└── results/
```

### 主要文件说明

| 文件 / 目录 | 作用 |
|---|---|
| `app.py` | Flask Web 应用入口和 HTTP 路由 |
| `config.py` | 5 个预测任务的模型路径、缩放器路径、名称和单位配置 |
| `predictor.py` | 加载全部模型和 scaler，并提供统一模型管理 |
| `预测_GAT_dropout_残差_改进.py` | 分子特征提取、图数据构建、GAT/残差模型定义和单分子预测 |
| `services/batch_service.py` | 批量文件读取、SMILES 列识别、批量预测和结果保存 |
| `services/structure_service.py` | SMILES 解析、二维结构图生成和分子描述符计算 |
| `services/prediction_service.py` | 当前版本文件存在，但代码中暂未放置实际逻辑 |
| `templates/index.html` | Flask HTML 页面 |
| `static/script.js` | 前端交互、单分子预测、批量上传和下载逻辑 |
| `static/style.css` | 页面样式 |
| `models/` | 预测所需的模型权重和 scaler |
| `uploads/` | 用户上传文件的运行时目录 |
| `results/` | 批量预测结果的运行时目录 |

---

## 3. 模型与特征

项目使用的核心预测模型为基于 PyTorch Geometric 的图神经网络结构。

代码中定义了：

- `StableResidualGAT`
- `StableGNN`

其中 `StableGNN` 综合使用三类信息：

1. 分子图表示（GAT + 残差结构）
2. Morgan Fingerprint
3. 分子级全局特征

### 3.1 图特征

代码对分子进行 RDKit 解析，并构建：

- 原子特征
- 键特征
- 分子图 `edge_index`
- 键属性 `edge_attr`

原子特征中包含原子序数、原子度数、形式电荷、氢数量、芳香性、环信息、原子质量、杂化方式、Gasteiger 电荷、电负性、极化率、价电子数、邻居电负性、芳香距离以及 π 相关特征等。

### 3.2 Morgan Fingerprint

模型使用：

```text
radius = 2
size = 2048
```

的 Morgan Fingerprint。

### 3.3 分子级全局特征

模型代码中使用了包括以下内容在内的分子级特征：

- 芳香环数
- 芳香原子数
- 共轭键数
- 环相关特征
- HBD
- HBA
- TPSA
- 可旋转键数
- BalabanJ
- BertzCT
- Chi0n
- Chi1n
- Kappa1
- Kappa2
- Gasteiger 电荷统计量

模型参数通过 `models/<target>/gnn_model.pth` 加载，预测结果通过对应的 `scaler.pkl` 执行反标准化。

---

## 4. 环境要求

当前项目的 `requirements.txt` 中指定了以下主要版本：

```text
Python 3.9.25
Flask 3.1.2
Werkzeug 3.1.3
pandas 2.3.3
numpy 1.26.4
torch 2.8.0
torch-geometric 2.5.0
requests 2.32.5
joblib 1.5.2
scikit-learn 1.6.1
openpyxl 3.1.5
rdkit-pypi 2025.03.5
```

项目在预测代码中会自动检查 CUDA：

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

因此代码本身支持在检测到 CUDA 时使用 GPU，否则使用 CPU。

---

## 5. 安装

### 5.1 创建虚拟环境

Windows PowerShell 示例：

```powershell
python -m venv .venv
```

激活：

```powershell
.venv\Scripts\Activate.ps1
```

如果使用 CMD：

```cmd
.venv\Scripts\activate
```

### 5.2 安装项目依赖

```powershell
pip install -r requirements.txt
```

### 5.3 PyTorch Geometric 相关依赖

当前 `requirements.txt` 中额外给出了 CPU 环境下的安装命令：

```powershell
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
```

如果环境中已经能够正常导入相关 PyTorch Geometric 组件，则不需要重复安装。

> 如果你准备使用 GPU，建议根据本机 CUDA、PyTorch 和 PyG 的实际组合调整对应的 PyG 扩展安装方式，而不要直接照搬 CPU 安装命令。

---

## 6. 启动 Web 平台

进入项目根目录后执行：

```powershell
python app.py
```

默认启动配置为：

```text
Host: 127.0.0.1
Port: 5000
Debug: False
Threaded: True
```

启动成功后，在浏览器访问：

```text
http://127.0.0.1:5000
```

程序启动时会初始化 `ModelManager`，并尝试加载 `config.py` 中配置的全部 5 个模型及对应 scaler。

如果模型文件不存在、路径错误或者模型权重与当前代码结构不匹配，程序可能在启动阶段加载模型时直接报错。

---

## 7. 单分子预测使用方法

### 第一步：输入 SMILES

在输入框中输入合法 SMILES，例如：

```text
C1(N=CN2)=C2C=CC=C1
```

也可以点击“画分子”使用结构编辑器绘制分子。

### 第二步：选择预测性质

目前支持：

- 三线态能量
- 单线态能量
- 氧化电势
- 还原电势
- 激发波长

### 第三步：开始预测

点击“开始预测”。

系统会先调用结构接口生成二维结构和分子信息，然后调用预测接口获得结果。

预测结果页面会显示：

```text
预测性质
预测值 + 单位
推理时间
```

---

## 8. 批量预测使用方法

### 8.1 准备输入文件

支持：

```text
CSV
XLSX
XLS
```

输入文件至少需要存在一个能够被程序识别的 SMILES 列。

例如：

```csv
smiles
C
CC
CCC
c1ccccc1
```

其中程序会自动寻找前面列出的常见 SMILES 列名。

### 8.2 上传文件

在网页切换到：

```text
Batch Prediction
```

选择输入文件后点击：

```text
Upload & Preview
```

系统会：

1. 保存上传文件到 `uploads/`
2. 读取 CSV / Excel
3. 自动寻找 SMILES 列
4. 返回列名
5. 返回前 5 行数据用于预览

### 8.3 选择目标性质

选择：

```text
三线态能量
单线态能量
氧化电势
还原电势
激发波长
```

然后点击：

```text
开始批量预测
```

### 8.4 下载结果

预测完成后，结果文件会保存到：

```text
results/
```

输出文件名会在原始文件名前添加：

```text
result_
```

例如：

```text
input.csv
```

会生成：

```text
result_input.csv
```

CSV 输出使用：

```text
utf-8-sig
```

Excel 输出会保存为 Excel 文件。

预测结果会作为新列追加到原始数据中，列名对应所选择的预测性质，例如：

```text
三线态能量
```

---

## 9. 后端 API

当前 Flask 应用提供以下主要接口。

### `GET /`

返回主页面。

### `POST /predict`

用于单分子性质预测。

请求 JSON 示例：

```json
{
  "smiles": "C1(N=CN2)=C2C=CC=C1",
  "target": "triplet"
}
```

其中 `target` 支持：

```text
triplet
singlet
oxidation
reduction
absorption
```

成功响应结构类似：

```json
{
  "success": true,
  "data": {
    "property": "三线态能量",
    "prediction": 0.0,
    "unit": "kcal/mol",
    "time": 0.123
  }
}
```

### `POST /structure`

根据 SMILES 生成二维结构图和分子信息。

请求 JSON：

```json
{
  "smiles": "CCO"
}
```

返回内容包含：

- Base64 编码的 PNG 结构图
- 化学式
- 分子量
- TPSA
- LogP
- HBA
- HBD
- 可旋转键数
- 芳香环数

### `POST /upload`

上传 CSV / Excel 文件并获取数据预览。

请求方式：

```text
multipart/form-data
```

字段：

```text
file
```

### `POST /batch_predict`

执行批量预测。

请求 JSON：

```json
{
  "filename": "example.csv",
  "target": "triplet"
}
```

成功后返回生成的结果文件名。

### `GET /download/<filename>`

下载批量预测结果。

---

## 10. 模型文件目录

模型配置位于：

```text
config.py
```

对应目录：

```text
models/
├── triplet/
│   ├── gnn_model.pth
│   └── scaler.pkl
├── singlet/
│   ├── gnn_model.pth
│   └── scaler.pkl
├── oxidation/
│   ├── gnn_model.pth
│   └── scaler.pkl
├── reduction/
│   ├── gnn_model.pth
│   └── scaler.pkl
└── absorption/
    ├── gnn_model.pth
    └── scaler.pkl
```

其中：

- `gnn_model.pth`：模型权重
- `scaler.pkl`：对应目标性质的数据缩放器

程序启动时会加载全部模型，而不是等到第一次预测时再加载单个模型。

因此部署或复制项目时，需要确保上述模型文件完整存在，并与代码中的模型结构兼容。

---

## 11. 数据与运行时目录

### `uploads/`

用于保存网页上传的输入文件。

该目录属于运行时数据目录，不建议将实际用户上传数据长期提交到 GitHub。

### `results/`

用于保存批量预测生成的结果文件。

同样建议作为运行时数据目录处理，不将运行产生的大量结果提交到 GitHub。

### `__pycache__/`

Python 运行自动生成的缓存目录，不属于项目源代码，建议通过 `.gitignore` 忽略。

---

## 12. 本地测试

项目包含：

```text
代码测试.py
```

该脚本使用 `requests` 调用本地 Flask 服务，主要测试：

1. `/upload`
2. `/batch_predict`
3. `/download/<filename>`

脚本当前默认读取：

```python
open('test.csv', 'rb')
```

因此运行测试脚本前，需要在项目当前工作目录准备一个：

```text
test.csv
```

并确保 Flask 服务已经启动。

运行：

```powershell
python 代码测试.py
```

---

## 13. Git 管理建议

建议将 Python 缓存、虚拟环境、日志以及运行时上传/结果文件加入 `.gitignore`。

一个适合本项目的 `.gitignore` 至少应包含：

```gitignore
__pycache__/
**/__pycache__/
*.py[cod]

.venv/
venv/
env/

.vscode/
.idea/

.env
.env.*

uploads/*
results/*

*.log
*.tmp
*.temp

.DS_Store
Thumbs.db
```

---

## 14. 当前版本的注意事项

### 模型文件必须完整

项目的预测功能依赖 `models/` 中的 `.pth` 和 `.pkl` 文件，缺少任意目标性质的模型文件都可能导致应用启动失败。

### SMILES 必须能够被 RDKit 解析

如果 SMILES 无法解析，结构生成和预测都会失败或返回错误信息。

### 批量文件必须包含可识别的 SMILES 列

如果 CSV / Excel 中没有程序定义的候选列名，批量预测会报：

```text
Cannot find SMILES column.
```

### 运行时目录建议不要提交数据

`uploads/` 和 `results/` 用于运行时文件。实际部署时应结合项目使用场景考虑文件清理、权限和存储策略。

### 当前项目没有提供训练流程和评价指标

---

## 15. 技术栈

项目当前主要使用：

- Python
- Flask
- RDKit
- PyTorch
- PyTorch Geometric
- pandas
- NumPy
- scikit-learn
- joblib
- openpyxl
- JavaScript
- HTML / CSS
- Kekule.js

---

## 16. 许可证

本项目根据 **MIT License** 进行许可。

你可以自由使用、复制、修改、合并、发布、分发、再授权和销售本软件的副本，但需遵守 MIT 许可证的条款。

完整的许可文本请参见 [LICENSE](LICENSE) 文件。

> 注意：第三方库、依赖项、数据集和模型组件可能受到它们各自的许可或使用限制。在再分发前请务必检查相关许可。


## 17. 快速开始

最简启动流程：

```powershell
# 1. 进入项目目录
cd MolecularPropertyPlatform

# 2. 创建虚拟环境
python -m venv .venv

# 3. 激活虚拟环境
.venv\Scripts\Activate.ps1

# 4. 安装依赖
pip install -r requirements.txt

# 5. 启动 Flask
python app.py
```

然后打开：

```text
http://127.0.0.1:5000
```

即可进入 Molecular Property Prediction Platform。

---

## 18. 项目定位

本项目的核心目标是将分子结构解析、分子描述符计算、图神经网络推理以及 Web 交互整合到一个统一的平台中，为单分子和批量分子的性质预测提供直观的操作界面。

当前版本重点覆盖：

```text
SMILES
  ↓
RDKit 分子解析
  ↓
分子图 + Morgan Fingerprint + 分子级特征
  ↓
StableGNN / Residual GAT
  ↓
Scaler 反标准化
  ↓
预测结果
```

同时提供：

```text
CSV / Excel
  ↓
SMILES 列自动识别
  ↓
批量推理
  ↓
结果文件
```

