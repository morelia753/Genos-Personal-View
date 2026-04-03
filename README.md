# Genos-Personal-Server Backend

`backend` 是本项目的 FastAPI 推理服务，负责：

- 启动时加载模型与参考配置
- 接收上传的 FASTA / VCF 文件并缓存
- 支持 FASTA 模式与 VCF 模式 两种预测方式
- 返回前端 IGV 渲染所需 payload（正负链轨道）

## 1. 目录结构

```text
backend/
├── api.py                  # FastAPI 路由与生命周期
├── main.py                 # 启动入口（uvicorn）
├── prediction_service.py   # 预测核心逻辑（FASTA/VCF）
├── igv_payload.py          # IGV payload 构建
├── predict_user_region_online.py
├── run_backend.sh          # 后台启动脚本（nohup）
├── stop_backend.sh         # 停止脚本
├── requirements.txt
└── logs/
    ├── backend.nohup.log
    └── backend.pid
```

## 2. 运行前准备

### 2.1 Python 依赖

在项目根目录执行：

```bash
pip install -r backend/requirements.txt
```

说明：
- 项目包含 `torch` 与 `flash-attn` 依赖，建议使用已配置好的 CUDA 环境。
- `api.py` 会读取 `frontend/config.py` 中的路径和参数配置，因此请同时保证根目录 `.env` 配置完整。

### 2.2 环境变量（`.env`）

后端会自动读取以下两个文件（如果存在）：

1. `<repo_root>/.env`
2. `<repo_root>/frontend/.env`

建议在根目录维护统一 `.env`。以下变量对后端启动最关键：

- 服务与启动
  - `BACKEND_HOST`（默认 `0.0.0.0`）
  - `BACKEND_PORT`（`main.py` 默认 `8011`，建议在 `.env` 显式指定）
  - `ROOT_DIR`
  - `BACKEND_PYTHON_BIN`（推荐显式指定 Python 解释器）
  - `BACKEND_CONDA_ENV`（未指定 Python 时兜底）
- 模型与元数据
  - `BASE_MODEL_PATH`
  - `TOKENIZER_DIR`
  - `CHECKPOINT_PATH`
  - `INDEX_STAT_JSON`
  - `BIGWIG_LABELS_META_CSV`
- 参考基因组与注释
  - `HG38_FASTA_PATH`（VCF 模式会用到）
  - `LOCAL_FASTA_REL`
  - `LOCAL_FASTA_INDEX_REL`
  - `LOCAL_GTF_REL`
- 预测参数
  - `DEFAULT_GENOME`（当前仅支持 `hg38`）
  - `TARGET_LEN`（默认 `32768`）
  - `PREDICTION_MAX_POINTS`（默认 `900`）

## 3. 启动与停止

### 3.1 推荐：脚本启动

在项目根目录执行：

### 后端启动与停止
```bash
bash backend/run_backend.sh
```

停止：

```bash
bash backend/stop_backend.sh
```

### 前端启动与停止

python frontend/app.py


### 4.界面测试

测试数据路径：test_data

### fasta模式预测

![alt text](2475be0abf29102fa3df934a230435d5.png)


### VCF模式预测

![alt text](aad6483d310ddf8d2ba2e3132d2d4988.png)

