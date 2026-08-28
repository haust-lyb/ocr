# OCR 布局识别服务

基于 PaddleX + vLLM 视觉语言模型的文档版面分析与文字识别服务。

## 启动服务

```bash
cd /Users/liyibo/temp/ocrserver/ocr_server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## 重启服务

1. 停止当前服务（Ctrl+C）
2. 重新运行启动命令

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## 依赖服务

- **vLLM**：确保视觉模型已通过 OpenAI-compatible API 启动
  ```bash
  vllm serve /path/to/glm-ocr --served-model-name glm-ocr --port 8001
  export VLLM_BASE_URL=http://127.0.0.1:8001/v1
  export VLLM_API_KEY=EMPTY
  export VLLM_MODEL=glm-ocr
  ```

## 访问地址

- 首页：http://localhost:8000/
- API 文档：http://localhost:8000/docs
- 控制台：http://localhost:8000/console
- 可视化查看器：http://localhost:8000/viewer?id={识别ID}

## 项目结构

```
ocr_server/
├── app.py              # FastAPI 主应用
├── model.py            # PaddleX 版面识别模型
├── vllm_client.py      # vLLM OpenAI-compatible 客户端
├── storage.py          # SQLite 存储模块
├── requirements.txt    # Python 依赖
├── view/               # 前端页面
│   └── viewer.html     # 可视化查看器
├── target/             # 数据存储（自动创建，忽略版本控制）
│   ├── images/         # 上传图片存储
│   └── recognitions.db # SQLite 数据库
└── README.md
```
