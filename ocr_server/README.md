# OCR 布局识别服务

基于 PaddleX + Ollama glm-ocr 的文档版面分析与文字识别服务。

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

- **Ollama**：确保 glm-ocr 模型已下载，服务运行中
  ```bash
  ollama serve
  ollama pull glm-ocr:latest
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
├── ollama_client.py    # Ollama glm-ocr 客户端
├── storage.py          # SQLite 存储模块
├── requirements.txt    # Python 依赖
├── view/               # 前端页面
│   └── viewer.html     # 可视化查看器
├── target/             # 数据存储（自动创建，忽略版本控制）
│   ├── images/         # 上传图片存储
│   └── recognitions.db # SQLite 数据库
└── README.md
```
