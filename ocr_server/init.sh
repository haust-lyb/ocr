#!/bin/bash
sudo docker run -d \
  --gpus '"device=0"' \
  -e OLLAMA_HOST=http://10.15.15.16:11434 \
  -e OLLAMA_MODEL=glm-ocr:bf16 \
  -p 7788:8000 \
  --restart=always \
  --name ocr-server \
  ocr-server:gpu