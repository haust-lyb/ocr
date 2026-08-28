#!/bin/bash
sudo docker run -d \
  --gpus '"device=0"' \
  -e VLLM_BASE_URL=http://10.15.15.16:3004/v1 \
  -e VLLM_API_KEY=EMPTY \
  -e VLLM_MODEL=glm-ocr \
  -p 7788:8000 \
  --restart=always \
  --name ocr-server \
  ocr-server:gpu
