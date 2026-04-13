import os
import ollama
from PIL import Image
import io
import base64

OLLAMA_MODEL = "glm-ocr:latest"
#OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://10.15.15.164:11434")


def encode_image_to_base64(img_array) -> str:
    """numpy array (cv2 format, BGR) → base64 PNG"""
    pil_img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def call_glm_ocr(
    image,
    prompt: str = "请识别图片中的文本内容，直接输出识别结果，不要多余解释。",
    host: str = None,
) -> str:
    """
    调用 Ollama glm-ocr 模型进行 OCR 识别

    image: numpy array (cv2 format, BGR)
    host: Ollama 服务地址，默认从环境变量 OLLAMA_HOST 读取
    """
    img_base64 = encode_image_to_base64(image)
    client = ollama.Client(host=host or OLLAMA_HOST)
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [img_base64]
        }]
    )
    return response.message.content


