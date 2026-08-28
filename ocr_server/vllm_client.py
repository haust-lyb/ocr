import base64
import io
import os
import re
from typing import Optional

from openai import OpenAI
from PIL import Image


VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
VLLM_MODEL = os.getenv("VLLM_MODEL", "glm-ocr")
VLLM_REORDER_MODEL = os.getenv("VLLM_REORDER_MODEL", "")
VLLM_MAX_TOKENS = int(os.getenv("VLLM_MAX_TOKENS", "4096"))
VLLM_TIMEOUT = float(os.getenv("VLLM_TIMEOUT", "120"))


def encode_image_to_base64(img_array) -> str:
    """将 numpy 图片数组编码为 base64 PNG。"""
    pil_img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _client(base_url: Optional[str] = None) -> OpenAI:
    return OpenAI(
        base_url=base_url or VLLM_BASE_URL,
        api_key=VLLM_API_KEY,
        timeout=VLLM_TIMEOUT,
    )


def _chat(prompt: str, image=None, model: Optional[str] = None, base_url: Optional[str] = None) -> str:
    if image is None:
        content = prompt
    else:
        image_base64 = encode_image_to_base64(image)
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"},
            },
        ]

    response = _client(base_url).chat.completions.create(
        model=model or VLLM_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=VLLM_MAX_TOKENS,
    )
    return response.choices[0].message.content or ""


def call_glm_ocr(
    image,
    prompt: str = "请识别图片中的文本内容，直接输出识别结果，不要多余解释。",
    host: Optional[str] = None,
) -> str:
    """通过 vLLM OpenAI-compatible API 调用视觉模型进行 OCR。"""
    return _chat(prompt, image=image, base_url=host)


def call_glm_markdown(image, host: Optional[str] = None) -> str:
    """通过 vLLM 将图片识别为 Markdown。"""
    prompt = "请将图片中的内容识别为 Markdown 格式输出，保持原有的标题、列表、表格、代码块等结构，只输出 Markdown 原文，不要额外解释。"
    return _chat(prompt, image=image, base_url=host)


def call_glm_extract(image, prompt_json: str, host: Optional[str] = None) -> str:
    """通过 vLLM 根据 JSON 格式要求进行结构化提取。"""
    prompt = (
        "请根据以下 JSON 格式要求，从图片中提取对应字段信息，"
        "严格按照 JSON 格式输出，不要多余解释。\n\n"
        f"要求格式：\n{prompt_json}"
    )
    return _chat(prompt, image=image, base_url=host)


def reorder_layout_elements(boxes: list, model: Optional[str] = None) -> list:
    """调用 vLLM 文本模型，根据 OCR 结果的语义连贯性调整元素顺序。"""
    use_model = model or VLLM_REORDER_MODEL
    if not use_model:
        return boxes

    elements = []
    for i, box in enumerate(boxes):
        label = box.get("label", "unknown")
        ocr_text = box.get("ocr_text", "").strip()
        if ocr_text:
            text = ocr_text[:200].replace("\n", " ")
            elements.append(f"[{i}] [{label}] {text}")
        elif label in ("image", "chart", "footer_image", "header_image", "seal"):
            elements.append(f"[{i}] [{label}] [图片]")

    if len(elements) < 2:
        return boxes

    prompt = f"""你是一个文档阅读顺序分析专家。以下是文档中各段落/元素的OCR识别结果（已按版面分析的空间位置排序），但空间顺序不一定符合语义阅读顺序。

请仔细阅读每个元素的内容，分析它们之间的语义连贯性。
- 有时候一句话会被版面分析切成多块，比如"如图所示"在上一页末尾，"详细内容在下一页"在下一页开头，这两块应该在一起
- 有时候正文会被标题/注释打断，需要识别并恢复语义顺序
- 图片、表格通常跟随对它的引用说明

请判断当前顺序是否需要调整：
1. 如果顺序已经通顺，无需调整，输出: OK
2. 如果需要调整，请分析哪几个元素需要交换位置，然后输出调整后的完整顺序（只输出索引号，用逗号分隔，如: 0,3,1,2,4）

注意：
- 尽量保持最小调整，只移动真正打乱语义的元素
- 不要仅因为坐标位置就调整顺序，要看内容是否真的应该连在一起
- 输出必须是纯索引号序列，不要任何解释

元素列表：
{chr(10).join(elements)}

判断结果："""

    print(f"\n========== 重排序提示词 ==========\n{prompt}\n===================================\n")
    try:
        content = _chat(prompt, model=use_model).strip()
        print(f"========== 重排序结果 ==========\n{content}\n=================================\n")
        if content.upper() == "OK" or content == "顺序正确，无需调整":
            return boxes

        match = re.search(r"([\d,\s]+)", content)
        if not match:
            print(f"重排序无法解析: {content}")
            return boxes

        indices = [int(x.strip()) for x in match.group(1).split(",") if x.strip().isdigit()]
        if len(indices) != len(boxes) or sorted(indices) != list(range(len(boxes))):
            print(f"重排序索引无效: {indices}")
            return boxes
        return [boxes[i] for i in indices]
    except Exception as exc:
        print(f"重排序失败: {exc}")
        return boxes
