import os
import ollama
from PIL import Image
import io
import base64

OLLAMA_MODEL = "glm-ocr:latest"
OLLAMA_REORDER_MODEL = ""  # 重排序模型，如不需要请设为空字符串 ""
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


def call_glm_markdown(image, host: str = None) -> str:
    """
    调用 Ollama glm-ocr 模型，将图片识别为 Markdown 格式输出

    image: numpy array (cv2 format, BGR)
    host: Ollama 服务地址，默认从环境变量 OLLAMA_HOST 读取
    """
    prompt = "请将图片中的内容识别为 Markdown 格式输出，保持原有的标题、列表、表格、代码块等结构，只输出 Markdown 原文，不要额外解释。"
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


def call_glm_extract(image, prompt_json: str, host: str = None) -> str:
    """
    调用 Ollama glm-ocr 模型进行结构化提取

    image: numpy array (cv2 format, BGR)
    prompt_json: 用户提供的 JSON 格式提示词（字段描述 / schema）
    返回: 模型原始输出字符串
    """
    prompt = (
        "请根据以下 JSON 格式要求，从图片中提取对应字段信息，"
        "严格按照 JSON 格式输出，不要多余解释。\n\n"
        f"要求格式：\n{prompt_json}"
    )
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


def reorder_layout_elements(boxes: list, model: str = None) -> list:
    """
    调用 LLM 根据 OCR 结果的语义连贯性，重新调整元素顺序

    boxes: 版面分析结果列表，每个元素包含 label, ocr_text, coordinate 等字段
    model: 使用的模型名称，默认使用 OLLAMA_REORDER_MODEL
    返回: 重新排序后的 boxes 列表
    """
    if not OLLAMA_REORDER_MODEL and not model:
        return boxes

    use_model = model or OLLAMA_REORDER_MODEL
    if not use_model:
        return boxes

    # 构建输入内容，只用文本信息
    elements = []
    for i, box in enumerate(boxes):
        label = box.get("label", "unknown")
        ocr_text = box.get("ocr_text", "").strip()
        if ocr_text:
            # 截断过长的文本
            text = ocr_text[:200].replace('\n', ' ')
            elements.append(f"[{i}] [{label}] {text}")
        elif label in ("image", "chart", "footer_image", "header_image", "seal"):
            elements.append(f"[{i}] [{label}] [图片]")

    if len(elements) < 2:
        return boxes  # 少于2个元素无需排序

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

    client = ollama.Client(host=OLLAMA_HOST)
    try:
        response = client.chat(
            model=use_model,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        content = response.message.content.strip()
        print(f"========== 重排序结果 ==========\n{content}\n=================================\n")

        # 如果返回 OK，说明顺序正确，无需调整
        if content.upper() == "OK" or content == "顺序正确，无需调整":
            return boxes

        # 解析 LLM 返回的索引号
        import re
        match = re.search(r'([\d,\s]+)', content)
        if not match:
            print(f"重排序无法解析: {content}")
            return boxes

        indices = [int(x.strip()) for x in match.group(1).split(',') if x.strip().isdigit()]

        # 检查索引是否有效
        if len(indices) != len(boxes):
            print(f"重排序索引数量不匹配: {len(indices)} vs {len(boxes)}")
            return boxes

        # 重新排序
        reordered = [boxes[i] for i in indices]
        return reordered

    except Exception as e:
        print(f"重排序失败: {e}")
        return boxes


