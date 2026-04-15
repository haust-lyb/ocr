from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import cv2
import os
import uuid

from model import run_layout
from ollama_client import call_glm_ocr, reorder_layout_elements, OLLAMA_REORDER_MODEL
from storage import save_recognition, get_recognition, list_recognitions, delete_recognition

app = FastAPI()

# 路径配置
BASE_DIR = os.path.dirname(__file__)
TARGET_DIR = os.path.join(BASE_DIR, "target")
IMAGES_DIR = os.path.join(TARGET_DIR, "images")
VIEW_DIR = os.path.join(BASE_DIR, "view")

os.makedirs(TARGET_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# 静态文件（前端页面）
app.mount("/static", StaticFiles(directory=VIEW_DIR), name="static")

# OCR 并发线程池
executor = ThreadPoolExecutor(max_workers=4)


@app.get("/")
async def index():
    """首页"""
    return FileResponse(os.path.join(VIEW_DIR, "index.html"))


@app.get("/docs")
async def docs():
    """API 文档"""
    return FileResponse(os.path.join(VIEW_DIR, "docs.html"))


@app.get("/console")
async def console():
    """控制台"""
    return FileResponse(os.path.join(VIEW_DIR, "console.html"))


@app.get("/viewer")
async def viewer():
    """可视化查看器"""
    return FileResponse(os.path.join(VIEW_DIR, "viewer.html"))


# 需要 OCR 的文本类型
OCR_TYPES = {
    "text", "abstract", "aside_text", "content", "doc_title", "figure_title",
    "footer", "footnote", "formula_number", "header", "inline_formula",
    "number", "paragraph_title", "reference", "reference_content",
    "table", "vertical_text", "vision_footnote", "display_formula",
    "algorithm"
}

# 需要提取图片的类型
IMAGE_TYPES = {
    "image", "chart", "footer_image", "header_image", "seal"
}


def build_markdown_from_layout(boxes):
    """
    根据版面分析结果和 OCR 文本构建 Markdown
    boxes: 版面分析的区域列表（已包含 extracted_image 和 ocr_text）
    """
    md_lines = []

    for i, box in enumerate(boxes):
        label = box.get("label", "")
        ocr_text = box.get("ocr_text", "").strip()

        # 跳过没有 OCR 文本且不是图片的区域
        if not ocr_text and label not in IMAGE_TYPES:
            continue

        # 根据类型生成 Markdown
        if label == "doc_title":
            md_lines.append(f"# {ocr_text}\n")
        elif label == "header":
            md_lines.append(f"## {ocr_text}\n")
        elif label == "paragraph_title":
            md_lines.append(f"### {ocr_text}\n")
        elif label == "figure_title":
            md_lines.append(f"**图注**: {ocr_text}\n")
        elif label == "table":
            # 表格处理：如果 OCR 文本是 HTML 表格，尝试转换
            if ocr_text.startswith("<table"):
                md_lines.append(f"\n{table_to_markdown(ocr_text)}\n")
            else:
                md_lines.append(f"\n{ocr_text}\n")
        elif label in IMAGE_TYPES:
            # 直接使用已提取的图片路径
            img_filename = box.get("extracted_image")
            if img_filename:
                md_lines.append(f"\n![{label}]({img_filename})\n")
        elif label == "display_formula":
            md_lines.append(f"\n$$\n{ocr_text}\n$$\n")
        elif label == "inline_formula":
            md_lines.append(f"${ocr_text}$")
        elif label == "algorithm":
            md_lines.append(f"\n```\n{ocr_text}\n```\n")
        elif label in ("footer", "footnote", "reference", "reference_content", "vision_footnote"):
            # 脚注/参考文献类
            md_lines.append(f"\n*{ocr_text}*\n")
        elif label in OCR_TYPES:
            # 普通文本段落
            if ocr_text:
                md_lines.append(f"\n{ocr_text}\n")

    return "\n".join(md_lines)


def table_to_markdown(html_table):
    """简单的 HTML 表格转 Markdown 表格"""
    import re
    lines = []

    # 提取表头
    headers = re.findall(r'<th[^>]*>(.*?)</th>', html_table, re.DOTALL)
    if headers:
        # 清理 HTML 标签
        headers = [re.sub(r'<[^>]+>', '', h).strip() for h in headers]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    # 提取表格行
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html_table, re.DOTALL)
    for row in rows:
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
        if cells:
            cells = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
            lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    """版面识别 + OCR，返回识别 ID"""
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    image_name = file.filename or "unknown"
    rec_id = str(uuid.uuid4())[:8]
    image_filename = f"{rec_id}_{image_name}"
    image_path = os.path.join(IMAGES_DIR, image_filename)
    cv2.imwrite(image_path, img)

    # 带标注的图片路径
    annotated_filename = f"{rec_id}_annotated.png"
    annotated_path = os.path.join(IMAGES_DIR, annotated_filename)

    status = "success"
    result = {}

    try:
        # 版面识别，并保存带标注的图片
        result = run_layout(img, annotated_path)

        # 调试：打印所有检测到的区域及其分数
        if result and "res" in result[0] and "boxes" in result[0]["res"]:
            boxes = result[0]["res"]["boxes"]
            print(f"检测到 {len(boxes)} 个区域:")
            for box in boxes:
                coord = box.get('coordinate', [])
                score = box.get('score', 'N/A')
                print(f"  {box.get('label')}: score={score}, coord={coord}")
            boxes = result[0]["res"]["boxes"]

            # 并发 OCR 调用
            def ocr_box(box, cropped):
                try:
                    ocr_text = call_glm_ocr(cropped)
                    return box, ocr_text, None
                except Exception as e:
                    return box, None, str(e)

            # 提交所有 OCR 任务
            futures = {}
            for box in boxes:
                label = box.get("label")
                if label in OCR_TYPES:
                    x1, y1, x2, y2 = box["coordinate"]
                    cropped = img[y1:y2, x1:x2]
                    future = executor.submit(ocr_box, box, cropped)
                    futures[future] = box

            # 收集结果
            for future in as_completed(futures):
                box, ocr_text, error = future.result()
                if error:
                    box["ocr_error"] = error
                    status = "partial"
                else:
                    box["ocr_text"] = ocr_text

            # 重排序（如果配置了重排序模型）
            if OLLAMA_REORDER_MODEL:
                print(f"使用重排序模型: {OLLAMA_REORDER_MODEL}")
                boxes = reorder_layout_elements(boxes)

            # 提取图片（串行）
            img_index = 0
            for box in boxes:
                label = box.get("label")
                if label in IMAGE_TYPES:
                    try:
                        x1, y1, x2, y2 = box["coordinate"]
                        cropped = img[y1:y2, x1:x2]
                        cropped_filename = f"{rec_id}_{label}_{img_index}.png"
                        cropped_path = os.path.join(IMAGES_DIR, cropped_filename)
                        cv2.imwrite(cropped_path, cropped)
                        box["extracted_image"] = cropped_filename
                        img_index += 1
                    except Exception as e:
                        box["extract_error"] = str(e)

    except Exception as e:
        status = "error"
        result = {"error": str(e)}

    save_recognition(rec_id, image_name, image_filename, annotated_filename, result, status, rec_type="ocr")

    return {
        "id": rec_id,
        "image": image_filename,
        "status": status,
        "results": result if "error" not in result else None
    }


@app.post("/markdown")
async def markdown(file: UploadFile = File(...)):
    """图片识别为 Markdown 格式（基于版面分析结果构建）"""
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    image_name = file.filename or "unknown"
    rec_id = str(uuid.uuid4())[:8]
    image_filename = f"{rec_id}_{image_name}"
    image_path = os.path.join(IMAGES_DIR, image_filename)
    cv2.imwrite(image_path, img)

    # 带标注的图片路径
    annotated_filename = f"{rec_id}_annotated.png"
    annotated_path = os.path.join(IMAGES_DIR, annotated_filename)

    status = "success"
    result = {}

    try:
        # 版面分析，并保存带标注的图片
        layout_result = run_layout(img, annotated_path)

        if layout_result and "res" in layout_result[0] and "boxes" in layout_result[0]["res"]:
            boxes = layout_result[0]["res"]["boxes"]

            # 并发 OCR 调用
            def ocr_box(box, cropped):
                try:
                    ocr_text = call_glm_ocr(cropped)
                    return box, ocr_text, None
                except Exception as e:
                    return box, None, str(e)

            # 提交所有 OCR 任务
            futures = {}
            for box in boxes:
                label = box.get("label")
                if label in OCR_TYPES:
                    x1, y1, x2, y2 = box["coordinate"]
                    cropped = img[y1:y2, x1:x2]
                    future = executor.submit(ocr_box, box, cropped)
                    futures[future] = box

            # 收集 OCR 结果
            for future in as_completed(futures):
                box, ocr_text, error = future.result()
                if error:
                    box["ocr_error"] = error
                else:
                    box["ocr_text"] = ocr_text

            # 重排序（如果配置了重排序模型）
            if OLLAMA_REORDER_MODEL:
                print(f"使用重排序模型: {OLLAMA_REORDER_MODEL}")
                boxes = reorder_layout_elements(boxes)

            # 提取图片（用于 Markdown 图片引用）
            img_index = 0
            for box in boxes:
                label = box.get("label")
                if label in IMAGE_TYPES:
                    try:
                        x1, y1, x2, y2 = box["coordinate"]
                        cropped = img[y1:y2, x1:x2]
                        cropped_filename = f"{rec_id}_img_{img_index}.png"
                        cropped_path = os.path.join(IMAGES_DIR, cropped_filename)
                        cv2.imwrite(cropped_path, cropped)
                        box["extracted_image"] = cropped_filename
                        img_index += 1
                    except Exception as e:
                        box["extract_error"] = str(e)

            # 根据版面分析结果构建 Markdown
            md_text = build_markdown_from_layout(boxes)
            result = {
                "markdown": md_text,
                "boxes": boxes  # 同时保存版面分析结果
            }
        else:
            status = "error"
            result = {"error": "版面分析失败"}

    except Exception as e:
        status = "error"
        result = {"error": str(e)}

    save_recognition(rec_id, image_name, image_filename, annotated_filename, result, status, rec_type="markdown")

    return {
        "id": rec_id,
        "image": image_filename,
        "status": status,
        "markdown": result.get("markdown")
    }


@app.get("/view/{rec_id}")
async def view_recognition(rec_id: str):
    """查看识别结果"""
    data = get_recognition(rec_id)
    if not data:
        raise HTTPException(status_code=404, detail="识别结果不存在")
    # 统一使用 viewer.html，支持 OCR 和 Markdown 两种类型
    data["view_url"] = "/viewer?id=" + rec_id
    return data


@app.get("/list")
async def list_records():
    """列出识别记录"""
    return {"records": list_recognitions()}


@app.delete("/record/{rec_id}")
async def delete_record(rec_id: str):
    """删除识别记录（软删除）"""
    delete_recognition(rec_id)
    return {"success": True}


@app.get("/image/{filename}")
async def get_image(filename: str):
    """获取图片"""
    image_path = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="图片不存在")
    return FileResponse(image_path)
