from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import os
import uuid

from model import run_layout
from ollama_client import call_glm_ocr
from storage import save_recognition, get_recognition, list_recognitions

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


@app.post("/layout")
async def layout(file: UploadFile = File(...)):
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
            img_index = 0
            for box in boxes:
                label = box.get("label")
                x1, y1, x2, y2 = box["coordinate"]
                cropped = img[y1:y2, x1:x2]

                if label in OCR_TYPES:
                    try:
                        ocr_text = call_glm_ocr(cropped)
                        box["ocr_text"] = ocr_text
                    except Exception as e:
                        box["ocr_error"] = str(e)
                        status = "partial"

                elif label in IMAGE_TYPES:
                    try:
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

    save_recognition(rec_id, image_name, image_filename, annotated_filename, result, status)

    return {
        "id": rec_id,
        "image": image_filename,
        "status": status,
        "results": result if "error" not in result else None
    }


@app.get("/view/{rec_id}")
async def view_recognition(rec_id: str):
    """查看识别结果"""
    data = get_recognition(rec_id)
    if not data:
        raise HTTPException(status_code=404, detail="识别结果不存在")
    return data


@app.get("/list")
async def list_records():
    """列出识别记录"""
    return {"records": list_recognitions()}


@app.get("/image/{filename}")
async def get_image(filename: str):
    """获取图片"""
    image_path = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="图片不存在")
    return FileResponse(image_path)
