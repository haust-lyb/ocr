from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2

from model import run_layout

app = FastAPI()


@app.post("/layout")
async def layout(file: UploadFile = File(...)):
    # 读取图片
    image_bytes = await file.read()

    # bytes → numpy
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 调用模型
    result = run_layout(img)

    return {
        "results": result
    }
