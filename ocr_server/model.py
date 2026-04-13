from paddlex import create_model

# 全局加载（避免每次请求都初始化）
model = create_model("PP-DocLayoutV3")


def run_layout(image, saveAnnotatedPath: str = None):
    """
    image: numpy array (cv2 format)
    saveAnnotatedPath: 可选，保存带标注的图片路径
    """
    results = model.predict(image)

    output = []
    result_obj = None

    for r in results:
        result_obj = r
        # PaddleX result 通常有这些字段
        if hasattr(r, "json"):
            output.append(r.json)
        elif hasattr(r, "to_dict"):
            output.append(r.to_dict())
        else:
            # fallback
            output.append(str(r))

    # 保存带标注的图片
    if saveAnnotatedPath and result_obj:
        result_obj.save_to_img(saveAnnotatedPath)

    return output
