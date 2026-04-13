from paddlex import create_model

# 全局加载（避免每次请求都初始化）
model = create_model("PP-DocLayoutV3")


def run_layout(image):
    """
    image: numpy array (cv2 format)
    """
    results = model.predict(image)

    output = []

    for r in results:
        # PaddleX result 通常有这些字段
        if hasattr(r, "json"):
            output.append(r.json)
        elif hasattr(r, "to_dict"):
            output.append(r.to_dict())
        else:
            # fallback
            output.append(str(r))

    return output
