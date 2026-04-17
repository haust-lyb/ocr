from paddlex import create_model
import cv2
import numpy as np

# 全局加载（避免每次请求都初始化）
model = create_model("PP-DocLayoutV3")


def draw_boxes_on_image(image, boxes, save_path):
    """在图片上绘制检测框"""
    img_draw = image.copy()
    color_map = {
        'text': (255, 107, 107),
        'image': (78, 205, 196),
        'header': (255, 230, 109),
        'table': (137, 221, 255),
        'chart': (255, 159, 67),
        'figure_title': (84, 160, 255),
        'doc_title': (0, 212, 255),
        'paragraph_title': (85, 163, 255),
        'content': (255, 159, 243),
        'abstract': (10, 189, 227),
        'footer': (95, 39, 205),
        'footnote': (162, 155, 254),
        'reference': (178, 190, 195),
        'formula_number': (255, 107, 107),
        'display_formula': (238, 90, 36),
        'inline_formula': (253, 121, 168),
        'algorithm': (16, 172, 132),
        'aside_text': (254, 202, 87),
        'vertical_text': (225, 112, 85),
        'seal': (214, 48, 49),
        'footer_image': (1, 163, 164),
        'header_image': (0, 210, 211),
        'reference_content': (99, 110, 114),
        'vision_footnote': (116, 185, 255),
        'default': (130, 170, 255)
    }

    for box in boxes:
        label = box.get('label', 'default')
        coord = box.get('coordinate', [])
        if len(coord) >= 4:
            x1, y1, x2, y2 = map(int, coord)
            color = color_map.get(label, color_map['default'])
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)

    cv2.imwrite(save_path, img_draw)


def run_layout(image, saveAnnotatedPath: str = None):
    """
    image: numpy array (cv2 format)
    saveAnnotatedPath: 可选，保存带标注的图片路径
    """
    results = model.predict(image, threshold=0.45)
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
    if saveAnnotatedPath:
        if result_obj and hasattr(result_obj, 'save_to_img'):
            try:
                result_obj.save_to_img(saveAnnotatedPath)
            except Exception as e:
                print(f"save_to_img failed: {e}")
                # fallback: 使用 draw_boxes_on_image
                if output and "res" in output[0] and "boxes" in output[0]["res"]:
                    draw_boxes_on_image(image, output[0]["res"]["boxes"], saveAnnotatedPath)
        elif output and "res" in output[0] and "boxes" in output[0]["res"]:
            # 如果没有 save_to_img，直接用我们自己的绘制函数
            draw_boxes_on_image(image, output[0]["res"]["boxes"], saveAnnotatedPath)

    return output
