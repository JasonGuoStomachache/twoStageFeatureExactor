import cv2
import numpy as np
import os
from typing import List, Tuple, Optional


def draw_yolo_annotations(
    img_path: str,
    txt_path: str,
    output_path: str,
    class_colors: Optional[List[Tuple[int, int, int]]] = None,
    line_thickness: int = 2,
    font_scale: float = 0.5,
    font_thickness: int = 1,
) -> None:
    """
    从YOLO格式的txt标注文件中读取标注信息，在图片上绘制标注框（带序号）

    参数:
        img_path: 图片文件路径
        txt_path: YOLO格式标注文件路径
        output_path: 输出图片路径
        class_colors: 每个类别的颜色列表，格式为[(B, G, R), ...]，长度需≥类别数
                     若为None，将自动生成随机颜色
        line_thickness: 标注框线条粗细
        font_scale: 序号字体大小缩放比例
        font_thickness: 序号字体粗细
    """
    # 1. 读取图片
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {img_path}")
    img_h, img_w = img.shape[:2]

    # 2. 读取YOLO标注文件
    annotations = []
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                # YOLO格式：class_id x_center y_center width height（均为归一化值）
                parts = list(map(float, line.split()))
                if len(parts) != 5:
                    print(f"警告：标注文件第{idx+1}行格式错误，跳过")
                    continue
                class_id, x_center, y_center, width, height = parts
                annotations.append(
                    (int(class_id), x_center, y_center, width, height, idx + 1)
                )  # 加入序号
    else:
        print(f"警告：标注文件不存在: {txt_path}，仅保存原图")
        cv2.imwrite(output_path, img)
        return

    # 3. 处理类别颜色
    if class_colors is None:
        # 自动生成随机颜色（确保不同类别颜色不同）
        max_class_id = max([ann[0] for ann in annotations], default=0)
        class_colors = []
        np.random.seed(42)  # 固定随机种子，保证颜色一致性
        for _ in range(max_class_id + 1):
            color = tuple(np.random.randint(0, 256, 3).tolist())
            class_colors.append(color)
    else:
        # 检查颜色列表是否足够
        max_class_id = max([ann[0] for ann in annotations], default=0)
        if len(class_colors) <= max_class_id:
            raise ValueError(
                f"类别颜色列表长度不足（需要≥{max_class_id+1}，当前为{len(class_colors)}）"
            )

    # 4. 绘制标注框和序号
    for ann in annotations:
        class_id, x_center, y_center, width, height, seq_num = ann

        # 转换YOLO归一化坐标到像素坐标
        x1 = int((x_center - width / 2) * img_w)
        y1 = int((y_center - height / 2) * img_h)
        x2 = int((x_center + width / 2) * img_w)
        y2 = int((y_center + height / 2) * img_h)

        # 确保坐标在图片范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w - 1, x2)
        y2 = min(img_h - 1, y2)

        # 获取当前类别的颜色
        color = class_colors[class_id]

        # 绘制矩形框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)

        # 绘制序号（左上角）
        # 计算文本背景大小
        text = str(seq_num)
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        # 文本背景框（避免文字与边框重叠）
        bg_x1 = x1
        bg_y1 = y1 - 25 - baseline
        bg_x2 = x1 + 25
        bg_y2 = y1
        # 确保背景框在图片内
        bg_y1 = max(0, bg_y1)
        # 绘制背景框（与标注框同色）
        cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)  # -1表示填充
        # 绘制文字（白色文字，对比明显）
        cv2.putText(
            img,
            text,
            (x1, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        # print(min(x2-x1, y2-y1))
        # print(max(x2-x1, y2-y1))
        # print(min(x2-x1, y2-y1)/ max(x2-x1, y2-y1))
        print(x_center)
        print(1 - y_center)

    # 5. 保存结果图片
    cv2.imwrite(output_path, img)
    print(f"标注可视化完成，保存至: {output_path}")


def batch_draw_annotations(
    img_dir: str,
    txt_dir: str,
    output_dir: str,
    class_colors: Optional[List[Tuple[int, int, int]]] = None,
    line_thickness: int = 2,
    font_scale: float = 0.5,
    font_thickness: int = 1,
) -> None:
    """
    批量处理文件夹中的图片和标注文件

    参数:
        img_dir: 图片文件夹路径
        txt_dir: 标注文件文件夹路径（与图片文件名一一对应）
        output_dir: 输出文件夹路径
        其他参数: 同draw_yolo_annotations
    """
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有图片文件
    img_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif")
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(img_extensions)]

    for img_file in img_files:
        # 构建文件路径
        img_path = os.path.join(img_dir, img_file)
        txt_file = os.path.splitext(img_file)[0] + ".txt"
        txt_path = os.path.join(txt_dir, txt_file)
        output_path = os.path.join(output_dir, img_file)

        try:
            draw_yolo_annotations(
                img_path=img_path,
                txt_path=txt_path,
                output_path=output_path,
                class_colors=class_colors,
                line_thickness=line_thickness,
                font_scale=font_scale,
                font_thickness=font_thickness,
            )
        except Exception as e:
            print(f"处理 {img_file} 时出错: {str(e)}")


# ------------------------------
# 示例使用
# ------------------------------
if __name__ == "__main__":
    # 方式1：处理单张图片
    single_img_path = r"D:\25.10.29backup\25.7.24\pest_text\api\data_processed\images\03_fall_armyworm\PD16-MW-00300001.jpg"  # 输入图片路径
    single_txt_path = r"D:\25.10.29backup\25.7.24\pest_text\api\data_processed\bbox\03_fall_armyworm\PD16-MW-00300001.txt"  # 对应的YOLO标注文件路径
    single_output_path = "test_annotated.jpg"  # 输出图片路径

    # 自定义类别颜色（格式：BGR，与OpenCV一致）
    custom_colors = [
        (200, 180, 255),
        (23, 164, 243),
        (70, 83, 238),
        (150, 247, 255),
        (188, 115, 141),
        (212, 221, 164),
        (55, 171, 83),
        (197, 173, 50),
    ]

    try:
        draw_yolo_annotations(
            img_path=single_img_path,
            txt_path=single_txt_path,
            output_path=single_output_path,
            class_colors=custom_colors,  # 若不指定，将自动生成
            line_thickness=2,
            font_scale=0.6,
            font_thickness=1,
        )
    except Exception as e:
        print(f"单张图片处理失败: {str(e)}")

    # 方式2：批量处理文件夹（可选）
    # batch_img_dir = "images"      # 图片文件夹
    # batch_txt_dir = "labels"      # 标注文件文件夹
    # batch_output_dir = "annotated_images"  # 输出文件夹
    #
    # try:
    #     batch_draw_annotations(
    #         img_dir=batch_img_dir,
    #         txt_dir=batch_txt_dir,
    #         output_dir=batch_output_dir,
    #         class_colors=custom_colors,
    #         line_thickness=2,
    #         font_scale=0.6,
    #         font_thickness=1
    #     )
    # except Exception as e:
    #     print(f"批量处理失败: {str(e)}")
