import chardet


def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        raw_data = f.read(10000)  # 读取前10000字节用于检测
    result = chardet.detect(raw_data)
    return result["encoding"]


file_path = r"C:\Users\Jason Guo\Desktop\pest_text\duplicates\03_mianlingchong\cuts\PD16-MW-00300225.json"  # 替换成实际的文件路径
encoding = detect_encoding(file_path)
print(f"检测到的编码: {encoding}")

import json


def json_to_yolo(json_path, output_path):
    # 读取JSON文件
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 获取图像的宽度和高度
    image_width = data["imageWidth"]
    image_height = data["imageHeight"]

    # 打开输出文件
    with open(output_path, "w") as f:
        # 遍历所有的标注形状
        for shape in data["shapes"]:
            # 获取类别标签
            label = shape["label"]
            # 获取标注的点
            points = shape["points"]
            # 计算矩形的左上角和右下角坐标
            x1, y1 = points[0]
            x2, y2 = points[1]

            # 计算YOLO格式的中心点坐标和宽度、高度
            x_center = (x1 + x2) / (2 * image_width)
            y_center = (y1 + y2) / (2 * image_height)
            width = (x2 - x1) / image_width
            height = (y2 - y1) / image_height

            # 将标签和坐标信息写入输出文件
            f.write(f"{label} {x_center} {y_center} {width} {height}\n")


import os


json_path = r"C:\Users\35088\Desktop\25.7.24\pest_text\duplicates\02_nianchong\cut"

for filename in os.listdir(json_path):
    # 示例用法
    if filename.endswith("json"):
        input_path = os.path.join(json_path, filename)
        output_path = os.path.join(json_path, filename.split(".")[0] + ".txt")
        json_to_yolo(input_path, output_path)
