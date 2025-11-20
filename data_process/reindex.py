# -*- coding: utf-8 -*-
"""
YOLO标注文件+图片+描述文件批量处理工具
功能：
1. 按规则修改YOLO标注文件类别编号
2. 同步修改标注文件、图片文件、描述文件的文件名（更新类别编码）
3. 修改描述文件（JSON格式）内部的Image filename字段
4. 自动复制处理后的文件到输出目录（不覆盖原文件）
"""

# 导入必要模块
import os
import shutil
import re
import json

# ===================== 配置参数（请根据实际情况修改）=====================
# 原文件路径
ANNOTATIONS_INPUT_DIR = r"D:\25.10.29backup\25.7.24\pest_text\api\data\bbox\08_wheat_midge"  # 原标注文件目录
IMAGES_INPUT_DIR = r"D:\25.10.29backup\25.7.24\pest_text\api\data\images\08_wheat_midge"  # 原图片文件目录
CAPTION_CN_INPUT_DIR = r"D:\25.10.29backup\25.7.24\pest_text\api\data\caption\08_wheat_midge"  # 中文描述文件目录
CAPTION_EN_INPUT_DIR = r"D:\25.10.29backup\25.7.24\pest_text\api\data\caption_en\08_wheat_midge"  # 英文描述文件目录

# 输出文件路径（自动创建，避免覆盖原文件）
ANNOTATIONS_OUTPUT_DIR = r"D:\25.10.29backup\25.7.24\pest_text\api\data_processed\bbox_processed\07_wheat_midge"
IMAGES_OUTPUT_DIR = r"D:\25.10.29backup\25.7.24\pest_text\api\data_processed\images_processed\07_wheat_midge"
CAPTION_CN_OUTPUT_DIR = r"D:\25.10.29backup\25.7.24\pest_text\api\data_processed\caption_processed\07_wheat_midge"
CAPTION_EN_OUTPUT_DIR = r"D:\25.10.29backup\25.7.24\pest_text\api\data_processed\caption_en_processed\07_wheat_midge"

# 类别映射规则（key=原类别，value=新类别）
CLASS_MAPPING = {1: 3, 2: 0, 3: 2, 4: 1, 5: 4, 6: 5, 7: 6, 8: 7}

# 文件名配置（根据你的文件名格式调整）
FILE_PREFIX = "PD16-MW-"  # 文件名前缀
NUM_DIGITS_TOTAL = 8  # 文件名中数字部分总位数
CLASS_CODE_DIGITS = 3  # 数字部分中类别编码的位数（前N位）
# 文件后缀配置
IMAGE_SUFFIX = ".jpg"  # 图片文件后缀（原文件）
ANNOTATION_SUFFIX = ".txt"  # 标注文件后缀
CAPTION_CN_SUFFIX = "_caption.txt"  # 中文描述文件后缀
CAPTION_EN_SUFFIX = "_caption_en.txt"  # 英文描述文件后缀
IMAGE_JSON_SUFFIX = ".jpg"  # JSON中Image filename的后缀（示例中是.jpg）
# ========================================================================


# 创建输出目录（如果不存在）
def create_output_dirs():
    """创建所有输出目录"""
    dirs = [
        ANNOTATIONS_OUTPUT_DIR,
        IMAGES_OUTPUT_DIR,
        CAPTION_CN_OUTPUT_DIR,
        CAPTION_EN_OUTPUT_DIR,
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

    print(f"输出目录已创建/确认：")
    for dir_path in dirs:
        print(f"- {dir_path}")


# 解析基础文件名（提取类别编码和序号，支持所有文件类型）
def parse_base_filename(filename):
    """
    从任意类型文件中提取基础信息
    支持的文件名格式：
    - 标注文件：PD16-MW-XXXXXXXX.txt
    - 图片文件：PD16-MW-XXXXXXXX.png
    - 中文描述：PD16-MW-XXXXXXXX_caption.txt
    - 英文描述：PD16-MW-XXXXXXXX_caption_en.txt
    输出：(原类别编码int, 序号str, 文件类型标识str)
    文件类型标识：annotation/image/caption_cn/caption_en
    """
    # 定义所有支持的文件名模式
    patterns = [
        # (正则表达式, 文件类型标识)
        (
            rf"^{FILE_PREFIX}(\d{{{NUM_DIGITS_TOTAL}}}){ANNOTATION_SUFFIX}$",
            "annotation",
        ),
        (rf"^{FILE_PREFIX}(\d{{{NUM_DIGITS_TOTAL}}}){IMAGE_SUFFIX}$", "image"),
        (
            rf"^{FILE_PREFIX}(\d{{{NUM_DIGITS_TOTAL}}}){CAPTION_CN_SUFFIX}$",
            "caption_cn",
        ),
        (
            rf"^{FILE_PREFIX}(\d{{{NUM_DIGITS_TOTAL}}}){CAPTION_EN_SUFFIX}$",
            "caption_en",
        ),
    ]

    for pattern, file_type in patterns:
        match = re.match(pattern, filename)
        if match:
            num_str = match.group(1)  # 8位数字部分（如00200001）
            # 提取类别编码（前CLASS_CODE_DIGITS位）和序号（后几位）
            class_code_str = num_str[:CLASS_CODE_DIGITS]
            sequence_str = num_str[CLASS_CODE_DIGITS:]

            try:
                original_class_code = int(class_code_str)
            except ValueError:
                raise ValueError(
                    f"文件名中的类别编码不是数字：{class_code_str}（文件名：{filename}）"
                )

            return original_class_code, sequence_str, file_type

    # 未匹配到任何支持的格式
    raise ValueError(f"文件名格式不支持：{filename}")


# 生成新的文件名（根据基础信息和文件类型）
def generate_new_filename(original_class_code, sequence_str, file_type):
    """根据原类别编码、序号和文件类型生成新文件名"""
    # 获取映射后的新类别
    if original_class_code not in CLASS_MAPPING:
        raise KeyError(
            f"原类别 {original_class_code} 没有对应的映射规则，请检查CLASS_MAPPING配置"
        )

    new_class_code = CLASS_MAPPING[original_class_code]
    # 生成新的数字部分（类别编码补零到指定位数 + 序号）
    new_class_code_str = f"{new_class_code:0{CLASS_CODE_DIGITS}d}"
    new_num_str = new_class_code_str + sequence_str

    # 根据文件类型生成对应文件名
    if file_type == "annotation":
        return f"{FILE_PREFIX}{new_num_str}{ANNOTATION_SUFFIX}"
    elif file_type == "image":
        return f"{FILE_PREFIX}{new_num_str}{IMAGE_SUFFIX}"
    elif file_type == "caption_cn":
        return f"{FILE_PREFIX}{new_num_str}{CAPTION_CN_SUFFIX}"
    elif file_type == "caption_en":
        return f"{FILE_PREFIX}{new_num_str}{CAPTION_EN_SUFFIX}"
    else:
        raise ValueError(f"不支持的文件类型：{file_type}")


# 生成JSON中Image filename的新值
def generate_new_image_json_filename(original_class_code, sequence_str):
    """生成修改后的Image filename字段值（如PD16-MW-00000001.jpg）"""
    new_class_code = CLASS_MAPPING[original_class_code]
    new_class_code_str = f"{new_class_code:0{CLASS_CODE_DIGITS}d}"
    new_num_str = new_class_code_str + sequence_str
    return f"{FILE_PREFIX}{new_num_str}{IMAGE_JSON_SUFFIX}"


# 处理单个标注文件
def process_annotation_file(annotation_path, new_annotation_path):
    """读取原标注文件，修改类别编号，写入新文件"""
    modified_lines = []
    with open(annotation_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line_idx, line in enumerate(lines, 1):
        line = line.strip()
        if not line:  # 跳过空行
            modified_lines.append("")
            continue

        # YOLO标注格式：class_id x_center y_center width height
        parts = line.split()
        if len(parts) != 5:
            print(
                f"警告：标注文件 {os.path.basename(annotation_path)} 第{line_idx}行格式错误，跳过该行：{line}"
            )
            continue

        # 解析原类别ID
        try:
            original_class_id = int(parts[0])
        except ValueError:
            print(
                f"警告：标注文件 {os.path.basename(annotation_path)} 第{line_idx}行类别ID不是数字，跳过该行：{line}"
            )
            continue

        # 映射到新类别ID
        if original_class_id in CLASS_MAPPING:
            new_class_id = CLASS_MAPPING[original_class_id]
            new_line = f"{new_class_id} {' '.join(parts[1:])}"
            modified_lines.append(new_line)
        else:
            modified_lines.append(line)
            print(
                f"警告：标注文件 {os.path.basename(annotation_path)} 第{line_idx}行出现未配置的类别ID {original_class_id}，保留原类别"
            )

    # 写入新标注文件
    with open(new_annotation_path, "w", encoding="utf-8") as f:
        f.write("\n".join(modified_lines))

    return len(modified_lines) - modified_lines.count("")  # 返回有效标注行数


# 处理单个描述文件（JSON格式）
def process_caption_file(
    caption_path, new_caption_path, original_class_code, sequence_str
):
    """
    处理描述文件：
    1. 读取JSON内容
    2. 修改Image filename字段
    3. 保存到新文件
    """
    try:
        # 读取JSON文件（支持中文编码）
        with open(caption_path, "r", encoding="utf-8") as f:
            try:
                json_data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON格式错误：{str(e)}")

        # 检查并修改Image filename字段
        if "Image filename" in json_data:
            original_image_filename = json_data["Image filename"]
            new_image_filename = generate_new_image_json_filename(
                original_class_code, sequence_str
            )
            json_data["Image filename"] = new_image_filename
            print(
                f"📝 Image filename修改：{original_image_filename} → {new_image_filename}"
            )

        elif "图片的文件名" in json_data:
            original_image_filename = json_data["图片的文件名"]
            new_image_filename = generate_new_image_json_filename(
                original_class_code, sequence_str
            )
            json_data["图片的文件名"] = new_image_filename
            print(
                f"📝 图片的文件名修改：{original_image_filename} → {new_image_filename}"
            )
        else:
            print(
                f"警告：描述文件 {os.path.basename(caption_path)} 缺少Image filename字段，跳过该字段修改"
            )

        # 写入修改后的JSON文件（保持缩进格式）
        with open(new_caption_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

        return True

    except Exception as e:
        raise Exception(f"JSON处理失败：{str(e)}")


# 处理单个文件对（标注+图片，可选关联描述文件）
def process_annotation_image_pair(annotation_filename):
    """处理标注文件和对应的图片文件"""
    annotation_basename = annotation_filename
    annotation_path = os.path.join(ANNOTATIONS_INPUT_DIR, annotation_basename)

    try:
        # 解析原文件名
        original_class_code, sequence_str, _ = parse_base_filename(annotation_basename)

        # 1. 处理标注文件
        new_annotation_basename = generate_new_filename(
            original_class_code, sequence_str, "annotation"
        )
        new_annotation_path = os.path.join(
            ANNOTATIONS_OUTPUT_DIR, new_annotation_basename
        )
        valid_lines = process_annotation_file(annotation_path, new_annotation_path)
        print(
            f"✅ 标注文件：{annotation_basename} → {new_annotation_basename}（有效行：{valid_lines}）"
        )

        # 2. 处理图片文件
        image_basename = annotation_filename.replace(ANNOTATION_SUFFIX, IMAGE_SUFFIX)
        print(image_basename)
        image_path = os.path.join(IMAGES_INPUT_DIR, image_basename)
        if os.path.exists(image_path):
            new_image_basename = generate_new_filename(
                original_class_code, sequence_str, "image"
            )
            new_image_path = os.path.join(IMAGES_OUTPUT_DIR, new_image_basename)
            shutil.copy2(image_path, new_image_path)
            print(f"✅ 图片文件：{image_basename} → {new_image_basename}")
        else:
            print(f"⚠️  图片文件不存在：{image_path}，跳过")

        return True

    except Exception as e:
        print(f"❌ 处理失败 {annotation_basename}：{str(e)}")
        return False


# 批量处理描述文件（中文/英文）
def process_all_caption_files():
    """批量处理中文和英文描述文件"""
    print("\n" + "=" * 50)
    print("开始处理描述文件...")
    print("=" * 50)

    # 定义需要处理的描述文件目录和类型
    caption_configs = [
        (CAPTION_CN_INPUT_DIR, CAPTION_CN_OUTPUT_DIR, "caption_cn", "中文"),
        (CAPTION_EN_INPUT_DIR, CAPTION_EN_OUTPUT_DIR, "caption_en", "英文"),
    ]

    total_success = 0
    total_fail = 0

    for input_dir, output_dir, file_type, lang_name in caption_configs:
        print(f"\n【{lang_name}描述文件】")
        print(f"处理目录：{input_dir}")

        # 筛选符合条件的文件（以对应的caption后缀结尾）
        suffix = CAPTION_CN_SUFFIX if file_type == "caption_cn" else CAPTION_EN_SUFFIX
        caption_filenames = [f for f in os.listdir(input_dir) if f.endswith(suffix)]

        if not caption_filenames:
            print(f"⚠️  未找到符合条件的{lang_name}描述文件（需以{suffix}结尾）")
            continue

        print(f"找到 {len(caption_filenames)} 个{lang_name}描述文件，开始处理...")

        success_count = 0
        fail_count = 0

        for filename in caption_filenames:
            try:
                # 解析文件名
                original_class_code, sequence_str, parsed_type = parse_base_filename(
                    filename
                )
                if parsed_type != file_type:
                    print(f"⚠️  文件名格式不匹配{lang_name}描述文件：{filename}，跳过")
                    continue

                # 生成新文件名
                new_filename = generate_new_filename(
                    original_class_code, sequence_str, file_type
                )
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, new_filename)

                # 处理文件内容并保存
                if process_caption_file(
                    input_path, output_path, original_class_code, sequence_str
                ):
                    print(f"✅ {filename} → {new_filename}")
                    success_count += 1
                else:
                    fail_count += 1

            except Exception as e:
                print(f"❌ 处理失败 {filename}：{str(e)}")
                fail_count += 1

        # 统计当前语言的处理结果
        print(f"{lang_name}描述文件处理完成：成功{success_count}个，失败{fail_count}个")
        total_success += success_count
        total_fail += fail_count

    return total_success, total_fail


# 主处理函数
def main():
    print("=" * 60)
    print("YOLO标注+图片+描述文件批量处理工具")
    print("=" * 60)

    # 1. 创建输出目录
    create_output_dirs()

    # 2. 处理标注文件和图片文件
    print("\n" + "=" * 50)
    print("开始处理标注文件和图片文件...")
    print("=" * 50)

    # 获取所有符合条件的标注文件
    annotation_filenames = [
        f
        for f in os.listdir(ANNOTATIONS_INPUT_DIR)
        if f.endswith(ANNOTATION_SUFFIX)
        and re.match(rf"^{FILE_PREFIX}\d{{{NUM_DIGITS_TOTAL}}}{ANNOTATION_SUFFIX}$", f)
    ]

    if not annotation_filenames:
        print(f"⚠️  在标注目录 {ANNOTATIONS_INPUT_DIR} 中未找到符合格式的标注文件")
    else:
        print(f"找到 {len(annotation_filenames)} 个标注文件，开始处理...\n")

        ann_success = 0
        ann_fail = 0

        for idx, filename in enumerate(annotation_filenames, 1):
            print(f"\n【{idx}/{len(annotation_filenames)}】")
            if process_annotation_image_pair(filename):
                ann_success += 1
            else:
                ann_fail += 1

        print(f"\n标注+图片处理统计：成功{ann_success}个，失败{ann_fail}个")

    # 3. 处理描述文件
    cap_success, cap_fail = process_all_caption_files()

    # 4. 输出总体处理结果
    print("\n" + "=" * 60)
    print("全部处理完成！")
    print("=" * 60)
    print(
        f"标注+图片文件：成功{ann_success if 'ann_success' in locals() else 0}个，失败{ann_fail if 'ann_fail' in locals() else 0}个"
    )
    print(f"描述文件：成功{cap_success}个，失败{cap_fail}个")
    print(f"\n处理后的文件保存位置：")
    print(f"- 标注文件：{ANNOTATIONS_OUTPUT_DIR}")
    print(f"- 图片文件：{IMAGES_OUTPUT_DIR}")
    print(f"- 中文描述：{CAPTION_CN_OUTPUT_DIR}")
    print(f"- 英文描述：{CAPTION_EN_OUTPUT_DIR}")
    print("=" * 60)


# 执行主函数
if __name__ == "__main__":
    main()
