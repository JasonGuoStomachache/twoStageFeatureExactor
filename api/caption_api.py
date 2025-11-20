import json
import os
import asyncio
import aiofiles
import logging
from datetime import datetime
from openai import AsyncOpenAI
from PIL import Image
import base64
from tqdm import tqdm
import shutil
import sys


# 配置日志系统
def setup_logging(log_dir="logs"):
    """设置日志记录，同时输出到控制台和文件"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 日志文件名包含当前日期时间
    log_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"
    log_filepath = os.path.join(log_dir, log_filename)

    # 配置日志格式
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # 同时输出到控制台和文件
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_filepath, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    return log_filepath


# 初始化日志
log_file = setup_logging()
logging.info(f"日志文件已创建: {log_file}")

class_names = [
    "草地贪夜蛾",
    "黏虫",
    "棉铃虫",
    "玉米螟",
    "双斑萤叶甲",
    "蚜虫",
    "麦圆蜘蛛",
    "吸浆虫",
]

# 初始化异步Ark客户端
try:
    client = AsyncOpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key="f9f9e218-5f9d-4564-af7b-2eb63692ecdb",
    )
    logging.info("AsyncOpenAI客户端初始化成功")
except Exception as e:
    logging.error(f"AsyncOpenAI客户端初始化失败: {str(e)}", exc_info=True)
    raise

# 控制并发数量，避免API请求过于频繁
MAX_CONCURRENT_TASKS = 5  # 可根据API限制调整
MAX_BBOX_COUNT = 10  # 最大标注数量阈值，超过此数量则跳过


def encode_image(image_path):
    """编码图像为base64格式"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logging.error(f"编码图像 {image_path} 失败: {str(e)}")
        raise


async def process_single_image(filename, image_dir_path, bbox_dir_path, save_dir_path):
    """异步处理单张图片，同时复制标注文件"""
    # 检查文件是否已处理
    output_file = os.path.join(save_dir_path, filename.split(".")[0] + "_caption.txt")
    if os.path.exists(output_file):
        logging.info(f"图片 {filename} 已处理，跳过")
        return True

    try:
        image_path = os.path.join(image_dir_path, filename)
        bbox_path = os.path.join(bbox_dir_path, filename.split(".")[0] + ".txt")

        # 检查图片尺寸
        with Image.open(image_path) as img:
            width, height = img.size
            if width < 40 or height < 40:
                logging.warning(f"图片 {filename} 尺寸过小({width}x{height})，跳过处理")
                return False

        # 准备API调用参数
        img_b64_str = encode_image(image_path)
        img_type = "image/jpeg"

        # 提取类别信息
        try:
            class_index = filename[8:11]
            class_index = int(class_index) - 1
            class_name = class_names[class_index]
            class_prompt = {}
            class_prompt["图片文件名"] = filename
            class_prompt["害虫类别"] = class_name
            class_prompt = json.dumps(class_prompt, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"提取类别信息失败，文件名: {filename}, 错误: {str(e)}")
            return False

        # 构建提示词
        prompt = """你现在是一名农业虫害领域的专家，你的任务是帮助我提取图片中所有害虫的具体形态特征。我会提供一张包含害虫的图片，图片文件名，害虫的中文名称。提取害虫特征时请注意：
        1、以图像字幕的任务形式，生成这张包含害虫图片的文本描述，请注意要在包含少许害虫环境描述的情况下，主要要将描述重心放在害虫的具体形态特征上，并使用专业的农业词汇。 
        2、一个害虫存在多种生命阶段，请务必在对应的生命阶段寻找所提供图片中害虫出现的形态特征。
        3、我们还需要中英文双语的文本，请尽量让这两个版本可以相互对照翻译。
        最终必须使用json的格式进行输出，例如
        {
            "Image filename": "(需要你填入的具体图片的文件名)",
            "Pest category CN": "(需要你填入的具体的中文害虫名称)",
            "Pest category EN": "(需要你填入的具体的英文害虫名称)",
            "The life stage of pest CN": "(你提取到害虫生命阶段中文名称)",
            "The life stage of pest EN": "(你提取到害虫生命阶段英文名称，在Egg, Larva, Pupa, male adult, female adult, Nymph中选出)",
            "The image caption CN": "(你提取到的图像文本描述的中文)",
            "The image caption EN": "(你提取到的图像文本描述的英文)"
        }
        我提供的信息和图片如下:"""
        prompt += class_prompt

        # 异步调用API
        try:
            logging.info(f"开始调用API处理图片: {filename}")
            response = await client.chat.completions.create(
                model="doubao-1.5-vision-pro-250328",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{img_type};base64,{img_b64_str}"
                                },
                            },
                        ],
                    }
                ],
            )
            logging.info(f"API调用成功，图片: {filename}")
        except Exception as e:
            logging.error(
                f"API调用失败，图片: {filename}, 错误: {str(e)}", exc_info=True
            )
            return False

        # 保存结果
        try:
            async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
                await f.write(response.choices[0].message.content)
            logging.info(f"结果已保存: {output_file}")
        except Exception as e:
            logging.error(f"保存结果失败，文件: {output_file}, 错误: {str(e)}")
            return False

        # 复制图片
        try:
            target_image_path = os.path.join(save_dir_path, filename)
            shutil.copy2(image_path, target_image_path)
            logging.info(f"图片已复制: {target_image_path}")
        except Exception as e:
            logging.error(
                f"复制图片失败，源: {image_path}, 目标: {target_image_path}, 错误: {str(e)}"
            )
            return False

        # 复制标注文件到目标目录
        try:
            if os.path.exists(bbox_path):
                target_bbox_path = os.path.join(
                    save_dir_path, filename.split(".")[0] + ".txt"
                )
                shutil.copy2(bbox_path, target_bbox_path)
                logging.info(f"标注文件已复制: {target_bbox_path}")
        except Exception as e:
            logging.error(
                f"复制标注文件失败，源: {bbox_path}, 目标: {target_bbox_path}, 错误: {str(e)}"
            )
            # 标注文件复制失败不影响整体结果，继续执行

        return True

    except Exception as e:
        logging.error(f"处理图片 {filename} 时出错: {str(e)}", exc_info=True)
        return False


async def process_images_async(image_dir_path, bbox_dir_path, save_dir_path):
    """异步处理目录中的所有图片"""
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
        logging.info(f"创建保存目录: {save_dir_path}")

    # 获取所有jpg文件
    try:
        filename_list = [
            name
            for name in os.listdir(image_dir_path)
            if name.split(".")[-1].lower() == "jpg"
        ]
        logging.info(f"在目录 {image_dir_path} 中找到 {len(filename_list)} 个jpg文件")
    except Exception as e:
        logging.error(
            f"获取图片文件列表失败，目录: {image_dir_path}, 错误: {str(e)}",
            exc_info=True,
        )
        return

    # 使用信号量控制并发数量
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

    async def sem_task(filename):
        async with semaphore:
            return await process_single_image(
                filename, image_dir_path, bbox_dir_path, save_dir_path
            )

    # 创建所有任务并执行
    tasks = [sem_task(filename) for filename in filename_list]

    # 使用tqdm显示进度
    results = []
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="处理进度"):
        results.append(await f)

    success_count = sum(results)
    fail_count = len(results) - success_count
    logging.info(
        f"目录处理完成，成功: {success_count}, 失败: {fail_count}, 总数量: {len(results)}"
    )
    print(f"处理完成，成功: {success_count}, 失败: {fail_count}")


async def main():
    """主函数"""
    base_dir = r"C:\Users\35088\Desktop\25.7.24\pest_text\api\data\images"
    logging.info(f"开始处理基础目录: {base_dir}")

    try:
        # 获取所有子目录
        subdirs = [
            i for i in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, i))
        ]
        logging.info(f"找到 {len(subdirs)} 个子目录")

        # 逐个处理子目录
        for subdir in subdirs:
            logging.info(f"开始处理子目录: {subdir}")
            image_dir_path = os.path.join(base_dir, subdir)
            bbox_dir_path = os.path.join(base_dir.replace("images", "bbox"), subdir)
            save_dir_path = os.path.join(
                r"C:\Users\35088\Desktop\25.7.24\pest_text\api\old_data\caption", subdir
            )
            # save_dir_path = os.path.join(base_dir.replace("images", "caption"), subdir)

            await process_images_async(image_dir_path, bbox_dir_path, save_dir_path)
            logging.info(f"子目录 {subdir} 处理完成")
    except Exception as e:
        logging.error(f"处理目录时出错: {str(e)}", exc_info=True)


if __name__ == "__main__":
    try:
        logging.info("程序开始运行")
        asyncio.run(main())
        logging.info("程序正常结束")
    except Exception as e:
        logging.critical(f"程序运行出错并终止: {str(e)}", exc_info=True)
