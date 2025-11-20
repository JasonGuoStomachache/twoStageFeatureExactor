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


async def process_single_image(filename, base_dir_path, save_dir_path):
    """异步处理单张图片，同时复制标注文件"""
    # 检查文件是否已处理
    output_file = os.path.join(
        save_dir_path, filename.split("_")[0] + "_caption_en.txt"
    )
    if os.path.exists(output_file):
        logging.info(f"文件 {filename} 已处理，跳过{output_file}")
        return True

    try:
        # 准备API调用参数

        # 提取中文信息
        with open(os.path.join(base_dir_path, filename), "r", encoding="utf-8") as file:
            # 读取全部内容
            content = file.read()

        # 构建提示词
        prompt = """You are now an expert in the field of agricultural pest control, and your task is to help me translate a Chinese pest characteristic information into English. The Chinese information features I provide will be presented in strict JSON format, as shown in the following example:
        {
            "图片的文件名": "(需要你填入的具体图片的文件名)",
            "害虫类别": "(需要你填入的具体害虫名称)",
            "害虫1": {
                "害虫的相对位置信息": "(用户所提供的害虫1的相对位置信息)",
                "害虫所处的生命阶段": "(你提取到害虫生命阶段)",
                "害虫形态特征": "(结合提供的形态特征从图片中提取出的名词短语，使用英文逗号分隔)"
            },
            "害虫2": {
                "害虫的相对位置信息": "(用户所提供的害虫2的相对位置信息)",
                "害虫所处的生命阶段": "(你提取到害虫生命阶段)",
                "害虫形态特征": "(结合提供的形态特征从图片中提取出的名词短语，使用英文逗号分隔)"
            },
            ...
        }
        Please process the information I provided according to the following requirements:
        1、Directly use the input content in the Image filename and The bounding box of pest fields. 
        2、Translate professional analogies by combining agricultural knowledge in the Pest category field. 
        3、Select the appropriate one from Egg, Larva, Pupa, male adult, female adult, and Nymph in the 'The Life Stage of Pest' field. 
        4、In the 'Characteristics of pest' field, we first need to filter the pest features which include specific numbers, such as; "体长约 17.2 毫米", or "幼虫体长约 10.0 毫米". This type of content without generalization is removed, and then the removed content is translated into English based on agricultural expertise. 
        5、Regardless of Chinese or English, ensure to separate different pest characteristics with commas in English
        The final output result should include both Chinese and English, and the specific JSON format of the output is as follows. Please make sure to follow the standard.
        {
            "Image filename": "(Directly use the original content)",
            "Pest category CN": "(Original Chinese Content)",
            "Pest category EN": "(Translated English content)",
            "pest 1": {
                "The bounding box of pest": "(Directly use the pets 1 original content)",
                "The life stage of pest CN": "(Original Chinese Content)",
                "The life stage of pest EN": "(Translate using agricultural terminology such as Egg, Larva, Pupa, male adult, female adult, Nymph)",
                "The Characteristics of pest CN": "(Chinese content after removing content)",
                "The Characteristics of pest EN": "(Translated English content)"
            },
            "pest 2": {
                "The bounding box of pest": "(Directly use the pets 2 original content)",
                "The life stage of pest CN": "(Original Chinese Content)",
                "The life stage of pest EN": "(Translate using agricultural terminology such as Egg, Larva, Pupa, male adult, female adult, Nymph)",
                "The Characteristics of pest CN": "(Chinese content after removing content)",
                "The Characteristics of pest EN": "(Translated English content)"
            },
            ...
        }
        The specific information I provided:
        """
        prompt += content

        # 异步调用API
        try:
            logging.info(f"开始调用API处理图片: {filename}")
            response = await client.chat.completions.create(
                model="doubao-1-5-pro-32k-250115",
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                ],
            )
            logging.info(f"API调用成功，文件: {filename}")
        except Exception as e:
            logging.error(
                f"API调用失败，文件: {filename}, 错误: {str(e)}", exc_info=True
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

        return True

    except Exception as e:
        logging.error(f"处理文件 {filename} 时出错: {str(e)}", exc_info=True)
        return False


async def process_images_async(file_dir_path, save_dir_path):
    """异步处理目录中的所有图片"""
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
        logging.info(f"创建保存目录: {save_dir_path}")

    # 获取所有jpg文件
    try:
        filename_list = [
            name
            for name in os.listdir(file_dir_path)
            if name.split("_")[-1].lower() == "caption.txt"
        ]
        logging.info(
            f"在目录 {file_dir_path} 中找到 {len(filename_list)} 个caption文件"
        )
    except Exception as e:
        logging.error(
            f"获取caption文件列表失败，目录: {file_dir_path}, 错误: {str(e)}",
            exc_info=True,
        )
        return

    # 使用信号量控制并发数量
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

    async def sem_task(filename):
        async with semaphore:
            return await process_single_image(filename, file_dir_path, save_dir_path)

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
    base_dir = r"C:\Users\35088\Desktop\25.7.24\pest_text\api\data\caption"
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
            file_dir_path = os.path.join(base_dir, subdir)
            save_dir_path = os.path.join(
                base_dir.replace("caption", "caption_en"), subdir
            )

            await process_images_async(file_dir_path, save_dir_path)
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
