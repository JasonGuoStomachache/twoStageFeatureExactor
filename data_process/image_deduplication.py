import os
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Set

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 定义图像转换
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class ImageDataset(Dataset):
    """用于加载图像的数据集类"""

    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, image_path
        except Exception as e:
            print(f"无法加载图像 {image_path}: {e}")
            # 返回None作为占位符，后续会过滤掉
            return None, image_path


def get_feature_extractor():
    """获取预训练的ResNet50模型作为特征提取器"""
    model = models.resnet50(pretrained=True)
    # 移除最后的全连接层，只保留特征提取部分
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model


@torch.no_grad()
def extract_features(
    image_paths: List[str], batch_size: int = 32
) -> Dict[str, np.ndarray]:
    """从图像中提取特征"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_feature_extractor().to(device)

    dataset = ImageDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    features = {}

    for images, paths in dataloader:
        # 过滤掉无法加载的图像
        valid_indices = [i for i, img in enumerate(images) if img is not None]
        if not valid_indices:
            continue

        valid_images = torch.stack([images[i] for i in valid_indices]).to(device)
        valid_paths = [paths[i] for i in valid_indices]

        # 提取特征
        outputs = model(valid_images)
        # 将特征展平为一维向量
        outputs = outputs.squeeze(-1).squeeze(-1).cpu().numpy()

        # 保存特征
        for path, feature in zip(valid_paths, outputs):
            features[path] = feature

    return features


def find_duplicates(
    features: Dict[str, np.ndarray], threshold: float = 0.95
) -> List[Set[str]]:
    """找出相似的图像组"""
    paths = list(features.keys())
    feature_matrix = np.array([features[path] for path in paths])

    # 计算余弦相似度矩阵
    similarity_matrix = cosine_similarity(feature_matrix)

    # 找出相似度超过阈值的图像对
    duplicates = []
    visited = set()

    for i in range(len(paths)):
        if i in visited:
            continue

        group = {paths[i]}
        for j in range(i + 1, len(paths)):
            if similarity_matrix[i, j] >= threshold:
                group.add(paths[j])
                visited.add(j)

        if len(group) > 1:
            duplicates.append(group)

    return duplicates


def visualize_duplicates(duplicate_groups: List[Set[str]], max_groups: int = 5):
    """可视化重复的图像组"""
    for i, group in enumerate(duplicate_groups[:max_groups]):
        plt.figure(figsize=(15, 10))
        plt.suptitle(f"重复图像组 {i+1}", fontsize=16)

        for j, path in enumerate(group):
            try:
                img = Image.open(path)
                plt.subplot(1, len(group), j + 1)
                plt.imshow(img)
                plt.title(os.path.basename(path))
                plt.axis("off")
            except Exception as e:
                print(f"无法显示图像 {path}: {e}")

        plt.tight_layout()
        plt.show()


def save_duplicates_to_csv(duplicate_groups: List[Set[str]], output_file: str):
    """将重复图像组保存到CSV文件"""
    data = []
    for group_id, group in enumerate(duplicate_groups):
        for path in group:
            data.append(
                {
                    "group_id": group_id,
                    "image_path": path,
                    "file_name": os.path.basename(path),
                }
            )

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"重复图像信息已保存到 {output_file}")


def move_duplicates(
    duplicate_groups: List[Set[str]], destination_dir: str, keep_first: bool = True
):
    """移动重复的图像到指定目录"""
    os.makedirs(destination_dir, exist_ok=True)

    for group_id, group in enumerate(duplicate_groups):
        # 转换为列表以便索引
        group_list = list(group)
        # 如果keep_first为True，则保留第一个图像，移动其余的
        # 否则，移动所有图像
        start_idx = 1 if keep_first else 0

        for i in range(start_idx, len(group_list)):
            src_path = group_list[i]
            file_name = os.path.basename(src_path)
            # 添加组ID作为前缀，避免文件名冲突
            dst_path = os.path.join(destination_dir, f"group_{group_id}_{file_name}")
            try:
                shutil.move(src_path, dst_path)
                print(f"已移动: {src_path} -> {dst_path}")
            except Exception as e:
                print(f"无法移动 {src_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="基于深度学习的图片去重工具")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=r"C:\Users\35088\Desktop\25.7.24\pest_text\api\data\08_xijiangchong",
        help="包含图片的输入目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"C:\Users\35088\Desktop\25.7.24\pest_text\api\repeat\08_xijiangchong",
        help="重复图片的输出目录",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="相似度阈值，范围从0到1，值越大表示要求越严格",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="批量处理的图片数量")
    parser.add_argument(
        "--csv_file",
        type=str,
        default="./duplicates.csv",
        help="保存重复图片信息的CSV文件",
    )
    parser.add_argument(
        "--keep_first",
        action="store_true",
        default=True,
        help="保留每组中的第一张图片，只移动其余的",
    )
    parser.add_argument(
        "--no_move", action="store_true", help="只识别重复图片，不移动它们"
    )
    parser.add_argument(
        "--visualize", default=True, action="store_true", help="可视化部分重复图片组"
    )

    args = parser.parse_args()

    # 获取所有图片文件
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    image_paths = []

    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))

    print(f"找到 {len(image_paths)} 张图片")

    if not image_paths:
        print("没有找到图片，程序退出")
        return

    # 提取特征
    print("正在提取图片特征...")
    features = extract_features(image_paths, batch_size=args.batch_size)
    print(f"成功提取 {len(features)} 张图片的特征")

    # 找出重复图片
    print("正在查找重复图片...")
    duplicate_groups = find_duplicates(features, threshold=args.threshold)
    print(f"找到 {len(duplicate_groups)} 组重复图片")

    # 保存重复图片信息到CSV
    # save_duplicates_to_csv(duplicate_groups, args.csv_file)

    # 可视化重复图片
    if args.visualize and duplicate_groups:
        visualize_duplicates(duplicate_groups)

    # 移动重复图片
    if not args.no_move and duplicate_groups:
        move_duplicates(duplicate_groups, args.output_dir, keep_first=args.keep_first)

    print("图片去重处理完成!")


if __name__ == "__main__":
    main()
