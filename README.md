# Pest Image & Text Data Processing Toolkit  
## 害虫图像与文本数据处理工具集  

---

## 1. Overview / 概述  
This toolkit provides a complete end-to-end workflow for agricultural pest data processing, integrating **image analysis**, **annotation management**, **text mining**, and **bilingual caption generation**. It supports 8 common agricultural pest species and is designed to facilitate pest identification model development, dataset construction, and pest characteristic data mining for researchers and engineers in agricultural technology.  

该工具集提供农业害虫数据处理的完整端到端工作流，融合**图像分析**、**标注管理**、**文本挖掘**与**双语描述生成**四大核心能力，支持8种常见农业害虫类型。旨在为农业技术领域的研究者与工程师提供害虫识别模型开发、数据集构建及害虫特征数据挖掘的高效工具。  

---

## 2. Supported Pest Species / 支持的害虫种类  
| No. | English Name | Chinese Name | 害虫类别 |
|-----|--------------|--------------|----------|
| 1   | Fall Armyworm | 草地贪夜蛾 | 鳞翅目夜蛾科 |
| 2   | Armyworm | 黏虫 | 鳞翅目夜蛾科 |
| 3   | Cotton Bollworm | 棉铃虫 | 鳞翅目夜蛾科 |
| 4   | Corn Borer | 玉米螟 | 鳞翅目螟蛾科 |
| 5   | Two spotted LeafBeetle | 双斑萤叶甲 | 鞘翅目叶甲科 |
| 6   | Aphid | 蚜虫 | 半翅目蚜科 |
| 7   | Spider Mite | 麦圆蜘蛛 | 蜱螨目叶螨科 |
| 8   | Wheat Midge | 吸浆虫 | 双翅目蚊类 |

---

## 3. Core Functions & Corresponding Scripts / 核心功能与对应脚本  
| Function Module / 功能模块 | Corresponding Script / 对应脚本 | Key Features / 核心特性 |
|----------------------------|---------------------------------|-------------------------|
| **Image Deduplication**<br>图像去重 | `image_deduplication.py` | - Based on ResNet50 feature extraction<br>- Cosine similarity for duplicate detection<br>- Adjustable similarity threshold (0-1)<br>- Visualization of duplicate groups<br>- 基于ResNet50特征提取<br>- 余弦相似度检测重复图像<br>- 可调节相似性阈值（0-1）<br>- 重复图像组可视化 |
| **Annotation Conversion**<br>标注格式转换 | `json2yolo.py` | - Convert JSON annotation files to YOLO format<br>- Auto-detect file encoding (chardet)<br>- Batch processing of multi-file directories<br>- JSON标注文件转YOLO格式<br>- 自动检测文件编码（chardet）<br>- 多文件目录批量处理 |
| **Data Reindexing**<br>数据重新编号 | `reindex.py` | - Unified modification of YOLO class IDs<br>- Sync renaming of images/annotations/captions<br>- Update "Image filename" in JSON captions<br>- No overwriting of original files (output to new dir)<br>- 统一修改YOLO类别编号<br>- 同步重命名图像/标注/描述文件<br>- 更新JSON描述中的“Image filename”字段<br>- 不覆盖原文件（输出至新目录） |
| **Bounding Box Visualization**<br>标注框可视化 | `tobbox.py` | - Batch draw YOLO annotations on images<br>- Customizable box colors and line thickness<br>- Serial number display for multiple bboxes<br>- Support for single/image batch processing<br>- 批量在图像上绘制YOLO标注框<br>- 可自定义框颜色与线条粗细<br>- 多标注框序号显示<br>- 支持单图/批量处理 |
| **Image & Bbox Analysis**<br>图像与标注分析 | `1_imagebbox_analysis.py` | - Statistical analysis: pixel count, aspect ratio<br>- Bbox distribution: area ratio, center position<br>- Visualization of bbox center distribution<br>- Export results to PDF/SVG<br>- 统计分析：像素数量、宽高比<br>- 标注框分布：面积占比、中心位置<br>- 标注框中心点分布可视化<br>- 结果导出为PDF/SVG |
| **Text Feature Analysis**<br>文本特征分析 | `2_txt_analysis.py` | - Extract English text features from captions<br>- Text length distribution (with outlier removal)<br>- Unique feature count statistics<br>- Side-by-side chart visualization<br>- 从描述中提取英文文本特征<br>- 文本长度分布（含异常值移除）<br>- 独特特征数量统计<br>- 并列图表可视化 |
| **Word Cloud Generation**<br>词云生成 | `3_ciyun.py` | - Extract "characteristics/caption" fields from JSON<br>- English stopword filtering (NLTK)<br>- Top 10 word frequency display<br>- High-resolution word cloud export (300 DPI)<br>- 从JSON提取“特征/描述”字段<br>- 英文停用词过滤（NLTK）<br>- 显示Top10词频<br>- 高分辨率词云导出（300 DPI） |
| **Bilingual Caption Generation**<br>双语描述生成 | `stage1.py` + `stage2.py` | - Stage1: Extract Chinese features via LLM<br>- Stage2: Translate to English (agricultural terminology)<br>- Async API calls (controllable concurrency)<br>- Automatic log recording<br>- 阶段1：通过大模型提取中文特征<br>- 阶段2：翻译为英文（农业专业术语）<br>- 异步API调用（可控制并发）<br>- 自动日志记录 |
| **Image Caption API**<br>图像描述API | `caption_api.py` | - Generate bilingual captions (CN/EN)<br>- Focus on pest morphology + basic environment<br>- Sync copy of images/annotations<br>- Support for small image filtering (<40x40)<br>- 生成双语图像描述（中/英）<br>- 聚焦害虫形态+基础环境描述<br>- 同步复制图像/标注文件<br>- 支持小尺寸图像过滤（<40x40） |

---


## 5. Quick Start / 快速开始  

### 5.1 Environment Setup / 环境配置  
First, install the required dependencies via `pip`:  
首先通过`pip`安装依赖包：  
```bash
# Install core dependencies / 安装核心依赖
pip install numpy==1.26.4 pandas==2.2.1 matplotlib==3.8.4 pillow==10.3.0 opencv-python==4.9.0.80
pip install torch==2.2.2 torchvision==0.17.2 scikit-learn==1.4.2 nltk==3.8.1 wordcloud==1.9.3
pip install tqdm==4.66.2 aiofiles==23.2.1 openai==1.13.3 chardet==5.2.0
```


### 5.2 Data Preparation / 数据准备
1. Organize pest images into data/images/ (create subdirectories for each species, e.g., data/images/01_fall_armyworm).
将害虫图像按种类分目录放入data/images/（例：data/images/01_fall_armyworm）。
2. Place corresponding YOLO annotation files (.txt) into data/bbox/ (same directory structure as images).
将对应的 YOLO 标注文件（.txt）放入data/bbox/（目录结构与images一致）。
3. Configure file paths in each script (e.g., main_data_path in 2_txt_analysis.py, input_dir in image_deduplication.py) to match your local environment.
在各脚本中配置文件路径（如2_txt_analysis.py的main_data_path、image_deduplication.py的input_dir），确保与本地环境一致。

### 5.3 Core Workflow Execution / 核心流程执行

Step 1: Image Deduplication / 图像去重

```bash
python api/image_deduplication.py \
  --input_dir "data/images/01_fall_armyworm" \
  --output_dir "data/repeat_images/01_fall_armyworm" \
  --threshold 0.95 \
  --batch_size 32 \
  --visualize True
```
Step 2: Annotation Conversion (JSON → YOLO) / 标注格式转换（JSON 转 YOLO）

```bash
# Modify `json_path` in `json2yolo.py` first, then run:
# 先修改`json2yolo.py`中的`json_path`，再执行：
python api/json2yolo.py
```

Step 3: Data Reindexing / 数据重新编号

```bash
# Configure `CLASS_MAPPING` and directory paths in `reindex.py`, then run:
# 先在`reindex.py`中配置`CLASS_MAPPING`与目录路径，再执行：
python api/reindex.py
```

Step 4: Bounding Box Visualization / 标注框可视化

```bash
# Modify `single_img_path`/`batch_img_dir` in `tobbox.py`, then run:
# 先在`tobbox.py`中修改`single_img_path`/`batch_img_dir`，再执行：
python api/tobbox.py
```

Step 5: Bilingual Caption Generation / 双语描述生成

```bash
# Stage 1: Extract Chinese features (need valid API key)
# 阶段1：提取中文特征（需有效API密钥）
python api/stage1.py

# Stage 2: Translate to English
# 阶段2：翻译为英文
python api/stage2.py
```

Step 6: Data Analysis & Visualization / 数据分析与可视化

```bash
# Image & bbox analysis (export PDF)
# 图像与标注分析（导出PDF）
python api/1_imagebbox_analysis.py

# Text feature analysis (export PNG/PDF)
# 文本特征分析（导出PNG/PDF）
python api/2_txt_analysis.py

# Word cloud generation (export PNG)
# 词云生成（导出PNG）
python api/3_ciyun.py
```

### Notes / 注意事项

1. API Key Requirement: Scripts like stage1.py, stage2.py, and caption_api.py require a valid AsyncOpenAI API key (configure in the script).
API 密钥需求：stage1.py、stage2.py、caption_api.py等脚本需有效AsyncOpenAI API 密钥（在脚本中配置）。
2. Directory Consistency: Ensure the directory structure of images/, bbox/, and caption/ is identical (same subdirs and filenames).
目录一致性：确保images/、bbox/、caption/的目录结构完全一致（相同子目录与文件名）。
3. Hardware Requirements: Image deduplication and LLM calls require basic GPU support (recommended: NVIDIA GTX 1060+). For CPU-only environments, reduce batch_size to 8-16.
硬件需求：图像去重与大模型调用需基础 GPU 支持（推荐：NVIDIA GTX 1060+）。纯 CPU 环境需将batch_size降至 8-16。
4. File Formats: Supported image formats: .jpg, .jpeg, .png, .bmp, .webp; supported annotation formats: YOLO (.txt), JSON (.json).
文件格式：支持图像格式：.jpg、.jpeg、.png、.bmp、.webp；支持标注格式：YOLO（.txt）、JSON（.json）。
5. Log Management: Logs are saved to the logs/ directory (with timestamps) for troubleshooting.
日志管理：日志保存至logs/目录（带时间戳），便于问题排查。

### Contact / 联系方式
For technical issues, feature requests, or bug reports, please submit an Issue or contact the author via [email](guojiaxuan@stu.xidian.edu.cn).
如遇技术问题、功能需求或 Bug 反馈，请提交Issue或通过邮件联系作者。