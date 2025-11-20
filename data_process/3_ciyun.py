import os
import json
import re
import string
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 下载必要的NLTK资源
nltk.download("punkt_tab")
nltk.download("stopwords")


def extract_characteristics(folder_path):
    """遍历文件夹，提取所有文件中的The Characteristics of pest EN字段"""
    characteristics_list = []
    # 遍历文件夹下的所有文件
    for filename in os.listdir(folder_path):
        # 仅处理txt文件（根据实际文件格式调整）
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    # 读取文件内容并解析为JSON
                    content = f.read()
                    # 处理可能的格式问题（如多余的逗号等）
                    content = re.sub(r",\s*}", "}", content)
                    data = json.loads(content)

                    # 提取所有pest的characteristics
                    pest_num = 1
                    while True:
                        pest_key = f"pest {pest_num}"
                        if pest_key not in data:
                            break
                        # 获取EN特征字段
                        charac_en = data[pest_key].get(
                            "The Characteristics of pest EN", ""
                        )
                        if charac_en:
                            characteristics_list.append(charac_en)
                        pest_num += 1
                print(f"已处理文件: {filename}")
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
    return characteristics_list


def preprocess_text(text):
    """预处理文本：小写化、去除标点和特殊字符"""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text


def tokenize_and_filter(text):
    """分词并过滤停用词"""
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [
        token for token in tokens if token not in stop_words and len(token) > 1
    ]
    return filtered_tokens


def generate_wordcloud(texts, output_file="pest_characteristics_wordcloud_feature.png"):
    """根据提取的特征生成词云"""
    combined_text = " ".join(texts)
    processed_text = preprocess_text(combined_text)
    tokens = tokenize_and_filter(processed_text)
    word_freq = Counter(tokens)

    print(f"\n共提取到 {len(texts)} 条特征描述")
    print(f"处理后得到 {len(tokens)} 个有效词语，其中独特词语 {len(word_freq)} 个")
    print("出现频率最高的10个词：")
    for word, count in word_freq.most_common(10):
        print(f"{word}: {count}次")

    # 生成词云
    wordcloud = WordCloud(
        width=1000,
        height=600,
        background_color="white",
        max_words=300,
        contour_width=1,
        contour_color="green",  # 选用与害虫相关的绿色
    ).generate_from_frequencies(word_freq)

    # 显示并保存词云
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\n词云已保存至 {output_file}")
    plt.show()


if __name__ == "__main__":
    # 目标文件夹路径（根据你的实际路径修改）
    folder_path = (
        r"C:\Users\35088\Desktop\25.7.24\pest_text\api\data\caption_en\01_caoditanyee"
    )

    # 提取特征描述
    print(f"开始遍历文件夹: {folder_path}")
    characteristics = extract_characteristics(folder_path)

    if characteristics:
        # 生成词云
        generate_wordcloud(characteristics)
    else:
        print("未提取到任何特征描述，请检查文件路径和文件格式")


import os
import json
import re
import string
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 下载必要的NLTK资源
nltk.download("punkt_tab")
nltk.download("stopwords")


def extract_characteristics(folder_path):
    """遍历文件夹，提取所有文件中的The image caption EN字段"""
    characteristics_list = []
    # 遍历文件夹下的所有文件
    for filename in os.listdir(folder_path):
        # 仅处理txt文件（根据实际文件格式调整）
        if filename.endswith("caption.txt"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    # 读取文件内容并解析为JSON
                    content = f.read()
                    # 处理可能的格式问题（如多余的逗号等）
                    content = re.sub(r",\s*}", "}", content)
                    data = json.loads(content)

                    # 提取所有pest的characteristics
                    charac_en = data.get("The image caption EN", "")
                    if charac_en:
                        characteristics_list.append(charac_en)
                print(f"已处理文件: {filename}")
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
    return characteristics_list


def preprocess_text(text):
    """预处理文本：小写化、去除标点和特殊字符"""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text


def tokenize_and_filter(text):
    """分词并过滤停用词"""
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [
        token for token in tokens if token not in stop_words and len(token) > 1
    ]
    return filtered_tokens


def generate_wordcloud(texts, output_file="pest_characteristics_wordcloud_caption.png"):
    """根据提取的特征生成词云"""
    combined_text = " ".join(texts)
    processed_text = preprocess_text(combined_text)
    tokens = tokenize_and_filter(processed_text)
    word_freq = Counter(tokens)

    print(f"\n共提取到 {len(texts)} 条特征描述")
    print(f"处理后得到 {len(tokens)} 个有效词语，其中独特词语 {len(word_freq)} 个")
    print("出现频率最高的10个词：")
    for word, count in word_freq.most_common(10):
        print(f"{word}: {count}次")

    # 生成词云
    wordcloud = WordCloud(
        width=1000,
        height=600,
        background_color="white",
        max_words=300,
        contour_width=1,
        contour_color="green",  # 选用与害虫相关的绿色
    ).generate_from_frequencies(word_freq)

    # 显示并保存词云
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\n词云已保存至 {output_file}")
    plt.show()


if __name__ == "__main__":
    # 目标文件夹路径（根据你的实际路径修改）
    folder_path = (
        r"C:\Users\35088\Desktop\25.7.24\pest_text\api\old_data\caption\01_caoditanyee"
    )

    # 提取特征描述
    print(f"开始遍历文件夹: {folder_path}")
    characteristics = extract_characteristics(folder_path)

    if characteristics:
        # 生成词云
        generate_wordcloud(characteristics)
    else:
        print("未提取到任何特征描述，请检查文件路径和文件格式")
