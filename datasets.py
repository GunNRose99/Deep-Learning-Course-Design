import torch
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import random



def process_data():
    # 12个新闻标签
    label_names = ['娱乐', '财经', '房地产', '旅游', '科技', '体育', '健康', '教育', '汽车', '新闻', '文化', '女人']
    label_dict = {label_names[i]: i for i in range(len(label_names))}

    # 处理训练集数据
    df_train = pd.read_csv("data/sohu_train.txt", encoding="gbk")
    df_train = df_train.loc[:, ["文章", "分类"]]
    df_train.columns = ["content", "label"]
    # 转换为数字标签
    df_train["label"] = df_train["label"].map(label_dict)

    # 处理测试集数据
    df_test = pd.read_csv("data/sohu_test.txt", sep="\t", header=None).iloc[:, [1, 0]]
    df_test.columns = ["content", "label"]
    df_test["label"] = df_test["label"].map(label_dict)

    # 合并训练集和测试集
    df_all = pd.concat([df_train, df_test], axis=0)

    # 输出合并后的样本总数和每个类别的样本数
    print("样本总数:", len(df_all))
    print("每个类别的样本数：")
    print(df_all['label'].value_counts())

    # 使用分层抽样划分训练集、验证集和测试集
    df_train, df_temp = train_test_split(df_all, test_size=0.3, stratify=df_all['label'], random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, stratify=df_temp['label'], random_state=42)

    # 保存划分后的数据
    df_train.to_csv("datasets/data_train.txt", sep="\t", index=False)
    df_val.to_csv("datasets/data_val.txt", sep="\t", index=False)
    df_test.to_csv("datasets/data_test.txt", sep="\t", index=False)

    # 打印划分后的数据集样本数量及每类样本数量
    print("训练集样本数:", len(df_train))
    print("验证集样本数:", len(df_val))
    print("测试集样本数:", len(df_test))

    print("\n训练集每个类别的样本数：")
    print(df_train['label'].value_counts())

    print("\n验证集每个类别的样本数：")
    print(df_val['label'].value_counts())

    print("\n测试集每个类别的样本数：")
    print(df_test['label'].value_counts())


class DataGenerator(Dataset):
    def __init__(self, root, word2index_dict, max_len):
        super(DataGenerator, self).__init__()
        self.root = root
        self.max_len = max_len
        self.word2index_dict = word2index_dict
        self.sentences, self.labels = self.get_datasets()

    def __getitem__(self, item):
        sentences = self.sentences[item]
        if len(sentences) < self.max_len:
            sentences += [0] * (self.max_len - len(sentences))
        else:
            sentences = sentences[:self.max_len]
        return torch.LongTensor(sentences), torch.from_numpy(np.array(self.labels[item])).long()

    def __len__(self):
        return len(self.labels)

    def get_datasets(self):
        sentences, labels = [], []
        with open(self.root, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                line_split = line.strip().split("\t")
                sentences.append([self.word2index_dict.get(word, 1) for word in line_split[0]])
                labels.append(int(line_split[-1]))
        return sentences, labels


# 构建词汇到索引的映射字典
def get_word2index_dict(n_common=4000):
    word_count = Counter()  # 快速统计每个词的出现次数
    data_train = pd.read_csv("datasets/data_train.txt", sep="\t")
    data_val = pd.read_csv("datasets/data_val.txt", sep="\t")
    data_test = pd.read_csv("datasets/data_test.txt", sep="\t")
    data_total = pd.concat([data_train, data_val, data_test], axis=0)
    sentences = data_total["content"].values.tolist()
    for sentence in sentences:
        for word in sentence:
            word_count[word] += 1

    # 最常见的 n_common 个词
    most_common = word_count.most_common(n_common)
    word2index_dict = {word: index + 2 for index, (word, count) in enumerate(most_common)}
    word2index_dict["PAD"] = 0  # 填充符
    word2index_dict["UNK"] = 1  # 未知词（词典中没有的）

    return word2index_dict


def get_random_word2index_dict(n_common=4000):
    word_set = set()
    data_train = pd.read_csv("datasets/data_train.txt", sep="\t")
    data_val = pd.read_csv("datasets/data_val.txt", sep="\t")
    data_test = pd.read_csv("datasets/data_test.txt", sep="\t")
    data_total = pd.concat([data_train, data_val, data_test], axis=0)

    sentences = data_total["content"].values.tolist()

    for sentence in sentences:
        for word in sentence.split():
            word_set.add(word)

    random_words = random.sample(word_set, min(n_common, len(word_set)))
    word2index_dict = {word: index + 2 for index, word in enumerate(random_words)}
    word2index_dict["PAD"] = 0  # 填充符的索引
    word2index_dict["UNK"] = 1  # 未知词的索引

    return word2index_dict


if __name__ == '__main__':
    process_data()
