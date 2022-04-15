import os
import re
import sys
import numpy as np
import jieba.analyse
import matplotlib.pyplot as plt
from tqdm import tqdm
from gensim.models import word2vec
import logging

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)     # 添加debug信息
plt.rcParams['font.sans-serif'] = ['SimHei']                                                    # 中文导入
plt.rcParams['axes.unicode_minus'] = False                                                      # 正负号导入
# fix the random seed
np.random.seed(100)                                                                             # 固定随机数种子
sys.stdout.flush()                                                                              # 刷新缓冲区
nFile = 200                                                                                     # 文件数目
root_path = "../data/"                                                                          # 文件路径
dir_list = os.listdir(root_path)
rate = 0.7                                                                                      # 训练集、测试集分割率

def train_model():
    """
    训练word2vec模型
    :return: NULL
    """
    all_word_list = []
    for dir in dir_list:
        dir_path = root_path + "/" + dir
        file_list = os.listdir(dir_path)
        for file in file_list:
            file_path = dir_path + "/" + file
            with open(file_path, "r", encoding="utf-8") as f:
                txt = f.read()
                txt = re.sub("[    \t\n]*", "", txt)
                word_list = jieba.analyse.textrank(txt, topK=None, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
                all_word_list.extend(word_list)
    result = " ".join(all_word_list)
    with open("result.txt", "w", encoding="utf-8") as f:
        f.write(result)
    sentences = word2vec.Text8Corpus("result.txt")                                          # 加载语料
    model = word2vec.Word2Vec(sentences, min_count=1, vector_size=200)                      # 训练模型，维度设置为200，最小词频设置为1
    model.save("my_model.model")                                                            # 模型保存

def get_word_list():
    """
    统计训练集中词频
    :return: class_all_words  词频
    """
    class_all_words = {}
    print("对训练集词频统计")
    for c in tqdm(dir_list):
        all_words = {}
        class_path = root_path + "/" + c
        file_list = os.listdir(class_path)
        for name in file_list[:int(nFile*rate)]:
            file_path = class_path + "/" + name
            with open(file_path, "r", encoding="utf-8") as f:
                txt = f.read()
                txt = re.sub("[    \t\n]*", "", txt)
                word_list = jieba.analyse.textrank(txt, topK=None, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
                for word in word_list:
                    if len(word) != 1:
                        if word in all_words.keys():
                            all_words[word] += 1
                        else:
                            all_words[word] = 1
                # 统计每一类的词频
            class_all_words[c] = all_words
    return class_all_words

def average_class_vector(class_all_words, keyword_num, model):
    """
    获得每个类的平均特征向量
    :param class_all_words: 统计得到的词频列表
    :param keyword_num: 关键词个数
    :param model: 训练好的word2vec模型
    :return: 平均特征向量 average_class_dic
    """
    average_class_dic = {}
    for c in class_all_words:
        all_words_list = sorted(class_all_words[c].items(), key=lambda item:item[1], reverse=True)
        total = 0
        if keyword_num > len(all_words_list):
            for t in range(len(all_words_list)):
                total += model.wv[all_words_list[t][0]]
        else:
            for t in range(keyword_num):
                total += model.wv[all_words_list[t][0]]
        average_class_dic[c] = total / keyword_num
    return average_class_dic

def predict(average_class_dic_input, data):
    """
    结果预测
    :param average_class_dic_input: 已获取的平均特征向量
    :param data: 待类比词向量
    :return: 最相似的词向量类别 test_words_list[0][0]
    """
    sim = {}
    for c in average_class_dic_input:
        sim[c] = cos_sim(data, average_class_dic_input[c])
    test_words_list = sorted(sim.items(), key=lambda item: item[1], reverse=True)
    return test_words_list[0][0]

def acc(average_class_dic_data, keyword_num, model):
    """
    测试准确率
    :param average_class_dic_data: 已获取的平均特征向量
    :param keyword_num: 词向量个数
    :param model: 训练好的word2vec模型
    :return: 测试准确率 true / (true + false)
    """
    true = 0
    false = 0
    print("Keyword_num: {}".format(keyword_num))
    for c in tqdm(dir_list):
        class_path = root_path + "" + c
        file_list = os.listdir(class_path)

        for name in file_list[int(nFile*rate):]:
            file_path = class_path + "/" + name
            with open(file_path, "r", encoding="utf-8") as f:
                test_data_words = {}
                txt = f.read()
                txt = re.sub("[    \t\n]*", "", txt)
                word_list = jieba.analyse.textrank(txt, topK=None, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
                for word in word_list:
                    if len(word) != 1:
                        if word in test_data_words.keys():
                            test_data_words[word] += 1
                        else:
                            test_data_words[word] = 1
                test_words_list = sorted(test_data_words.items(), key=lambda item: item[1])

                total = 0
                if keyword_num > len(test_words_list):
                    for t in range(len(test_words_list)):
                        total += model.wv[test_words_list[t][0]]
                else:
                    for t in range(keyword_num):
                        total += model.wv[test_words_list[t][0]]
                average_test_vector = total / keyword_num
                pre = predict(average_class_dic_data, average_test_vector)
                if pre == c:
                    true += 1
                else:
                    false += 1
    return true / (true + false)


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量a
    :param vector_b: 向量b
    :return: sim 余弦相似度
    """
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)
    a_norm = np.linalg.norm(vector_a)
    b_norm = np.linalg.norm(vector_b)
    sim = np.dot(vector_a, vector_b)/(a_norm*b_norm)
    return sim


if __name__ == "__main__":
    keyword_num_list = [1, 3, 5, 10, 15, 20, 30, 50, 100, 130, 150, 180]
    # train_model()                                                                           # 训练模型
    acc_list = []
    class_all_words = get_word_list()                                                       # 获得词频
    model = word2vec.Word2Vec.load('my_model.model')                                        # 加载模型
    print("calculate the accuracy")
    for keyword in keyword_num_list:                                                        # 对每个关键词个数进行测试
        average_class_dic = average_class_vector(class_all_words, keyword, model)           # 计算对应关键词的平均特征向量
        acc_list.append(round(acc(average_class_dic, keyword, model), 3))                   # 根据平均向量测试得到标签
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("关键词个数与准确率的关系")
    ax.set_xlabel("关键词个数")
    ax.set_ylabel("准确率")
    plt.plot(keyword_num_list, acc_list, color="black", markerfacecolor="r", marker="o")
    for a, b in zip(keyword_num_list, acc_list):
        plt.text(a, b, (a, b), ha="center", va="bottom", fontsize=10)
    plt.show()