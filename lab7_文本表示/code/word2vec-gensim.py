# 31 34注释，32 33反注释
# -*- coding: utf-8 -*-
# @Time : 2019/11/13 14:55
# @FileName: word2vec-gensim.py
# @Author : yip
# @Email : 522364642@qq.com
# @Blog : https://blog.csdn.net/qq_30189255
# @Github : https://github.com/yip522364642


import warnings

warnings.filterwarnings("ignore") # 忽略警告消息

if_more_train = False

'''
1 获取文本语料并查看
'''
# with open('text8', 'r', encoding='utf-8') as file:
#     for line in file.readlines():
#         print(line)

'''
2 载入数据，训练并保存模型
'''
from gensim.models import word2vec
from gensim import corpora
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)  # 输出日志信息
sentences = word2vec.Text8Corpus('text8')  # 将语料保存在sentence中（加载训练数据，保存在sentences）

# word2vec.corpora.dictionary(sentences)

# # 创建Word2Vec模型并训练：数据，使用skip-gram模型，词向量维度100，上下文窗口5，忽略次数小于5的单词
# model = word2vec.Word2Vec(sentences, sg=1, vector_size=100,  window=5,  min_count=5,  negative=3, sample=0.001, hs=1, workers=4)  # 生成词向量空间模型
# # 保存模型
# model.save('text8_word2vec.model')  # 保存模型

# dictionary = corpora.Dictionary('text8'.split(' '))


'''
3 加载模型，实现各个功能
'''
# 加载模型
model = word2vec.Word2Vec.load('text8_word2vec.model')

# # 1.打印每个词对应的ID
# for index,word in enumerate(model.wv.index_to_key):
#     print(str(index)+":"+word)
print("\n================================")

# 2.计算两个词的相似度/相关程度
print("计算两个词的相似度/相关程度")
word1 = u'man'
word2 = u'woman'
result1 = model.wv.similarity(word1, word2)

print(word1 + "和" + word2 + "的相似度为：", result1)
print("\n================================")

# 3.计算某个词的相关词列表
print("计算某个词的相关词列表")
word = u'bad'
result2 = model.wv.most_similar(word, topn=10)  # 10个最相关的
print("和" + word + "最相关的词有：")
for item in result2:
    print(item[0], item[1])
print("\n================================")

# 4.寻找对应关系
print("寻找对应关系")
print(' "boy" is to "father" as "girl" is to ...? ')
result3 = model.wv.most_similar(['girl', 'father'], ['boy'], topn=3)
for item in result3:
    print(item[0], item[1])
print("\n")

more_examples = ["she her he", "small smaller bad", "going went being"]
for example in more_examples:
    a, b, x = example.split() # .split()默认按空格分字符串
    predicted = model.wv.most_similar([x, b], [a])[0][0] # 没有topn默认找10个最相关的
    print("'%s' is to '%s' as '%s' is to '%s'" % (a, b, x, predicted))
print("\n================================")

# 5.寻找不合群的词
print("寻找不合群的词")
result4 = model.wv.doesnt_match("flower grass pig tree".split())
print("不合群的词：", result4)
print("\n================================")

# 6.查看词向量（只在model中保留中的词）
print("查看词向量（只在model中保留中的词）")
word = 'girl'
print(word, model.wv[word])
# for word in model.wv.vocab.keys():  # 查看所有单词
#     print(word, model[word])

# 7.查看单词间的距离
word1 = 'girl'
word2 = 'boy'
word3 = 'woman'
a = model.wv.distance(word1, word2)
b = model.wv.distance(word1, word3)

print(f'the distance of girl and boy is {a}')
print(f'the distance of girl and woman is {b}')
print("\n================================")

'''
4 增量训练
'''
if if_more_train:
    model = word2vec.Word2Vec.load('text8_word2vec.model')
    more_sentences = [['Advanced', 'users', 'can', 'load', 'a', 'model', 'and', 'continue', 'training', 'it', 'with', 'more', 'sentences']] # 定义一个新的训练数据集：more_sentences
    model.build_vocab(more_sentences, update=True) # 用于更新原模型的词汇表   
    model.train(more_sentences, total_examples=model.corpus_count, epochs=1) # 使用train方法来对原模型进行增量训练，total_examples=model.corpus_count表示将包括原模型中已经训练的文本示例数量
    model.save('text8_word2vec.model')
