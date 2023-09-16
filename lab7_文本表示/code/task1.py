from gensim import corpora
from collections import defaultdict

# 将字符串切分为单词，并去掉停用词
text = "This is a sample text document for the demonstration of gensim corpora and dictionary usage" # 文本
stoplist = set('for a of the and to in'.split()) # 定义停用词集合，.split()是Python字符串方法，用于将一个字符串分割成一个由子字符串组成的列表，分割是基于指定的分隔符完成的。默认的分隔符是空格。
words = []
for word in text.lower().split():
    if word not in stoplist:
        words.append(word)

# 统计每个词的出现频度，只出现1次的词去掉
frequency = defaultdict(int)
for word in words:
    frequency[word] += 1

if frequency[word] > 1:
    for word in words:
        words.append(word)

# 使用gensim.corpora.Dictionary库将字符串转化为id的字典
dictionary = corpora.Dictionary([words])

# 字典包含了词汇表中每个词的id映射
print("词汇表的id映射：", dictionary.token2id)

# 将文本转化为id的列表
text_to_id = dictionary.doc2idx(words)
print("字符串转化为id的列表：", text_to_id)
