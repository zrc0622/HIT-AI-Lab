# glove是静态词向量，使用时只需要将所有的文本都转为glove中的词向量，再加入CNN里训练即可
# bert是动态词向量，使用时还需要将bert的12层网络加载到CNN网络之前组合成一个新的网络一起训练


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from transformers import BertTokenizer, BertModel
import os
import numpy as np

import logging

import jieba

class TextCNN(nn.Module):
    def __init__(self, bert_embedding_dim, filter_num, sentence_max_size, label_size, kernel_list):
        super(TextCNN, self).__init__()
        chanel_num = 1
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(chanel_num, filter_num, (kernel, bert_embedding_dim)),
            nn.ReLU(),
            nn.MaxPool2d((sentence_max_size - kernel + 1, 1))
        )
            for kernel in kernel_list])
        self.fc = nn.Linear(filter_num * len(kernel_list), label_size)
        self.dropout = nn.Dropout(0.5)
        self.sm = nn.Softmax(0)

    def forward(self, x):
        in_size = x.size(0)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(in_size, -1)
        out = F.dropout(out)
        out = self.fc(out)
        return out

class MyDataset(Dataset):
    def __init__(self, file_list, label_list, sentence_max_size, bert_model, tokenizer, stopwords):
        self.x = file_list
        self.y = label_list
        self.sentence_max_size = sentence_max_size
        self.bert_model = bert_model # bert模型
        self.tokenizer = tokenizer # 执行分词任务和添加特殊标记
        self.stopwords = stopwords # 储存停用词（不重要的词）

    def __getitem__(self, index): # 特殊方法，用于获取一个样本
        # 读取内容
        words = []
        with open(self.x[index], "r", encoding="utf8") as file:
            for line in file.readlines():
                words.extend(segment(line.strip(), self.stopwords)) # 删除前后空白和停用词，并拼接成词列表
        # 生成词向量矩阵
        tensor = generate_bert_embeddings(words, self.sentence_max_size, self.bert_model, self.tokenizer)
        return tensor, self.y[index]

    def __len__(self):
        return len(self.x)

def generate_bert_embeddings(sentence, sentence_max_size, bert_model, tokenizer):
    inputs = tokenizer(sentence, padding='max_length', truncation=True, max_length=sentence_max_size, return_tensors='pt')
    print(sentence)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state
    return embeddings

def train_textcnn_model(net, train_loader, epoch, lr):
    print("begin training")
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for i in range(epoch):
        print(enumerate(train_loader))
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            logging.info("train epoch=" + str(i) + ",batch_id=" + str(batch_idx) + ",loss=" + str(loss.item() / 64))
    print('Finished Training')

def textcnn_model_train(net, train_loader):
    net.eval()
    correct = 0
    total = 0
    test_acc = 0.0
    with torch.no_grad():
        for i, (data, label) in enumerate(train_loader):
            logging.info("test batch_id=" + str(i))
            outputs = net(data)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        print('Accuracy of the network on train set: %d %%' % (100 * correct / total))
        return 100 * correct / total

def textcnn_model_test(net, test_loader):
    net.eval()
    correct = 0
    total = 0
    test_acc = 0.0
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            logging.info("test batch_id=" + str(i))
            outputs = net(data)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        print('Accuracy of the network on test set: %d %%' % (100 * correct / total))
        return 100 * correct / total

def load_stopwords(stopwords_dir):
    stopwords = []
    with open(stopwords_dir, "r", encoding="utf8") as file:
        for line in file.readlines():
            stopwords.append(line.strip())
    return stopwords

def segment(content, stopwords):
    res = []
    for word in jieba.cut(content):
        if word not in stopwords and word.strip() != "":
            res.append(word)
    return res

def get_file_list(source_dir): # 获取文件夹下所有文件的路径名
    file_list = []
    if os.path.isdir(source_dir):
        for root, dirs, files in os.walk(source_dir):
            file = [os.path.join(root, filename) for filename in files]
            file_list.extend(file)
        return file_list
    else:
        print("the path is not existed")
        exit(0)

def get_label_list(file_list):
    label_name_list = [file.split("\\")[-2] for file in file_list] # 按反斜杠进行划分，第二个元素代表类别
    label_list = []
    for label_name in label_name_list:
        if label_name == "neg":
            label_list.append(0)
        elif label_name == "pos":
            label_list.append(1)
    return label_list

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
    train_dir = os.path.join(os.getcwd(),"aclIdmb\\train") # 训练数据路径
    test_dir = os.path.join(os.getcwd(),"aclIdmb\\test") # 测试数据路径
    stopwords_dir = os.path.join(os.getcwd(),"stopwords.txt") # 停用词路径
    net_dir = ".\\model\\net.pkl" # 加载网络路径
    sentence_max_size = 300
    batch_size = 64
    filter_num = 100
    epoch = 1
    kernel_list = [3, 4, 5]
    label_size = 2
    lr = 0.001

    # Load BERT tokenizer and model 加载bert
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    bert_model = BertModel.from_pretrained('bert-base-cased')

    # Load stopwords 加载停用词
    stopwords = load_stopwords(stopwords_dir)

    # Get training data 获取训练数据
    train_set = get_file_list(train_dir) # 获取所有文件
    train_label = get_label_list(train_set) # 获取文件对应的标签
    train_dataset = MyDataset(train_set, train_label, sentence_max_size, bert_model, tokenizer, stopwords)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Get test data
    test_set = get_file_list(test_dir)
    test_label = get_label_list(test_set)
    test_dataset = MyDataset(test_set, test_label, sentence_max_size, bert_model, tokenizer, stopwords)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Define the model
    net = TextCNN(bert_model.config.hidden_size, filter_num, sentence_max_size, label_size, kernel_list)

    # Train the model
    logging.info("Start training the model")
    train_textcnn_model(net, train_dataloader, epoch, lr)
    torch.save(net, net_dir)
    
    # Test the model
    logging.info("Start testing the model")
    train_accuracy = textcnn_model_train(net, train_dataloader)
    test_accuracy = textcnn_model_test(net, test_dataloader)
    print('Accuracy of the network on train set: %d %%' % (train_accuracy))
    print('Accuracy of the network on test set: %d %%' % (test_accuracy))
