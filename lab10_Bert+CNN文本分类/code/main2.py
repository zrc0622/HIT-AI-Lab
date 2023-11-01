import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import os
import numpy as np
import logging
import jieba
import math
from tqdm import tqdm

# TextCNN模型
class TextCNNWithBert(nn.Module):
    def __init__(self, bert_model, filter_num, sentence_max_size, label_size, kernel_list):
        super(TextCNNWithBert, self).__init__()
        self.bert_model = bert_model # bert模型
        for param in self.bert_model.parameters(): # 遍历了self.bert_model中的所有参数
            param.requires_grad = False # 在反向传播（backpropagation）过程中不会计算这些参数的梯度，从而使它们不会被更新
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1, filter_num, (kernel, bert_model.config.hidden_size)), # 1个通道；filter_num个卷积核；卷积核大小为（kernel,词向量长度）
            nn.ReLU(),
            nn.MaxPool2d((sentence_max_size - kernel + 1, 1))
        ) for kernel in kernel_list])
        self.fc = nn.Linear(filter_num * len(kernel_list), label_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert_model(input_ids, attention_mask) # 输入句子的ids（batch_size，300）和注意力矩阵（batch_size,300）
        bert_embeddings = bert_outputs.last_hidden_state # 输出句子的embedding（batch_size,300,768）
        out = [conv(bert_embeddings.unsqueeze(1)) for conv in self.convs] # 将embedding由（batch_size,300,768）变为（batch_size,1,300,768） （假设filter num=2）:卷积（每种卷积核有两个，所以输出管道为2）：2:(64,2,299,1)  3:(64,2,298,1)  4:(64,2,297,1)   池化：2:(64,2,1,1)  3:(64,2,1,1)  4:(64,2,1,1) (?-2/3/4+1实际上就是sentence_max_size-kernel+1)
        out = torch.cat(out, dim=1) # (64,6,1,1)
        out = out.view(out.size(0), -1)  # (64,6)
        out = self.fc(out)
        return out

# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, file_list, label_list, sentence_max_size, bert_model, tokenizer, stopwords):
        self.x = file_list # 训练文件
        self.y = label_list # 文件对应标签
        self.sentence_max_size = sentence_max_size
        self.tokenizer = tokenizer
        self.stopwords = stopwords

    def __getitem__(self, index):
        words = []
        with open(self.x[index], "r", encoding="utf8") as file:
            for line in file.readlines():
                words.extend(segment(line.strip(), self.stopwords))  # 删除前后空白和停用词，并拼接成词列表
        input_ids, attention_mask = generate_bert_embeddings(words, self.sentence_max_size, self.tokenizer) # 返回embedding(1,300,768)
        return input_ids, attention_mask, self.y[index] # 返回句子词向量和对应标签

    def __len__(self):
        return len(self.x)

# 生成BERT词向量函数
def generate_bert_embeddings(sentence, sentence_max_size, tokenizer):
    sentence_text = " ".join(sentence) # 将词列表还原为句子
    inputs = tokenizer(sentence_text, padding='max_length', truncation=True, max_length=sentence_max_size, return_tensors='pt') # 将句子中的词转为bert中的id，并将句子扩充为300长度，再添加注意力矩阵，告诉哪些部分是填充的
    # with torch.no_grad():
    #     outputs = bert_model(**inputs) # inputs是字典，将字典解码为关键词（句子中词的id）和键值（注意力矩阵）作为输入
    # embeddings = outputs.last_hidden_state # 句子的embedding（300，767），每个词的词向量长度为767，共300个词
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    return input_ids, attention_mask

# 训练TextCNNWithBert模型
def train_textcnn_with_bert_model(net, train_loader, epoch, lr):
    print("开始训练")
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    total_samples = len(train_dataset)
    total_batches = math.ceil(total_samples / batch_size)  
    for i in range(epoch):
        # with tqdm(total=total_batches, desc="Processing") as pbar:
        for batch_idx, (input_ids, attention_mask, target) in enumerate(train_loader):
            # print('3'*100)
            if torch.cuda.is_available():
                input_ids = input_ids.to('cuda')
                attention_mask = attention_mask.to('cuda')
                target = target.to('cuda') # 数据在train循环内放入GPU中
                print("data in GPU")
            optimizer.zero_grad()
            # print(input_ids.shape)
            output = net(input_ids.squeeze(), attention_mask.squeeze()) # 将ids由（batch_size,1,300）变为（batch_size，300）
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            logging.info(f"训练 - epoch={i}, batch_id={batch_idx}, loss={loss.item()}")
            # pbar.update(1)
    print('训练完成')

# 测试TextCNNWithBert模型
def test_textcnn_with_bert_model(net, test_loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, attention_mask, target in test_loader:
            # input_ids 是输入的 BERT token IDs
            # attention_mask 是注意力掩码
            # target 是目标标签
            input_ids=input_ids.to('cuda')
            attention_mask=attention_mask.to('cuda')
            target=target.to('cuda')
            outputs = net(input_ids.squeeze(), attention_mask.squeeze())
            
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            if total>5000:
                break    
    accuracy = 100 * correct / total
    return accuracy

def calculate_train_accuracy(net, train_loader):
    net.eval()  # 设置模型为评估模式

    correct = 0
    total = 0

    with torch.no_grad():
        for input_ids, attention_mask, target in train_loader:
            # 将数据传递到模型中进行预测
            input_ids=input_ids.to('cuda')
            attention_mask=attention_mask.to('cuda')
            target=target.to('cuda')
            outputs = net(input_ids.squeeze(), attention_mask.squeeze())
            _, predicted = torch.max(outputs.data, 1)

            total += target.size(0)
            correct += (predicted == target).sum().item()
            if total>5000:
                break  

    train_accuracy = 100 * correct / total
    return train_accuracy


# 加载停用词
def load_stopwords(stopwords_dir):
    stopwords = []
    with open(stopwords_dir, "r", encoding="utf8") as file:
        for line in file.readlines():
            stopwords.append(line.strip())
    return stopwords

# 分词函数
def segment(content, stopwords):
    res = []
    for word in jieba.cut(content):
        if word not in stopwords and word.strip() != "":
            res.append(word)
    return res

# 获取文件列表和标签列表的函数
def get_file_list(source_dir):
    file_list = []
    if os.path.isdir(source_dir):
        for root, dirs, files in os.walk(source_dir):
            file = [os.path.join(root, filename) for filename in files]
            file_list.extend(file)
        return file_list
    else:
        print("路径不存在")
        exit(0)

def get_label_list(file_list):
    label_name_list = [file.split("\\")[-2] for file in file_list]
    label_list = []
    for label_name in label_name_list:
        if label_name == "neg":
            label_list.append(0)
        elif label_name == "pos":
            label_list.append(1)
    return label_list

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
    train_dir = os.path.join(os.getcwd(), "aclIdmb\\train") # 训练数据路径
    test_dir = os.path.join(os.getcwd(), "aclIdmb\\test") # 测试数据路径
    stopwords_dir = os.path.join(os.getcwd(), "stopwords.txt") # 停用词路径
    sentence_max_size = 300
    batch_size = 64
    filter_num = 100
    epoch = 1
    kernel_list = [3, 4, 5]
    label_size = 2
    lr = 0.001

    # 加载BERT模型和停用词表
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    bert_model = BertModel.from_pretrained('bert-base-cased')

    # 加载停用词
    stopwords = load_stopwords(stopwords_dir)

    # 获取训练数据
    train_set = get_file_list(train_dir) # 获取所有的训练数据文件路径
    train_label = get_label_list(train_set) # 获取训练数据对应的标签
    train_dataset = MyDataset(train_set, train_label, sentence_max_size, bert_model, tokenizer, stopwords) # 包含词向量+标签
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # 创建训练数据加载器


    # 获取测试数据
    test_set = get_file_list(test_dir)
    test_label = get_label_list(test_set)
    test_dataset = MyDataset(test_set, test_label, sentence_max_size, bert_model, tokenizer, stopwords)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # 创建TextCNNWithBert模型
    textcnn_with_bert = TextCNNWithBert(bert_model, filter_num, sentence_max_size, label_size, kernel_list)
    if torch.cuda.is_available():
        textcnn_with_bert = textcnn_with_bert.to('cuda')
        print("successfully running in GPU") # 定义网络的时候放入GPU中

    # 训练TextCNNWithBert模型
    train_textcnn_with_bert_model(textcnn_with_bert, train_dataloader, epoch, lr)

    # 测试TextCNNWithBert模型
    # 在训练完成后调用这个函数来计算训练集的准确率
    train_accuracy = calculate_train_accuracy(textcnn_with_bert, train_dataloader)
    print(f'模型在训练集上的准确率: {train_accuracy:.2f}%')
    test_accuracy = test_textcnn_with_bert_model(textcnn_with_bert, test_dataloader)
    print(f'模型在测试集上的准确率: {test_accuracy:.2f}%')
