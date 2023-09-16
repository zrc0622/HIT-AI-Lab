import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset


# 读取数据（csv文件就是文本文件，但是不同的列之间以','作为分割）
data = pd.read_csv('./data/ionosphere.data', delimiter = ',', header=None) # 读取数据集，以逗号作为数据的分割，没有标题行；读取到的格式为pandas.core.frame.DataFrame，变为张量前需要先转为numpy数组
X = data.iloc[:, :-1] # 输入的集合，前34列为输入（X.shape = (351, 34)）
Y = data.iloc[:, -1] # 输出的集合，最后一列为输出（Y.shape = (351,1)）
Y = np.where(Y == 'g', 1, 0) # 重塑输出，good为1，bad为0，以处理二分类问题（此步骤已经将转为numpy数组了）

# 将numpy数组转为gpu能处理的pytorch张量
X = X.to_numpy() # X为pandas.core.frame.DataFrame，数据中有int有float，需要变为numpy数组，统一数据格式
tensor_X = torch.FloatTensor(X) # X转为张量
tensor_Y = torch.FloatTensor(Y) # Y转为张量

# 划分数据
X_train, X_test, Y_train, Y_test = train_test_split(tensor_X, tensor_Y, test_size=0.2, random_state=42) # 将数据划分为训练集和测试集，测试集用于评估模型，test_size表示测试集的比例

# 创建数据集和数据加载器
train_dataset = TensorDataset(X_train, Y_train) # 将输入数据X_train和对应的标签数据Y_train组合成一个数据集
batch_size = 32  # 设置批量大小
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # 使用DataLoader创建一个数据加载器train_loader。DataLoader是PyTorch提供的用于加载数据的工具，它可以根据批量大小从数据集中创建小批量数据，并且可以进行随机打乱

# 搭建MLP网络
class MLP(nn.Module): # MLP网络的python类，以继承troch.nn.Module并重写__init__和forward函数的⽅式创建⼀个MLP网络
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size): # MLP类的构造函数，初始化各层和参数
        super(MLP, self).__init__() # 继承自父类troch.nn.Module的构造
        self.fc1 = nn.Linear(input_size, hidden_size1) # 输入到第一个隐藏层的全连接层
        self.fc2 = nn.Linear(hidden_size1, hidden_size2) # 第一个隐藏层到第二个隐藏层的全连接层
        self.fc3 = nn.Linear(hidden_size2, output_size) # 第二个隐藏层到输出的全连接层

    def forward(self, x): # 此方法用于定义前向传播操作
        x = F.relu(self.fc1(x)) # 将输入数据x传递给第一个全连接层self.fc1，然后通过ReLU激活函数F.relu
        x = F.relu(self.fc2(x)) # 将第一个隐藏层的输出传递给第二个隐藏层，并再次应用ReLU激活函数
        x = torch.sigmoid(self.fc3(x)) # 将第二个隐藏层的输出传递给输出层，并应用Sigmoid激活函数，将输出映射到[0, 1]之间的概率值
        return x

# 定义模型和优化器
input_size = X_train.shape[1] # 输入维度（34个自变量，34维）
hidden_size1 = 128 # 第一个隐藏层的神经元数（神经元是隐藏层，包括输入和输出；线是全连接层；插入的激活函数在隐藏层的输入输出之间）
hidden_size2 = 64 # 第二个隐藏层的神经元数
output_size = 1 # 输出维度（二分类问题，一维）

model = MLP(input_size, hidden_size1, hidden_size2, output_size) # 搭建网络
criterion = nn.BCELoss() # 损失函数为BCE函数
optimizer = optim.SGD(model.parameters(), lr=0.01) # 优化器为SGD

# 训练模型
num_epochs = 1000 # 训练的总episode（训练周期）；一个batch相当于一批样本从头到尾训练一次（模型训练的最小单元）；一个epoch相当于把所有的样本都从头到尾训练一次，也相当于执行多个batch直到所有样本都训练过一次（几批不重复的样本（几个batch）合在一起叫epoch）（batch_size完全靠随机性的）
with tqdm(total=num_epochs, desc="Processing") as pbar: # 训练进度条
    for epoch in range(num_epochs): 
        optimizer.zero_grad() # 优化器梯度清零
        for batch_X, batch_Y in train_loader:  # 遍历每个小批量数据
            outputs = model(X_train) # 将训练数据传入MLP网络，并获取MLP网络输出
            loss = criterion(outputs, Y_train.view(-1, 1))  # 通过MLP输出和期望输出计算损失，目标是最小化损失值（将y变成列向量以匹配期望输出形状）
            loss.backward() # 执行反向传播，计算损失相对于模型参数的梯度，以便在优化器中更新模型参数
            optimizer.step() # 使用优化器来更新模型的参数。优化器根据计算得到的梯度信息，更新模型的权重和偏差，以减小损失函数的值
        pbar.update(1) # 进度条+1
        if (epoch + 1) % 5000 == 0: # 每100个epoch输出一次loss
            print()
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

# 在封闭测试集上进行预测
model.eval() # 评估当前网络
with torch.no_grad(): # 在其内部的操作中禁用梯度计算
    Y_pred = model(X_train) # 使用训练好的模型对训练数据 X_train 进行前向传播，以获取模型的预测结果 Y_pred

# 将预测值转换为 NumPy 数组
Y_pred = (Y_pred > 0.5).numpy() # 将模型的预测结果 Y_pred 转换为NumPy数组。通常，在二分类问题中，模型的输出可以被解释为概率值，然后可以根据一个阈值（通常是0.5）来判断样本属于哪个类别。这一行代码将大于0.5的值设为1，小于等于0.5的值设为0，将二分类的概率结果转化为二元标签

# 计算准确率
accuracy = accuracy_score(Y_train.numpy(), Y_pred) # 计算模型在训练集上的准确率
print(" closed accuracy:", accuracy) # 输出封闭准确率

# 在开放测试集上进行预测
model.eval() # 评估当前网络
with torch.no_grad(): # 在其内部的操作中禁用梯度计算
    Y_pred = model(X_test) # 使用训练好的模型对测试数据 X_test 进行前向传播，以获取模型的预测结果 Y_pred

# 将预测值转换为 NumPy 数组
Y_pred = (Y_pred > 0.5).numpy() # 将模型的预测结果 Y_pred 转换为NumPy数组。通常，在二分类问题中，模型的输出可以被解释为概率值，然后可以根据一个阈值（通常是0.5）来判断样本属于哪个类别。这一行代码将大于0.5的值设为1，小于等于0.5的值设为0，将二分类的概率结果转化为二元标签

# 计算准确率
accuracy = accuracy_score(Y_test.numpy(), Y_pred) # 计算模型在测试集上的准确率
print("opened accuracy:", accuracy) # 输出开放准确率