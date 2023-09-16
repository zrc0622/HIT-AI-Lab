import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

# 处理手写阿拉伯数字
# 每个数字是28*28的矩阵（共60000个数字），如果像素点是白则为0，如果不是白则根据黑的程度赋值
# 卷积核kernel相当于窗函数，对原始矩阵进行加权求和
# 输入数据的标签：图片的标签以一维数组的one-hot编码形式给出，[0,0,0,0,0,1,0,0,0,0]，每个元素表示图片对应的数字出现的概率，显然，该向量标签表示的是数字5 55。

# 步骤1：加载MNIST数据集
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) # 定义了一个预处理操作列表：先将数据转为tensor张量，再执行标准化操作（x-0.5）/0.5
transform = transforms.Compose([transforms.ToTensor()]) # 定义了一个预处理操作列表：将数据集转为tensor张量
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform) # 从./data文件夹中获取数据和标签，trian表示训练数据，进行在线下载，同时进行预处理 
trainloader = DataLoader(trainset, batch_size=512, shuffle=True) # 将预处理后的数据转为trainloader，shuffle表示随机取batch，trian的数据集中需要随机
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform) # 从./data文件夹中获取数据和标签，trian false表示测试数据，进行在线下载，同时进行预处理 
testloader = DataLoader(testset, batch_size=1024, shuffle=False) # 测试数据集的batchsize可以放大一些，加快测试速度
weidu = 64*5*5

# 步骤2：构建CNN网络
class CNN(nn.Module): # cnn网络继承自nn.module
    def __init__(self): # 初始化
        super(CNN, self).__init__() # 继承操作，必要的
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3) # 第一层卷积层，conv2d表示二维卷积层（常用于图像处理）（输入1维：1表示输入通道数量，输出32维：32表示输出feature map数量，卷积核2*2）
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3) # 第二层卷积层（输入32维，输出64维，卷积核2*2）（有32个图像输入，有64个卷积核进行处理，每个卷积核的通道数为32（相当于每个卷积核由32个卷积向量构成），输出的feature map为64）
        self.fc1 = nn.Linear(weidu, 128) # 第一层全连接层
        self.fc2 = nn.Linear(128, 10) # 第二层全连接层，输出对应0-9的概率
        # 输入a，卷积核x，步幅y，填充z，则输出a+2*z-x+y，默认步幅1，填充0 
        
    def forward(self, x):
        x = torch.relu(self.conv1(x)) # 输入图像先进入第一层卷积层后经过激活函数 (256,28,28,1)->(256,27,27,32)->(256,27,27,32)
        x = torch.max_pool2d(x, 2) # 池化操作，将图像划分为2*2的小区域，每一个区域选择最大值 (256,27,27,32)->(256,13,13,32)（向下取整）
        x = torch.relu(self.conv2(x)) # 进入第二层卷积层后经过激活函数 (256,13,13,32)->(256,12,12,64)->(256,12,12,64)
        x = torch.max_pool2d(x, 2) # 池化操作，将图像划分为2*2的小区域，每一个区域选择最大值 (256,12,12,64)->(256,6,6,64)
        x = x.view(-1, weidu) # 调整张量（Tensor）的形状，-1表示维度自动计算（等于batch_size）（相当于把64个6*6的feature map的像素单独抽出来放到全连接层中）
        x = torch.relu(self.fc1(x)) # 64*6*6个像素输入到第一个全连接层，随后经过激活函数
        x = self.fc2(x) # 随后输出到第二个全连接层
        # (256,28,28,1)
        # 第一层：卷积层，32个卷积核，每个卷积核1通道(每个通道4个参数(卷积核2*2大小))，每个卷积核后为激活函数和最大池化函数
        # --卷积+激活-->  (256,27,27,32) --池化--> (256,13,13,32)
        # 第二层：卷积层，64个卷积核，每个卷积核32通道，每个卷积核后为激活函数和最大池化函数
        # --卷积+激活-->  (256,12,12,64) --池化--> (256,6,6,64) --reshape--> (256,6*6*64)
        # 第三层：全连接层，128个神经元，每条线2个参数(权重和偏置)，每个神经元后为激活函数
        # 第四层：全连接层，10个神经元
        # (256,6*6*64) ----> (256,10)
        return x

CNN = CNN()

# 步骤3：定义损失函数和优化器
criterion = nn.CrossEntropyLoss() # 交叉熵损失（Cross-Entropy Loss）通常用于多类别分类任务。这个损失函数用于度量模型输出和真实标签之间的差异，以便通过反向传播来调整神经网络的权重，使损失最小化。在多类别分类任务中，交叉熵损失是一个常用的选择
optimizer = optim.SGD(CNN.parameters(), lr=0.01, momentum=0.9) # 这行代码定义了优化器（optimizer），使用了随机梯度下降（Stochastic Gradient Descent，SGD）优化算法

# 步骤4：训练模型
for epoch in range(50):  # 迭代50次
    running_loss = 0.0 # 初始化损失
    for i, data in enumerate(trainloader, 0): # i用于遍历batch（起始索引值为0）
        inputs, labels = data # data取出当前batch的输入和标签
        optimizer.zero_grad() # 梯度清零
        outputs = CNN(inputs) # 从网络中获取输出
        loss = criterion(outputs, labels) # 计算损失
        loss.backward() # 梯度回传
        optimizer.step() # 优化操作
        running_loss += loss.item() # 计算损失
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}") # 输出平均损失

print("Finished Training")

# 步骤5：计算训练精度和测试精度
correct_train = 0 # 训练数据上正确分类的样本数量
total_train = 0 # 训练数据上总样本数量
correct_test = 0 # 测试数据上正确分类的样本数量
total_test = 0 # 测试数据上总样本数量

# max函数
with torch.no_grad(): # 计算精度时禁用梯度计算。在精度计算过程中，我们不需要计算梯度，因为这只是一个评估阶段，而不是训练阶段。禁用梯度计算可以提高性能和减少内存消耗
    for data in trainloader: # 遍历训练数据集中的每个批次
        inputs, labels = data # 从数据批次中解包输入数据和对应的标签
        outputs = CNN(inputs) # 使用神经网络模型 CNN 对输入数据进行前向传播，得到预测输出
        _, predicted = torch.max(outputs.data, 1) # 使用 torch.max 函数找到每个样本的最大预测值和对应的类别索引，然后将这些索引存储在 predicted 变量中
        # torch.max(outputs.data, 1) 是 PyTorch 中的一个函数，用于计算一个张量的指定维度上的最大值。在这种情况下，它在 outputs.data 张量的维度 1 上找到最大值。
        # torch.max(outputs.data, 1) 的结果是一个包含两个元素的元组：第一个元素（通常用 _ 表示）是在维度 1 上找到的最大值。第二个元素（通常用 predicted 表示）是找到这个最大值所在的索引，即预测的类别标签。
        total_train += labels.size(0) # labels.size(0)表示当前批次的大小（当前批次的标签数量）
        correct_train += (predicted == labels).sum().item() # 如果预测正确则累加（.item()表示转为标量）

    for data in testloader: # 同上
        inputs, labels = data
        outputs = CNN(inputs)
        _, predicted = torch.max(outputs.data, 1) 
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

print(f"Training Accuracy: {100 * correct_train / total_train}%") # 输出封闭精度
print(f"Testing Accuracy: {100 * correct_test / total_test}%") # 输出开放精度
