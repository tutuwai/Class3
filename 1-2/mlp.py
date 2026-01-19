from torch import nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import utils
import time

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()

        # 创建两个全连接层（单隐藏层，降低复杂度）
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # 只使用第一层全连接
        # return self.fc1(x)
        
        x = self.fc1(x)
        
        # ===== 不同激活函数对比 =====
        # 注释/取消注释下面的激活函数来观察不同激活函数的训练效果
        
        # 1. ReLU - 最常用，计算快，容易出现Dead ReLU问题
        # x = F.relu(x)
        
        # 2. Sigmoid - 输出范围(0,1)，容易梯度消失
        # x = torch.sigmoid(x)
        
        # 3. Tanh - 输出范围(-1,1)，容易梯度消失
        # x = torch.tanh(x)
        
        # 4. LeakyReLU - ReLU的改进版，避免Dead ReLU
        # x = F.leaky_relu(x, negative_slope=0.01)
        
        # 5. ELU - 光滑的非线性激活函数
        # x = F.elu(x)
        
        # 6. GELU - 新型激活函数，在Transformer中常用
        x = F.gelu(x)
        
        
        # ===== 激活函数对比结束 =====

        # 输出层
        return self.fc2(x)
    
if __name__ == '__main__':
    # 读入数据、转为 tensor、绘制
    X, Y = utils.read_csv_data('1-2\data_2.csv')
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    print(X.shape, Y.shape)
    utils.draw_2d_scatter(X, Y)
    
    # 模型、损失函数、优化器
    model = SimpleMLP()
    criterion = nn.MSELoss()
    # 优化器是 Adam，作用是自动调整学习率，0.1 是初始学习率，最大不超过 0.1,weight_decay=1e-4 是L2正则化参数
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 创建数据集和加载器
    dataset = TensorDataset(X, Y)
    # batch_size 可以调整，代表了每次训练使用多少样本
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    start_time = time.time()
    # 训练模型
    epochs = 200
    for epoch in range(epochs):
        epoch_loss = 0
        
        # 这里batch_x的形状是 (batch_size, 1)，batch_y的形状是 (batch_size, 1)
        for batch_x, batch_y in data_loader:
            # 预测输出、计算损失
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            # 计算梯度、更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 累积损失
            epoch_loss += loss.item()

        # 打印本轮的损失值
        print(f'Epoch {epoch}, Loss: {epoch_loss / len(data_loader)}')
    end_time = time.time()
    print(f"训练时间: {end_time - start_time} 秒")
    # 使用训练好的模型预测
    predicted = model(X)
    utils.draw_2d_scatter(X, Y, predicted.detach().numpy())
