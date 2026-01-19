import time

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import utils


# 在这里定义你的模型，不要直接复制代码！
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()

        # 创建四个全连接层
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        
    def forward(self, x):
        # 第一层全连接后使用ReLU激活函数
        x = self.fc1(x)
        # x = F.leaky_relu(x)
        # 尝试用gelu激活函数
        x = F.gelu(x)
        #第二层全连接后使用gelu激活函数
        x = self.fc2(x)
        x = F.gelu(x)   
        #第三层全连接后使用gelu激活函数
        x = self.fc3(x)
        x = F.gelu(x)
        # 第四层全连接后直接输出
        return self.fc4(x)


if __name__ == '__main__':
    # 在这里读入不同的 csv 文件
    X, Y = utils.read_csv_data('1-2-homework\data_666.csv')

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    print(X.shape, Y.shape)

    # 根据特征维度，绘制 2D 或 3D 散点图
    utils.draw_2d_scatter(X, Y)
    # utils.draw_3d_scatter(X, Y)

    # 在这里开始你的表演，不要直接复制代码！
    # 模型、损失函数、优化器
    model = SimpleMLP()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

    # 创建数据集和加载器
    dataset = TensorDataset(X, Y)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 训练模型
    start_time = time.time()
    epochs = 500
    for epoch in range(epochs):
        epoch_loss = 0
    
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
    print(f"训练时间: {end_time - start_time:.2f} 秒")

    # 查看预测效果
    predicted = model(X)
    utils.draw_2d_scatter(X, Y, predicted.detach().numpy())
    # utils.draw_3d_scatter(X, Y, predicted.detach().numpy())
