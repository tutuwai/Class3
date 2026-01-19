import torch
import torch.nn as nn
# optim 是 PyTorch 中用于优化模型参数的模块
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

import utils

# 定义线性回归模型，继承自 nn.Module
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        # 定义一个线性层
        self.linear = nn.Linear(2, 1)  # 输入维度为2，输出维度为1
    
    def forward(self, x):
        # 前向传播函数
        return self.linear(x)

if __name__ == "__main__":
    # 指定分布的面
    W_ture = np.array([0.8, -0.4])
    b_ture = -0.2
    
    # 生成数据集
    data_size = 200
    X, Y = utils.make_linear_data(W_ture, b_ture, data_size)
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    utils.draw_3d_scatter(X, Y)
    
    # 初始化模型
    model = LinearRegressionModel()
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-5)
    
    # 指定一些超参数
    epochs = 50
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        for i in range(data_size):
            # 取单独的一个样本点
            x_i = torch.tensor(X[i], dtype=torch.float32) # shape=(1,2)
            y_i = torch.tensor(Y[i], dtype=torch.float32) # shape=(1,1)
            
            # 前向传播
            y_pred = model(x_i)  # shape=(1,1)
            
            # 计算损失
            loss = criterion(y_pred, y_i)
            
            # 反向传播和优化
            optimizer.zero_grad()  # 清零梯度
            loss.backward()        # 反向传播
            optimizer.step()       # 更新参数
            # 累计损失
            epoch_loss += loss.item()
        print(f"第{epoch+1}轮次：损失={epoch_loss/data_size}")
    
    # 训练完成，打印参数
    print(f'W_true: {W_ture}, W: {model.linear.weight.detach().numpy().flatten()}')
    print(f'b_true: {b_ture}, b: {model.linear.bias.item()}')
    
    # 绘制拟合结果
    y_hat = model(torch.tensor(X, dtype=torch.float32)).detach().numpy()
    utils.draw_3d_scatter(X, Y, y_hat)
