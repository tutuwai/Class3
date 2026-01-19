import torch
import numpy as np

import utils

if __name__ == "__main__":
    # 指定分布的面
    W_ture = np.array([0.8, -0.4])
    b_ture = -0.2
    
    # 生成数据集
    data_size = 1000
    X, Y = utils.make_linear_data(W_ture, b_ture, data_size)
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    
    #初始化参数
    #torch.normal(mean=平均值, std=标准差, size=形状， requires_grad=是否需要梯度)
    w = torch.normal(mean=0.0, std=1.0, size=(2,1), requires_grad=True,dtype=torch.float32)
    b = torch.zeros(1, requires_grad=True,dtype=torch.float32)
    
    # 指定一些超参数
    lr = 1e-5
    epochs = 1000
    
    for epoch in range(epochs):
        # Batch 的方法
        # #前向传播
        # y_pred = torch.matmul(torch.from_numpy(X).float(), w) + b  # X.shape=(num,2), w.shape=(2,1)
        
        # #计算均方误差损失
        # loss = torch.mean((torch.from_numpy(Y).float() - y_pred)**2)
        
        # #反向传播
        # loss.backward()
        
        # #更新参数
        # with torch.no_grad():
        #     w -= lr * w.grad
        #     b -= lr * b.grad
            
        #     #清零梯度
        #     w.grad.zero_()
        #     b.grad.zero_()
        
        # if (epoch+1) % 100 == 0:
        #     print(f"第{epoch+1}轮次：损失={loss.item()}")
        # Stochastic Gradient Descent (SGD) 方法
        
        #本轮次的累计损失
        epoch_loss = 0.0
        
        for i in range(data_size):
            #取单独的一个样本点
            x_i = torch.tensor(X[i],dtype= torch.float32)
            y_i = torch.tensor(Y[i],dtype= torch.float32)
            #前向传播 1*1
            y_pred = x_i @ w + b  # x_i.shape=(1,2), w.shape=(2,1)
            #计算均方误差损失
            loss = (y_i - y_pred)**2  # shape=(1,1)
            #反向传播
            #求梯度
            loss.backward()
            #更新参数
            w.data -= lr * w.grad
            b.data -= lr * b.grad
            #清零梯度
            w.grad.zero_()          
            b.grad.zero_()
            #累计损失
            epoch_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"第{epoch+1}轮次：损失={epoch_loss/data_size}")
            
    print(f"训练后参数: w={w.reshape(-1).data.numpy()}, b={b.item()}")
    print(f"实际参数: w={W_ture.reshape(-1)}, b={b_ture}")
    # 查看结果画出拟合结果.detach() 返回一个新的tensor,与原tensor共享数据内存,但不会计算梯度
    # .numpy() 将tensor转换为numpy数组
    #.item() 将只有一个元素的tensor转换为对应的python数值
    y_hat = X @ w.detach().numpy() + b.item()
    utils.draw_3d_scatter(X, Y, y_hat)