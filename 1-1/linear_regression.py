import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 构建人工数据集
X = torch.linspace(-10,10 ,100).unsqueeze(1)

#生成y
Y = 3*X + 2 + torch.randn(X.size())*2

# 画图
plt.scatter(X.numpy(),Y.numpy())
plt.show()

# 定义线性回归模型
# 这里括号内的nn.Module表示继承自pytorch的Module类
# 如果写道__init__()的内容则是参数初始化的内容
class LinearRegressionModel(nn.Module):
    def __init__(self):
        # 调用父类的构造函数（python面向对象的内容）
        super().__init__()
        # 定义线性层，输入和输出都是1维（核心）
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)
# 实例化模型
model = LinearRegressionModel()

# 定义损失函数和优化器
#这个MSELoss已经帮我们实现了均方误差损失函数
criterion = nn.MSELoss()
# SGD随机梯度下降优化器
# parameters()会把model的所有参数传进去
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 训练模型
epochs = 2000
for epoch in range(epochs):
    # 前向传播
    y_pred = model(X)
    # 计算损失
    loss = criterion(y_pred, Y)
    # 反向传播和优化
    optimizer.zero_grad()  # 清零梯度
    loss.backward()        # 反向传播计算梯度
    optimizer.step()       # 更新参数
    # 每10个epoch打印一次损失
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 训练后查看参数
w = model.linear.weight.item()
b = model.linear.bias.item()
print(f"训练后参数: w={w}, b={b}")
print("实际参数: w=3, b=2")
# 查看结果画出拟合结果
y_pred = model(X)
# 真实数据
plt.scatter(X.numpy(),Y.numpy(),label='real data')
# 拟合直线
plt.plot(X.numpy(),y_pred.detach().numpy(),color='r',label='Linear Regression Fit')
plt.xlabel('X')
plt.ylabel('y') 
plt.legend()
plt.title(f'Linear Regression: y={w:.2f}x + {b:.2f}')
plt.show()