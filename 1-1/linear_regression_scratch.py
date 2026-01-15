import torch
import matplotlib.pyplot as plt

# 构建人工数据集
# 假设真实的数据关系是y=3x + 2，并添加一些噪声

#生成-10，10，100个点
#unsqueeze(dim) = "在第 dim 维度处插入一个大小为1的新维度"
#比如 现在100 的张量  unsqueeze(1) 变成 100*1 的张量,这样做的目的是为了后面的矩阵运算
X = torch.linspace(-10, 10, 100).unsqueeze(1)

#生成噪声 torch.randn(X.size()) 生成和X同样大小的标准正态分布噪声
noise = torch.randn(X.size())*2

#生成y
y =  3*X + 2 + noise

#画图
plt.scatter(X.numpy(),y.numpy())
plt.show()

# 定义线性回归模型,设置requires_grad=True表示需要计算梯度，这里是一个Tensor类型，numpy不支持自动求导
w = torch.randn(1,requires_grad=True)  # 权重
b = torch.randn(1,requires_grad=True)    # 偏置

print(f"初始参数: w={w.item()}, b={b.item()}")

# 定义前向传播函数
def forword(x):
    return x*w + b
# 定义损失函数(均方误差)
def mse_loss(pred, true):
    return ((pred - true)**2).mean()

# 训练模型
learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):
    # 前向传播
    y_pred = forword(X)
    # 计算损失
    mse_loss_value = mse_loss(y_pred,y)
    # 反向传播优化w和b
    # 计算梯度，这里会使得每个Tensor参数对mse_loss_value的grad属性被更新,即 w.grad 和 b.grad
    mse_loss_value.backward()
    
    # 更新参数 - 需要用 torch.no_grad() 包裹，否则会破坏梯度链
    # 这里写w = w - learning_rate * w.grad 会报错
    # 因为右侧的w - learning_rate * w.grad 创建了一个新的Tensor
    # 这个新的Tensor默认 requires_grad=False，所以不能对它继续调用backward()
    # 它变成了非叶子节点，后续再调用backward()会报错
    # 可以用with torch.no_grad() 来禁止梯度计算
    # 如下是最简单的写法
    w.data -= learning_rate * w.grad
    b.data -= learning_rate * b.grad
    
    # 清零梯度,否则每次backward会累加梯度
    w.grad.zero_()
    b.grad.zero_()
    
    # 每隔10轮次打印一次损失
    if (epoch+1) % 10 == 0:
        print(f"第{epoch+1}轮次：损失={mse_loss_value.item()}")

print(f"训练后参数: w={w.item()}, b={b.item()}")
print("实际参数: w=3, b=2")
# 查看结果画出拟合结果
y_pred = forword(X)
# 真实数据
plt.scatter(X.numpy(),y.numpy(),label='real data')
# 拟合直线
plt.plot(X.numpy(),y_pred.detach().numpy(),color='r',label='Linear Regression Fit')
plt.xlabel('X')
plt.ylabel('y') 
plt.legend()
plt.title(f'Linear Regression: y={w.item():.2f}x + {b.item():.2f}')
plt.show()