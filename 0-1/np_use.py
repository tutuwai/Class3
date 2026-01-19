import numpy as np
import sys

# =====================================================
# NumPy 基础用法示例
# =====================================================

def print_section(title):
    """打印章节标题"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_example(name, description):
    """打印示例标题"""
    print(f"\n【示例】{name}")
    print(f"说明: {description}")
    print("-" * 50)

# =====================================================
# 1. 数组创建
# =====================================================
print_section("1. 数组创建")

print_example("创建基本数组", "从Python列表创建NumPy数组")
arr1 = np.array([1, 2, 3, 4, 5])
print(f"一维数组: {arr1}")
print(f"数组类型: {type(arr1)}, 元素类型: {arr1.dtype}")

print_example("创建二维数组", "从嵌套列表创建矩阵")
arr2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"二维数组:\n{arr2}")
print(f"形状(shape): {arr2.shape}, 维度(ndim): {arr2.ndim}")

print_example("zeros函数", "创建全0数组")
zeros = np.zeros((3, 4))
print(f"3x4的零矩阵:\n{zeros}")

print_example("ones函数", "创建全1数组")
ones = np.ones((2, 3), dtype=int)
print(f"2x3的1矩阵:\n{ones}")

print_example("arange函数", "创建等差数列（类似range）")
range_arr = np.arange(0, 10, 2)
print(f"从0到10,步长为2: {range_arr}")    

print_example("linspace函数", "创建等间距数列（指定元素个数）")
linspace_arr = np.linspace(0, 10, 5)
print(f"从0到10,均匀分成5个数: {linspace_arr}")

print_example("random函数", "创建随机数组")
random_arr = np.random.rand(2, 3)
print(f"2x3的随机数组:\n{random_arr}")

print_example("eye函数", "创建单位矩阵（对角线为1）")
eye_mat = np.eye(3)
print(f"3x3单位矩阵:\n{eye_mat}")

# =====================================================
# 2. 数组属性
# =====================================================
print_section("2. 数组属性")

test_arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

print_example("shape属性", "获取数组的形状")
print(f"数组:\n{test_arr}")
print(f"形状: {test_arr.shape} (3行4列)")

print_example("size属性", "获取数组中元素总数")
print(f"元素总数: {test_arr.size}")

print_example("dtype属性", "获取数组元素的数据类型")
print(f"数据类型: {test_arr.dtype}")

print_example("itemsize属性", "获取每个元素占用的字节数")
print(f"每个元素占用字节数: {test_arr.itemsize}")

# =====================================================
# 3. 数组形状操作
# =====================================================
print_section("3. 数组形状操作")

print_example("reshape函数", "改变数组形状（不改变元素个数）")
arr = np.arange(12)
print(f"原数组: {arr}")
reshaped = arr.reshape(3, 4)
print(f"改为3x4:\n{reshaped}")

print_example("flatten函数", "将多维数组展平为1维")
print(f"2维数组:\n{test_arr}")
flattened = test_arr.flatten()
print(f"展平后: {flattened}")

print_example("transpose函数", "数组转置（行列互换）")
arr_t = np.array([[1, 2], [3, 4], [5, 6]])
print(f"原数组(3x2):\n{arr_t}")
print(f"转置后(2x3):\n{arr_t.T}")

# =====================================================
# 4. 索引和切片
# =====================================================
print_section("4. 索引和切片")

arr = np.arange(10)
print_example("基本索引", "获取单个元素")
print(f"数组: {arr}")
print(f"第3个元素(arr[2]): {arr[2]}")
print(f"最后一个元素(arr[-1]): {arr[-1]}")

print_example("切片操作", "获取元素范围")
print(f"前5个元素(arr[:5]): {arr[:5]}")
print(f"从第2个开始到第5个(arr[2:5]): {arr[2:5]}")
print(f"每隔1个取一个(arr[::2]): {arr[::2]}")
print(f"每隔2个取一个(arr[::3]): {arr[::3]}")
print(f"每隔0个取一个(arr[::]): {arr[::]}")
print_example("二维数组索引", "获取矩阵元素和行列")
mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"矩阵:\n{mat}")
print(f"第0行(mat[0]): {mat[0]}")
print(f"第1列(mat[:, 1]): {mat[:, 1]}")
print(f"第1行第2列的元素(mat[1, 2]): {mat[1, 2]}")

print_example("布尔索引", "根据条件选择元素")
arr = np.array([1, 2, 3, 4, 5, 6])
print(f"原数组: {arr}")
mask = arr > 3
print(f"条件(arr > 3): {mask}")
print(f"条件(类型): {mask.dtype}")
print(f"选中的元素: {arr[mask]}")

# =====================================================
# 5. 基本运算
# =====================================================
print_section("5. 基本运算")

a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

print_example("元素级运算", "对应位置进行运算")
print(f"a: {a}")
print(f"b: {b}")
print(f"加法(a + b): {a + b}")
print(f"减法(a - b): {a - b}")
print(f"乘法(a * b): {a * b}")
print(f"除法(b / a): {b / a}")
print(f"乘方(a ** 2): {a ** 2}")

print_example("与标量运算", "数组与单个数值的运算")
print(f"数组: {a}")
print(f"数组 * 3: {a * 3}")
print(f"数组 + 10: {a + 10}")

# =====================================================
# 6. 统计函数
# =====================================================
print_section("6. 统计函数")

arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

print_example("sum函数", "计算数组元素的和")
print(f"数组: {arr}")
print(f"总和: {np.sum(arr)}")

print_example("mean函数", "计算平均值")
print(f"平均值: {np.mean(arr)}")

print_example("std函数", "计算标准差")
print(f"标准差: {np.std(arr)}")

print_example("max和min函数", "找最大值和最小值")
print(f"最大值: {np.max(arr)}")
print(f"最小值: {np.min(arr)}")

print_example("argmax和argmin函数", "找最大值和最小值的索引")
print(f"最大值索引: {np.argmax(arr)}")
print(f"最小值索引: {np.argmin(arr)}")

print_example("sort函数", "排序数组")
print(f"原数组: {arr}")
print(f"排序后: {np.sort(arr)}")

# =====================================================
# 7. 矩阵运算
# =====================================================
print_section("7. 矩阵运算")

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print_example("点积(矩阵乘法)", "使用dot函数进行矩阵乘法")
print(f"矩阵A:\n{A}")
print(f"矩阵B:\n{B}")
print(f"A·B的结果:\n{np.dot(A, B)}")

print_example("元素级乘法", "对应位置相乘（不是矩阵乘法）")
print(f"A * B:\n{A * B}")

print_example("矩阵转置", "使用T属性转置矩阵")
print(f"原矩阵:\n{A}")
print(f"转置:\n{A.T}")

# =====================================================
# 8. 广播(Broadcasting)
# =====================================================
print_section("8. 广播(Broadcasting)")

print_example("广播操作", "不同形状的数组自动扩展进行运算")
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([10, 20, 30])
print(f"矩阵a(2x3):\n{a}")
print(f"向量b(1x3): {b}")
print(f"a + b(自动扩展b):\n{a + b}")

# =====================================================
# 9. 数组连接和分割
# =====================================================
print_section("9. 数组连接和分割")

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print_example("concatenate函数", "连接两个数组")
print(f"数组a: {a}")
print(f"数组b: {b}")
print(f"连接结果: {np.concatenate([a, b])}")

print_example("stack函数", "堆叠数组成新维度")
print(f"堆叠后(2x3):\n{np.stack([a, b])}")

print_example("split函数", "分割数组")
arr = np.arange(20)
print(f"原数组: {arr}")
splits = np.split(arr, 4)
print(f"分成4个部分:")
for i, part in enumerate(splits):
    print(f"  部分{i}: {part}")

# =====================================================
# 10. 数学函数
# =====================================================
print_section("10. 数学函数")

arr = np.array([0, np.pi/2, np.pi])

print_example("三角函数", "sin, cos, tan等")
print(f"数组: {arr}")
print(f"sin值: {np.sin(arr)}")
print(f"cos值: {np.cos(arr)}")

print_example("指数和对数", "exp和log函数")
arr2 = np.array([1, 2, 3, 4, 5])
print(f"数组: {arr2}")
print(f"e的幂次(exp): {np.exp(arr2)}")
print(f"自然对数(log): {np.log(arr2)}")
print(f"以10为底的对数(log10): {np.log10(arr2)}")

print_example("平方根和绝对值", "sqrt和abs")
arr3 = np.array([-2, -1, 0, 1, 4, 9])
print(f"数组: {arr3}")
print(f"平方根(sqrt): {np.sqrt(np.abs(arr3))}")
print(f"绝对值(abs): {np.abs(arr3)}")

# =====================================================
# 11. 随机数生成
# =====================================================
print_section("11. 随机数生成")

print_example("rand函数", "生成[0,1)均匀分布的随机数")
print(f"1D随机数组(5个): {np.random.rand(5)}")

print_example("randn函数", "生成标准正态分布的随机数")
print(f"1D随机数组(5个): {np.random.randn(5)}")

print_example("randint函数", "生成指定范围的随机整数")
print(f"0-10之间的5个随机整数: {np.random.randint(0, 10, 5)}")

print_example("choice函数", "从数组中随机选择")
arr = np.array([10, 20, 30, 40, 50])
print(f"从{arr}中随机选择3个: {np.random.choice(arr, 3)}")

# =====================================================
# 12. 总结
# =====================================================
print_section("总结")
print("""
NumPy的主要用途：
  1. 创建和操作多维数组
  2. 进行快速的数值计算（比Python列表快得多）
  3. 执行线性代数运算（矩阵乘法、求逆等）
  4. 生成随机数
  5. 进行统计分析
  6. 与其他科学库(pandas, matplotlib等)无缝集成

关键概念：
  - ndarray: NumPy的核心数据结构
  - 广播(Broadcasting): 自动扩展不同形状的数组
  - 矢量化: 避免使用循环，直接对整个数组操作
  - dtype: 数组元素的数据类型
""")

print("\n" + "="*60)
print("  学习完毕！")
print("="*60 + "\n")
