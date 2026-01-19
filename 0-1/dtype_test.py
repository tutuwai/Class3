import numpy as np

print("="*60)
print("  NumPy 数据类型转换演示")
print("="*60)

# 测试1: 直接赋值浮点数到整型数组
print("\n【测试1】直接赋值浮点数到整型数组的某个位置")
print("-" * 50)
arr = np.array([1, 2, 3, 4, 5])
print(f"原数组: {arr}")
print(f"原数据类型: {arr.dtype}")

arr[0] = 6.4
print(f"\n执行: arr[0] = 6.4")
print(f"修改后数组: {arr}")
print(f"数据类型: {arr.dtype} （未改变！）")
print(f"结果: 6.4 被截断为 6")

# 测试2: 使用append添加浮点数
print("\n\n【测试2】使用append添加浮点数到整型数组")
print("-" * 50)
arr = np.array([1, 2, 3, 4, 5])
print(f"原数组: {arr}, dtype: {arr.dtype}")

result = np.append(arr, 6.4)
print(f"\n执行: np.append(arr, 6.4)")
print(f"结果数组: {result}")
print(f"结果数据类型: {result.dtype}")
print(f"说明: append 返回新数组，类型自动升级为 float64")

# 测试3: 算术运算
print("\n\n【测试3】整型数组与浮点数进行算术运算")
print("-" * 50)
arr = np.array([1, 2, 3, 4, 5])
print(f"原数组: {arr}, dtype: {arr.dtype}")

result = arr + 6.4
print(f"\n执行: arr + 6.4")
print(f"结果: {result}")
print(f"结果数据类型: {result.dtype}")
print(f"说明: 运算结果自动升级为 float64，原数组不变")

# 测试4: 创建混合类型数组
print("\n\n【测试4】创建包含整数和浮点数的数组")
print("-" * 50)
arr = np.array([1, 2, 3, 4.5])
print(f"执行: np.array([1, 2, 3, 4.5])")
print(f"结果数组: {arr}")
print(f"数据类型: {arr.dtype}")
print(f"说明: 整个数组自动升级为 float64")

# 测试5: 显式转换
print("\n\n【测试5】显式转换数据类型")
print("-" * 50)
arr = np.array([1, 2, 3, 4, 5])
print(f"原数组: {arr}, dtype: {arr.dtype}")

arr_float = arr.astype(float)
print(f"\n执行: arr.astype(float)")
print(f"转换后数组: {arr_float}")
print(f"数据类型: {arr_float.dtype}")
print(f"原数组: {arr}, dtype: {arr.dtype}")
print(f"说明: astype() 返回新数组，原数组不变")

# 总结
print("\n\n" + "="*60)
print("  总结")
print("="*60)
print("""
1. 赋值给特定位置：
   - int数组[index] = 6.4 → 6.4被截断为6，数组类型不变

2. append/连接操作：
   - 返回新数组，类型自动升级为float64

3. 算术运算：
   - 返回新数组，类型自动升级为float64，原数组不变

4. 创建数组时混合类型：
   - 整个数组自动升级为浮点类型

5. 关键原则：
   ✓ NumPy 优先保持原数组的类型和内容不变
   ✓ 生成新数组时，类型会自动升级（int → float）
   ✓ 赋值给现有元素时，会进行类型转换（截断或舍入）
""")
