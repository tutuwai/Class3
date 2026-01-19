import numpy as np

print("="*60)
print("  NumPy astype() 是否深层拷贝演示")
print("="*60)

# 测试1: 验证astype()创建新数组
print("\n【测试1】astype() 创建新数组的验证")
print("-" * 50)
arr = np.array([1, 2, 3, 4, 5])
arr_float = arr.astype(float)

print(f"原数组: {arr}, id: {id(arr)}, dtype: {arr.dtype}")
print(f"转换后数组: {arr_float}, id: {id(arr_float)}, dtype: {arr_float.dtype}")
print(f"是否是同一个对象？ {arr is arr_float}")
print(f"内存地址是否相同？ {np.shares_memory(arr, arr_float)}")
print("结论: 创建了新的独立数组对象")

# 测试2: 修改新数组，原数组是否改变
print("\n\n【测试2】修改新数组，原数组是否改变")
print("-" * 50)
arr = np.array([1, 2, 3, 4, 5])
arr_float = arr.astype(float)

print(f"原数组修改前: {arr}")
print(f"新数组修改前: {arr_float}")

arr_float[0] = 99.9
print(f"\n执行: arr_float[0] = 99.9")
print(f"原数组: {arr}")
print(f"新数组: {arr_float}")
print(f"\n结论: 修改新数组，原数组不受影响 ✓")

# 测试3: 修改原数组，新数组是否改变
print("\n\n【测试3】修改原数组，新数组是否改变")
print("-" * 50)
arr = np.array([1, 2, 3, 4, 5])
arr_float = arr.astype(float)

print(f"原数组修改前: {arr}")
print(f"新数组修改前: {arr_float}")

arr[0] = 100
print(f"\n执行: arr[0] = 100")
print(f"原数组: {arr}")
print(f"新数组: {arr_float}")
print(f"\n结论: 修改原数组，新数组不受影响 ✓")

# 测试4: 与view()进行对比（不同数据类型）
print("\n\n【测试4】astype() 与 view() 的对比（不同数据类型）")
print("-" * 50)
arr = np.array([1, 2, 3, 4, 5], dtype=int)

# astype() - 深层拷贝
print("astype() 的行为（创建深层拷贝）:")
arr_astype = arr.astype(float)
print(f"原数组: {arr}, dtype: {arr.dtype}")
print(f"astype数组: {arr_astype}, dtype: {arr_astype.dtype}")
arr_astype[0] = 99.9
print(f"\n修改 arr_astype[0] = 99.9 后:")
print(f"原数组: {arr}")
print(f"astype数组: {arr_astype}")
print(f"共享内存？ {np.shares_memory(arr, arr_astype)} (NO = 深层拷贝)")

# view() - 浅层拷贝（共享数据，但改变解释方式）
print("\n\nview() 的行为（创建视图，同类型）:")
arr = np.array([1, 2, 3, 4, 5], dtype=int)
arr_view = arr.view()
print(f"原数组: {arr}, dtype: {arr.dtype}")
print(f"view数组: {arr_view}, dtype: {arr_view.dtype}")
arr_view[0] = 99
print(f"\n修改 arr_view[0] = 99 后:")
print(f"原数组: {arr}")
print(f"view数组: {arr_view}")
print(f"共享内存？ {np.shares_memory(arr, arr_view)} (YES = 浅层拷贝)")
print(f"说明: 修改视图会影响原数组！")

# 测试5: copy()方法的对比
print("\n\n【测试5】astype() 与 copy() 的对比")
print("-" * 50)
arr = np.array([1, 2, 3, 4, 5], dtype=int)

print("astype()的行为（隐式复制）:")
arr_astype = arr.astype(int)
arr_astype[0] = 99
print(f"原数组: {arr}")
print(f"astype数组: {arr_astype}")
print(f"共享内存？ {np.shares_memory(arr, arr_astype)}")

print("\ncopy()的行为（显式复制）:")
arr = np.array([1, 2, 3, 4, 5], dtype=int)
arr_copy = arr.copy()
arr_copy[0] = 99
print(f"原数组: {arr}")
print(f"copy数组: {arr_copy}")
print(f"共享内存？ {np.shares_memory(arr, arr_copy)}")

# 测试6: 切片操作（创建视图，共享内存）
print("\n\n【测试6】切片操作（创建视图）")
print("-" * 50)
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"原数组: {arr}")

# 切片
slice_arr = arr[2:5]
print(f"\n执行: slice_arr = arr[2:5]")
print(f"切片数组: {slice_arr}")
print(f"共享内存？ {np.shares_memory(arr, slice_arr)} (YES = 视图)")

# 修改切片，原数组会改变
print(f"\n执行: slice_arr[0] = 99")
slice_arr[0] = 99
print(f"原数组: {arr}")
print(f"切片数组: {slice_arr}")
print(f"说明: 修改切片会影响原数组！")

# 测试7: 多维数组切片
# 这是一个比较有趣的特性，多维数组有这样，但是一维数组没有这样的，只是获得一个标量值
print("\n\n【测试7】多维数组的切片和视图")
print("-" * 50)
mat = np.arange(20).reshape(4, 5)
print(f"原矩阵(4x5):\n{mat}")

# 获取一行
row = mat[1]
print(f"\n执行: row = mat[1]（获取第1行）")
print(f"行向量: {row}")
print(f"共享内存？ {np.shares_memory(mat, row)}")

row[0] = 999
print(f"\n修改 row[0] = 999 后:")
print(f"原矩阵:\n{mat}")
print(f"说明: 修改行向量会影响原矩阵")

# 测试8: 显式复制与隐式视图的对比
print("\n\n【测试8】显式复制切片 vs 隐式视图切片")
print("-" * 50)
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 隐式视图（默认切片）
slice_view = arr[2:5]
print(f"隐式视图: slice_view = arr[2:5]")
print(f"共享内存？ {np.shares_memory(arr, slice_view)}")

# 显式复制（使用copy()）
slice_copy = arr[2:5].copy()
print(f"\n显式复制: slice_copy = arr[2:5].copy()")
print(f"共享内存？ {np.shares_memory(arr, slice_copy)}")

# 修改两者，对比差异
print(f"\n修改两个切片:")
slice_view[0] = 99
slice_copy[0] = 99
print(f"原数组: {arr}")
print(f"视图切片: {slice_view}")
print(f"复制切片: {slice_copy}")
print(f"说明: 只有视图修改影响原数组")

# 总结
print("\n\n" + "="*60)
print("  总结")
print("="*60)
print("""
【三种操作对比】

1. astype() - 数据类型转换（深层拷贝）
   ❌ 不共享内存
   ✓ 创建完全独立的新数组
   用途: 改变数据类型时使用
   
2. view() - 创建视图（浅层拷贝）
   ✓ 共享内存
   ⚠️  修改视图会影响原数组
   用途: 改变数据解释方式，节省内存
   
3. copy() - 显式复制（深层拷贝）
   ❌ 不共享内存
   ✓ 创建完全独立的新数组
   用途: 需要独立副本时使用

【切片操作】

1. 默认切片 - 创建视图
   arr[2:5] → 视图（共享内存）
   修改切片会影响原数组
   
2. copy切片 - 创建副本
   arr[2:5].copy() → 新数组（不共享内存）
   修改不影响原数组
   
3. 多维切片
   mat[1] → 获取一行（视图）
   mat[1, :] → 获取一行（视图）
   mat[1].copy() → 复制一行

【内存共享情况速查表】
┌─────────────────────┬──────────────┬────────┐
│ 操作                │ 共享内存     │ 类型   │
├─────────────────────┼──────────────┼────────┤
│ arr.view()          │ YES (✓)     │ 视图   │
│ arr[2:5]            │ YES (✓)     │ 视图   │
│ arr.copy()          │ NO (❌)     │ 副本   │
│ arr[2:5].copy()     │ NO (❌)     │ 副本   │
│ arr.astype(float)   │ NO (❌)     │ 副本   │
│ np.array(arr)       │ NO (❌)     │ 副本   │
└─────────────────────┴──────────────┴────────┘

【最佳实践】
- 需要独立数据 → 用 copy()
- 只是查看 → 用 view() 节省内存
- 改变类型 → 用 astype()
- 做切片且要改动 → 用 slice.copy()
""")
