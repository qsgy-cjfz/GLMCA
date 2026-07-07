import numpy as np
import re

def extract_floats(text):
    """
    从文本中提取所有有效的浮点数
    处理正常数字和拼接数字（如 '0.03136382997035980.0'）
    """
    # 匹配所有浮点数（包括整数）
    pattern = r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+\.\d+|[-+]?\d+'
    matches = re.findall(pattern, text)
    results = []
    for m in matches:
        try:
            results.append(float(m))
        except ValueError:
            continue
    return results

# 读取文件
with open("run/result1/second/gnn_result.txt", 'r') as f:
    content = f.read()

# 方法1：按空白字符（空格、换行、制表符）分割
# 这会处理所有分隔符，并自动处理拼接数字
tokens = content.split()
print(f"原始分割后得到 {len(tokens)} 个token")

# 提取所有数字
all_values = []
for token in tokens:
    # 尝试直接转换
    try:
        val = float(token)
        all_values.append(val)
    except ValueError:
        # 如果转换失败，尝试从token中提取数字（处理拼接情况）
        extracted = extract_floats(token)
        all_values.extend(extracted)

print(f"提取到 {len(all_values)} 个数值")

# 检查数据完整性
if len(all_values) % 12 != 0:
    print(f"警告：数据点数 {len(all_values)} 不是12的倍数！")
    print("将截断到最近的12的倍数...")
    all_values = all_values[: (len(all_values) // 12) * 12]
    print(f"截断后数据点数: {len(all_values)}")

n_users = len(all_values) // 12
print(f"总用户数: {n_users}")
print(f"总数据点数: {len(all_values)}")

# 重塑数据
results = np.array(all_values).reshape(n_users, 12)
labels = ['pre@5','rec@5','ndcg@5','pre@10','rec@10','ndcg@10',
          'pre@15','rec@15','ndcg@15','pre@20','rec@20','ndcg@20']

# 全零用户
all_zero = np.all(results == 0, axis=1).sum()
print(f"\n全零用户数: {all_zero}/{n_users} ({100*all_zero/n_users:.1f}%)")
print(f"有命中用户数: {n_users - all_zero}/{n_users} ({100*(n_users-all_zero)/n_users:.1f}%)")

# 各指标平均
print("\n=== 各指标平均值（所有用户） ===")
means = np.mean(results, axis=0)
for i, label in enumerate(labels):
    print(f"  {label}: {means[i]:.6f}")

# 仅非零用户的平均
non_zero_mask = ~np.all(results == 0, axis=1)
if non_zero_mask.any():
    print(f"\n=== 仅非零用户的平均指标（{non_zero_mask.sum()}个用户） ===")
    nz_means = np.mean(results[non_zero_mask], axis=0)
    for i, label in enumerate(labels):
        print(f"  {label}: {nz_means[i]:.6f}")

# 分K值汇总
print("\n=== 按K值汇总 ===")
for k_idx, k in enumerate([5,10,15,20]):
    col_base = k_idx * 3
    pre = means[col_base]
    rec = means[col_base+1]
    ndcg = means[col_base+2]
    print(f"  K={k:2d}: pre={pre:.6f}, rec={rec:.6f}, ndcg={ndcg:.6f}")

# 额外统计：各指标的最大值、最小值、标准差
print("\n=== 各指标统计（所有用户） ===")
for i, label in enumerate(labels):
    col_data = results[:, i]
    print(f"  {label}: mean={np.mean(col_data):.6f}, std={np.std(col_data):.6f}, "
          f"min={np.min(col_data):.6f}, max={np.max(col_data):.6f}")