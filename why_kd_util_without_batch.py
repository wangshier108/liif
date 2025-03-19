
import numpy as np
import torch
import torch.nn as nn
# 假设图像大小为 (H, W)
H, W = 48, 48
# 生成图像的像素坐标 (x, y)
image_coords = np.array([[x, y] for x in range(H) for y in range(W)])
# 定义每个叶子节点的最大点数
max_points_per_leaf = 500
# 递归划分函数
leaf_nodes = []
def calculate_variance(coords, axis):
    """计算指定轴上的方差"""
    return np.var(coords[:, axis])
def recursive_split(points_indices, depth=0):
    if len(points_indices) <= max_points_per_leaf:
        # 记录叶子节点的索引和对应的点
        leaf_nodes.append((points_indices, image_coords[points_indices]))
        return
    
    # 计算每个轴的方差
    variances = [calculate_variance(image_coords[points_indices], axis) for axis in range(2)]
    axis = np.argmax(variances)  # 选择方差最大的轴
    
    # 按选定的轴排序并找到中位数
    sorted_indices = points_indices[np.argsort(image_coords[points_indices, axis])]
    median = len(sorted_indices) // 2
    
    # 递归划分
    recursive_split(sorted_indices[:median], depth + 1)
    recursive_split(sorted_indices[median:], depth + 1)
# 初始调用：对整个图像坐标进行划分
recursive_split(np.arange(len(image_coords)))
print(f"Number of leaf nodes: {len(leaf_nodes)}")
# 打印每个叶子节点的点数和对应的点
for i, (indices, points) in enumerate(leaf_nodes):
    print(f"Leaf {i}: {len(points)} points")
    print(f"Points: {points[:10]}")  # 打印前10个点
