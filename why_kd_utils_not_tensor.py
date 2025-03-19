import numpy as np
import torch
import torch.nn as nn

# 定义每个叶子节点的最大点数
max_points_per_leaf = 1000

mlp_num = 4

is_train = True

def reorder_hr_coords(hr_coords, point_indices):
    # 验证长度一致性
    total_indices = sum(len(idx) for idx in point_indices)
    assert total_indices == hr_coords.shape[0], "Total indices length doesn't match hr_coords"
    
    # 拼接所有索引
    combined_indices = torch.cat(point_indices, dim=0)
    
    # 验证索引唯一性（根据需求可选）
    unique_indices = torch.unique(combined_indices)
    assert len(unique_indices) == hr_coords.shape[0], "Indices are not unique"
    
    # 重新排序
    sorted_hr_coords = hr_coords[combined_indices.long()]
    
    return sorted_hr_coords

def calculate_variance(coords, axis):
    """计算指定轴上的方差"""
    return np.var(coords[:, axis])

def recursive_split_bak(image_coords, points_indices, max_points_per_leaf, depth=0):
    # print("why max_points_per_leaf: ", max_points_per_leaf)
    if len(points_indices) <= max_points_per_leaf:
        # 记录叶子节点的索引和对应的点
        return [(points_indices, image_coords[0, points_indices])]
    
    # 计算每个轴的方差
    variances = [calculate_variance(image_coords[0, points_indices], axis) for axis in range(2)]
    axis = np.argmax(variances)  # 选择方差最大的轴
    
    # 按选定的轴排序并找到中位数
    sorted_indices = points_indices[np.argsort(image_coords[0, points_indices, axis])]
    median = len(sorted_indices) // 2
    
    # 递归划分
    left_indices = sorted_indices[:median]
    right_indices = sorted_indices[median:]
    
    # 递归调用
    left_leaf_nodes = recursive_split(image_coords, left_indices, max_points_per_leaf, depth + 1)
    right_leaf_nodes = recursive_split(image_coords, right_indices, max_points_per_leaf, depth + 1)
    
    return left_leaf_nodes + right_leaf_nodes


def get_leaf_nodes_bak(coords): 
    max_points_per_leaf = 0
    # 初始调用：对每个批次的图像坐标进行划分
    print("why coords: ", coords.shape)
    all_leaf_nodes = []
    if(len(coords.shape) > 2):
        batch_size = coords.shape[0]
        # print("why coords.shape[1]", coords.shape[1])
        max_points_per_leaf = coords.shape[1] // 4
    else:
        batch_size = 1
        coords = np.expand_dims(coords, axis=0) 
        # print("why coords.shape[1]", coords.shape[1])
        max_points_per_leaf = coords.shape[1] // 4
    print("why max_points_per_leaf: ", max_points_per_leaf)
    for i in range(batch_size):
        batch_points_indices = np.arange(len(coords[0]))
        leaf_nodes = recursive_split(coords, batch_points_indices, max_points_per_leaf)
        all_leaf_nodes.append(leaf_nodes)
    
    # for batch_idx, batch_leaf_nodes in enumerate(all_leaf_nodes):
    #     # print(f"Batch {batch_idx}: {len(batch_leaf_nodes)} leaf nodes")
    #     for i, points in enumerate(batch_leaf_nodes):
    #         sub_point_indices, sub_points = points
    #         print(f"Leaf {i}: {len(sub_point_indices)}:{len(sub_points)} points")
    #         print(f"indices: {sub_point_indices}")
    #         print(f"Points: {sub_points[:10]}")  # 打印前10个点
    #         # point_indices.append(sub_point_indices)
    #         # point_coords.append(sub_points)
    print("why leaf_nodes: ", all_leaf_nodes)
    
    point_indices = []
    point_coords = []
    for each_batch_node in all_leaf_nodes:
        for points in each_batch_node:
            indices, coords = points
            point_indices.append(indices)
            point_coords.append(coords)
    print("why lenpoint_indices: ", len(point_indices))
    print("why lenpoint_coords: ", len(point_coords))
    print("why point_indices: ", point_indices)
    print("why point_coords: ", point_coords)
    return point_indices, point_coords
    # 将 all_leaf_nodes 转换为形状为 (batch_size, num_leaf_nodes, num_points, 2) 的列表
    # leaf_nodes = [np.array([np.array(points) for points in leaf_nodes]) for leaf_nodes in all_leaf_nodes]
    # print("why leaf_nodes: ", leaf_nodes)
    
    # return all_leaf_nodes


def recursive_split(image_coords, points_indices, depth=0):
    # print("why max_points_per_leaf: ", max_points_per_leaf)
    if len(points_indices) <= max_points_per_leaf:
        # 记录叶子节点的索引和对应的点
        return [(points_indices, image_coords[0, points_indices])]
    
    # 计算每个轴的方差
    variances = [calculate_variance(image_coords[0, points_indices], axis) for axis in range(2)]
    axis = np.argmax(variances)  # 选择方差最大的轴
    
    # 按选定的轴排序并找到中位数
    sorted_indices = points_indices[np.argsort(image_coords[0, points_indices, axis])]
    median = len(sorted_indices) // 2
    
    # 递归划分
    left_indices = sorted_indices[:median]
    right_indices = sorted_indices[median:]
    
    # 递归调用
    left_leaf_nodes = recursive_split(image_coords, left_indices, depth + 1)
    right_leaf_nodes = recursive_split(image_coords, right_indices, depth + 1)
    
    return left_leaf_nodes + right_leaf_nodes


def get_leaf_nodes(coords): 
    # max_points_per_leaf = 0
    # 初始调用：对每个批次的图像坐标进行划分
    # print("why coords: ", coords.shape)
    all_leaf_nodes = []
    if(len(coords.shape) > 2):
        batch_size = coords.shape[0]
    else:
        batch_size = 1
        coords = np.expand_dims(coords, axis=0) 
    for i in range(batch_size):
        batch_points_indices = np.arange(len(coords[0]))
        # print("why point_indices: ", batch_points_indices)
        leaf_nodes = recursive_split(coords, batch_points_indices)
        all_leaf_nodes.append(leaf_nodes)
    
    # for batch_idx, batch_leaf_nodes in enumerate(all_leaf_nodes):
    #     # print(f"Batch {batch_idx}: {len(batch_leaf_nodes)} leaf nodes")
    #     for i, points in enumerate(batch_leaf_nodes):
    #         sub_point_indices, sub_points = points
            # print(f"Leaf {i}: {len(sub_point_indices)}:{len(sub_points)} points")
            # print(f"indices: {sub_point_indices}")
            # print(f"Points: {sub_points[:10]}")  # 打印前10个点
            # point_indices.append(sub_point_indices)
            # point_coords.append(sub_points)
    
    # 将 all_leaf_nodes 转换为形状为 (batch_size, num_leaf_nodes, num_points, 2) 的列表
    leaf_nodes = [np.array([np.array(points) for _, points in leaf_nodes]) for leaf_nodes in all_leaf_nodes]
    # print("why leaf_nodes: ", leaf_nodes)
    
    return all_leaf_nodes


## 测试

# # 假设图像大小为 (H, W)
# H, W = 48, 48
# # 生成图像的像素坐标 (x, y)
# image_coords = np.array([[x, y] for x in range(H) for y in range(W)])
# image_coords = np.expand_dims(image_coords, axis=0)  # 增加批量维度
# batch_size = 3  # 假设批量大小为3
# image_coords = np.repeat(image_coords, batch_size, axis=0)  # 重复批量大小次

# all_leaf_nodes = get_leaf_nodes(image_coords)
# # 打印每个批次的叶子节点的点数和对应的点
# for batch_idx, batch_leaf_nodes in enumerate(all_leaf_nodes):
#     print(f"Batch {batch_idx}: {len(batch_leaf_nodes)} leaf nodes")
#     for i, points in enumerate(batch_leaf_nodes):
#         print(f"Leaf {i}: {len(points)} points")
#         print(f"Points: {points[:10]}")  # 打印前10个点