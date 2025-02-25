import numpy as np
import torch

# 定义每个叶子节点的最大点数
# 定义每个叶子节点的最大点数
max_points_per_leaf = 1000

mlp_num = 4

is_train = True

def calculate_variance(coords, axis):
    """计算指定轴上的方差"""
    return torch.var(coords[:, axis])

def recursive_split(image_coords, points_indices, depth=0):
    if len(points_indices) <= max_points_per_leaf:
        # 记录叶子节点的索引和对应的点
        return [(points_indices, image_coords[0, points_indices])]
    
    # 计算每个轴的方差
    variances = [calculate_variance(image_coords[0, points_indices], axis) for axis in range(2)]
    axis = torch.argmax(torch.tensor(variances))  # 选择方差最大的轴
    
    # 按选定的轴排序并找到中位数
    sorted_indices = points_indices[torch.argsort(image_coords[0, points_indices, axis])]
    median = len(sorted_indices) // 2
    
    # 递归划分
    left_indices = sorted_indices[:median]
    right_indices = sorted_indices[median:]
    
    # 递归调用
    left_leaf_nodes = recursive_split(image_coords, left_indices, depth + 1)
    right_leaf_nodes = recursive_split(image_coords, right_indices, depth + 1)
    
    return left_leaf_nodes + right_leaf_nodes

def get_leaf_nodes(coords): 
    # 初始调用：对每个批次的图像坐标进行划分
    all_leaf_nodes = []
    if(len(coords.shape) > 2):
        batch_size = coords.shape[0]
    else:
        batch_size = 1
        coords = coords.unsqueeze(0)
    for i in range(batch_size):
        batch_points_indices = torch.arange(len(coords[0])).cuda()
        # print("why batch_indices: ", batch_points_indices)
        leaf_nodes = recursive_split(coords, batch_points_indices)
        all_leaf_nodes.append(leaf_nodes)
    # batch_size = coords.shape[0]
    # for batch_idx, batch_leaf_nodes in enumerate(all_leaf_nodes):
    #     print(f"Batch {batch_idx}: {len(batch_leaf_nodes)} leaf nodes")
    #     for i, points in enumerate(batch_leaf_nodes):
    #         print(f"Leaf {i}: {len(points[0])} points")
    #         print(f"Points: {points[:2]}")  # 打印前10个点
        # print("why leaf_nodes: ", all_leaf_nodes)
    # 将 all_leaf_nodes 转换为形状为 (batch_size, num_leaf_nodes, num_points, 2) 的列表
    # all_leaf_nodes = [torch.stack([points for _, points in leaf_nodes]) for leaf_nodes in all_leaf_nodes]
    # print("why after all_leaf_nodes: ", all_leaf_nodes[0].shape)
    # print("why after all_leaf_nodes: ", all_leaf_nodes[0].shape)
    return all_leaf_nodes

def reorder(hr_coord, cell, hr_rgb):
    leaf_nodes = get_leaf_nodes(hr_coord)
            
    point_indices = []
    point_coords = []
    # print("why hr_coord.shape: ", hr_coord.shape)
    for batch_idx, batch_leaf_nodes in enumerate(leaf_nodes):
        # print(f"Batch {batch_idx}: {len(batch_leaf_nodes)} leaf nodes")
        sub_indices = []
        sub_coords = []
        for i, points in enumerate(batch_leaf_nodes):
            # print("why points : ", points)
            sub_point_indices, sub_points = points
            sub_indices.append(sub_point_indices)
            sub_coords.append(sub_points)
        
        sub_indices = torch.stack(sub_indices)
        sub_coords = torch.stack(sub_coords)
        
        point_indices.append(sub_indices)
        point_coords.append(sub_coords)
     
    point_indices = torch.stack(point_indices)
    point_coords = torch.stack(point_coords)
    # print("why point_coords: ", point_coords.shape)
    # print("why point_indices: ", point_indices.shape)
    if(len(cell.shape) > 2):
        cell = cell.view(cell.shape[0], mlp_num, -1, 2) 
        # print("why cell: ", cell.shape)
    else:
        cell = cell.view(mlp_num, -1, 2) 
    
    if(len(point_indices.shape) > 2):
        indices_flat = point_indices.view(point_indices.shape[0],-1)  #(batch, 2304)
        # print("why indices_flat: ", indices_flat.shape)
    else:
        indices_flat = point_indices.view(-1)
    # 使用索引重排hr_rgb
    if(len(hr_rgb.shape) > 2):
        for b, batch_hr_rgb in enumerate(hr_rgb):
            batch_hr_rgb = batch_hr_rgb[indices_flat[b], :]
            hr_rgb[b] = batch_hr_rgb
        # print("why hr_rgb: ", hr_rgb.shape)
    else:        
        hr_rgb = hr_rgb[indices_flat, :]
        
    return point_coords, cell, hr_rgb

# ## 测试

# # 假设图像大小为 (H, W)
# H, W = 48, 48
# # 生成图像的像素坐标 (x, y)
# image_coords = torch.tensor([[x, y] for x in range(H) for y in range(W)])
# image_coords = image_coords.unsqueeze(0)  # 增加批量维度
# batch_size = 3  # 假设批量大小为3
# image_coords = image_coords.repeat(batch_size, 1, 1)  # 重复批量大小次

# all_leaf_nodes = get_leaf_nodes(image_coords)
# # 打印每个批次的叶子节点的点数和对应的点
# for batch_idx, batch_leaf_nodes in enumerate(all_leaf_nodes):
#     print(f"Batch {batch_idx}: {len(batch_leaf_nodes)} leaf nodes")
#     for i, points in enumerate(batch_leaf_nodes):
#         print(f"Leaf {i}: {len(points)} points")
#         print(f"Points: {points[:10]}")  # 打印前10个点
