import torch
import numpy as np
from torch.nn.functional import conv2d

def compute_gradient(image):
    print("image: ", image.shape)
    """ Compute the gradient magnitude of an image. """
    if len(image.shape) == 3:
        image = image.unsqueeze(0)  # Add batch dimension
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    # sobel_x = sobel_x.repeat(3, 1, 1, 1)  # Repeat for 3 channels
    # sobel_y = sobel_y.repeat(3, 1, 1, 1)  # Repeat for 3 channels

    grad_x = conv2d(image, sobel_x, padding=1)
    grad_y = conv2d(image, sobel_y, padding=1)
    gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    return gradient.squeeze(0)  # Remove batch dimension

def gradient_based_sampling(img, sample_q):
    """ Sample points based on gradient magnitude. """
    # Compute gradient
    gradient = compute_gradient(img)
    gradient = gradient.mean(dim=0)  # Average over channels

    # Normalize gradient to probabilities
    prob = gradient / gradient.sum()
    prob = prob.view(-1).numpy()

    # Sample points
    sample_lst = np.random.choice(len(prob), sample_q, replace=False, p=prob)
    return sample_lst

def get_gradient_map(img):
    """ 获取图像的梯度图，用于边缘采样。"""
    img = img.unsqueeze(0)  # 添加 batch 维度
    grad_x = F.conv2d(img, torch.tensor([[[[-1, 1]]]]).float(), padding=1)
    grad_y = F.conv2d(img, torch.tensor([[[[-1], [1]]]]).float(), padding=1)
    grad_map = torch.sqrt(grad_x**2 + grad_y**2)
    return grad_map.squeeze(0)  # 移除 batch 维度

def sample_based_on_gradients(hr_coord, hr_rgb, grad_map, sample_q):
    """ 根据梯度图进行采样。"""
    # 将梯度图标准化到 [0, 1] 范围内
    grad_map = grad_map - grad_map.min()
    grad_map = grad_map / grad_map.max()
    
    # 将梯度图平铺成 1D
    grad_map = grad_map.view(-1).cpu().numpy()

    # 计算权重分布
    prob = grad_map / grad_map.sum()

    # 按照梯度图的权重进行采样
    sample_lst = np.random.choice(len(hr_coord), sample_q, replace=False, p=prob)
    return hr_coord[sample_lst], hr_rgb[sample_lst]

def sample_roi_or_random(hr_coord, hr_rgb, roi_mask, sample_q):
    """ 基于ROI区域或者随机选择样本。"""
    roi_indices = np.where(roi_mask.flatten())[0]
    num_roi = len(roi_indices)
    # print(f"num_roi:{num_roi}, roi_mask: {roi_mask.shape}")

    if num_roi >= sample_q:
        # 如果ROI区域足够，直接从ROI区域采样
        sample_lst = np.random.choice(roi_indices, sample_q, replace=False)
    else:
        # # 如果ROI区域不够，先从ROI区域采样，再从全图随机采样
        # remaining_sample_q = sample_q - num_roi
        # sample_lst_roi = roi_indices
        # sample_lst_random = np.random.choice(len(hr_coord), remaining_sample_q, replace=False)

        # 如果 ROI 区域不够，先从 ROI 区域采样，再从全图随机采样
        remaining_sample_q = sample_q - num_roi
        sample_lst_roi = roi_indices

        # 从全图中选择剩余的样本，排除 ROI 区域的索引
        all_indices = np.arange(len(hr_coord))  # 获取所有坐标的索引
        remaining_indices = np.setdiff1d(all_indices, roi_indices)  # 排除 ROI 区域的索引
        sample_lst_random = np.random.choice(remaining_indices, remaining_sample_q, replace=False)

        # 合并结果
        sample_lst = np.concatenate([sample_lst_roi, sample_lst_random])

    # return hr_coord[sample_lst], hr_rgb[sample_lst]
    return sample_lst

def sample_coordinates(img, sample_q, coord, rgb, grad_based=False, roi_based = False, roi_mask=None):
    """ 根据采样策略进行坐标选择。"""

    if grad_based:
        print("why gradient")
        sample_lst = gradient_based_sampling(img, sample_q)
        return sample_lst
    elif (roi_based and roi_mask is not None):
        # 基于ROI区域或随机选择
        # print("why roiu")
        return sample_roi_or_random(coord, rgb, roi_mask, sample_q)
    else:
        # 如果没有给出ROI，也没有选择基于梯度的采样，默认进行随机采样
        sample_lst = np.random.choice(len(coord), sample_q, replace=False)
        # return coord[sample_lst], rgb[sample_lst]
        return sample_lst