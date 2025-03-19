import torch
import numpy as np
from torch.nn.functional import conv2d

def compute_gradient(image):
    # print("image: ", image.shape)
    """ Compute the gradient magnitude of an image. """
    if len(image.shape) == 3:
        image = image.unsqueeze(0)  # Add batch dimension
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    # grad_x = conv2d(image, sobel_x, padding=1)
    # grad_y = conv2d(image, sobel_y, padding=1)
    
    grad_x = []
    grad_y = []
    for c in range(image.shape[1]):  # Loop over channels
        grad_x.append(conv2d(image[:, c:c+1, :, :], sobel_x, padding=1))  # Select single channel
        grad_y.append(conv2d(image[:, c:c+1, :, :], sobel_y, padding=1))
    
    # Concatenate results
    grad_x = torch.cat(grad_x, dim=1)
    grad_y = torch.cat(grad_y, dim=1)
   
    gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    return gradient.squeeze(0)  # Remove batch dimension

def gradient_based_sampling(img, sample_q):
    """ Sample points based on gradient magnitude. """
    # Compute gradient
    gradient = compute_gradient(img)
    gradient = gradient.mean(dim=0)  # Average over channels
    
    # Normalize gradient to probabilities
    # prob = gradient / gradient.sum()
    prob = (gradient + 1e-8) / (gradient.sum() + 1e-8)
    prob = prob.view(-1).numpy()
    
    if np.isnan(prob).any():
        print("Warning: Probabilities contain NaN values. Replacing NaN with uniform probabilities.")
        prob = np.nan_to_num(prob, nan=1.0 / len(prob))
    
    prob = prob / prob.sum()
    # Sample points
    total_pixels = len(prob)
    
    # Count the number of non-zero probabilities
    non_zero_count = np.count_nonzero(prob)
    
    # If sample_q is greater than the number of non-zero probabilities, prioritize high-gradient points
    if sample_q > non_zero_count:
        print(f"Warning: sample_q ({sample_q}) is greater than the number of non-zero gradient points ({non_zero_count}). "
              f"Sampling all non-zero gradient points and filling the rest randomly.")
        
        # Get indices of non-zero probabilities
        non_zero_indices = np.where(prob > 0)[0]
        
        # Sample all non-zero gradient points
        sample_lst = non_zero_indices[:min(sample_q, non_zero_count)]
        
        # If we still need more samples, randomly sample the remaining points (with replacement)
        if len(sample_lst) < sample_q:
            remaining_samples = sample_q - len(sample_lst)
            random_samples = np.random.choice(total_pixels, remaining_samples, replace=False)
            sample_lst = np.concatenate([sample_lst, random_samples])
    else:
        # Sample points based on gradient magnitude
        sample_lst = np.random.choice(total_pixels, sample_q, replace=False, p=prob)
    
    return sample_lst

def scale_gradient_based_sampling(img, sample_q):
    # print("why scale")
    """ Sample points based on gradient magnitude and random sampling, with equal weight (50%) for both. """
    # Compute gradient
    gradient = compute_gradient(img)
    gradient = gradient.mean(dim=0)  # Average over channels
    
    # Normalize gradient to probabilities
    prob = (gradient + 1e-8) / (gradient.sum() + 1e-8)
    prob = prob.view(-1).numpy()
    
    if np.isnan(prob).any():
        print("Warning: Probabilities contain NaN values. Replacing NaN with uniform probabilities.")
        prob = np.nan_to_num(prob, nan=1.0 / len(prob))
    
    prob = prob / prob.sum()

    total_pixels = len(prob)
    
    # Split the number of samples equally between gradient and random sampling
    # sample_q_gradient = sample_q // 2
    sample_q_gradient = int(sample_q * 0.6)  # 60% of sample_q for gradient-based sampling
    
    sample_q_random = sample_q - sample_q_gradient  # The remaining will be from random

    # Sample points based on gradient probabilities
    gradient_sample_lst = np.random.choice(total_pixels, sample_q_gradient, replace=False, p=prob)
    
    # Sample points randomly (uniform distribution)
    random_prob = np.ones(total_pixels) / total_pixels  # Uniform random probabilities
    random_sample_lst = np.random.choice(total_pixels, sample_q_random, replace=False, p=random_prob)

    # Combine the two lists of sampled points
    sample_lst = np.concatenate([gradient_sample_lst, random_sample_lst])

    return sample_lst


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

def sample_coordinates(img, sample_q, coord, rgb, grad_based=False, roi_mask=None):
    """ 根据采样策略进行坐标选择。"""

    if grad_based:
        # print("why gradient")
        # sample_lst = gradient_based_sampling(img, sample_q)
        sample_lst = scale_gradient_based_sampling(img, sample_q)
        return sample_lst
    elif roi_mask is not None:
        # 基于ROI区域或随机选择
        return sample_roi_or_random(coord, rgb, roi_mask, sample_q)
    else:
        # 如果没有给出ROI，也没有选择基于梯度的采样，默认进行随机采样
        sample_lst = np.random.choice(len(coord), sample_q, replace=False)
        # return coord[sample_lst], rgb[sample_lst]
        return sample_lst
