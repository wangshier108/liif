import torch.nn as nn
import copy
from models import register
import torch

import torch.nn.functional as F

# @register('refinedmlp')
# class REFINEDMLP(nn.Module):

#     def __init__(self, in_dim, out_dim, hidden_list, threshold=0.5):
#         super().__init__()
#         layers = []
#         refine_layers = []
#         lastv = in_dim
#         self.threshold = threshold
#         for hidden in hidden_list:
#             layers.append(nn.Linear(lastv, hidden))
#             layers.append(nn.ReLU())
#             lastv = hidden
#         refine_layers = copy.deepcopy(layers)
#         layers.append(nn.Linear(lastv, out_dim))
#         refine_layers.append(nn.Linear(lastv, 64))
#         refine_layers.append(nn.ReLU())
#         refine_layers.append(nn.Linear(64, out_dim))

#         self.base_layers = nn.Sequential(*layers)
#         self.refine_layers = nn.Sequential(*refine_layers)

#         self.confidence_mlp = nn.Sequential(
#             nn.Linear(in_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         # print("refine the mlp")
#         shape = x.shape[:-1]
#         x_flat = x.view(-1, x.shape[-1])

#         base_pred = self.base_layers(x_flat)  # [N * width * height, out_dim]
#         refine_pred = self.refine_layers(x_flat)  # [N * width * height, out_dim]
#         confidence = self.confidence_mlp(x_flat).squeeze(-1)  # [N * width * height]

#         # 当置信度低于阈值时使用细化预测，否则使用基础预测
#         mask = (confidence < self.threshold).float().unsqueeze(-1)  # [N * width * height, 1]
#         out = mask * refine_pred + (1 - mask) * base_pred  # [N * width * height, out_dim]

#         return out.view(*shape, -1)

@register('refinedmlp')
class REFINEDMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list, threshold=0.5):
        super().__init__()
        self.threshold = threshold

        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden

        # 共享基础 MLP
        self.base_layers = nn.Sequential(*layers, nn.Linear(lastv, out_dim))

        # 细化 MLP（共享前面的层，只修改最后几层）
        refine_layers = list(layers)  # 共享前面结构
        refine_layers.extend([
            nn.Linear(lastv, 64), nn.ReLU(),
            nn.Linear(64, out_dim)
        ])
        self.refine_layers = nn.Sequential(*refine_layers)

        self.confidence_mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print("why ---")
        shape = x.shape[:-1]
        x_flat = x.view(-1, x.shape[-1])

        base_pred = self.base_layers(x_flat)  # [N * width * height, out_dim]
        refine_pred = self.refine_layers(x_flat)  # [N * width * height, out_dim]
        
        # 置信度计算
        confidence = self.confidence_mlp(x_flat).squeeze(-1)  # [N * width * height]
        # print("why confidence: ", confidence.shape)

        # 置信度 mask ============> original
        # mask = (confidence < self.threshold).float().unsqueeze(-1)
        # out = mask * refine_pred + (1 - mask) * base_pred
        # print("why refine_pred: ", refine_pred.shape)
        
        
        # # 置信度 mask ============> soft
        # confidence = confidence.unsqueeze(-1)
        # out = confidence * refine_pred + (1 - confidence) * base_pred
        
        
        # 置信度 mask ============> residual soft
        confidence = confidence.unsqueeze(-1)
        out = confidence * refine_pred + base_pred
        return out.view(*shape, -1)

@register('refinedmlp2')
class REFINEDMLP2(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list, threshold=0.5):
        super().__init__()
        self.threshold = threshold

        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden

        # 共享基础 MLP
        self.base_layers = nn.Sequential(*layers, nn.Linear(lastv, out_dim))

        # 细化 MLP（共享前面的层，只修改最后几层）
        refine_layers = list(layers)  # 共享前面结构
        refine_layers.extend([
            nn.Linear(lastv, 64), nn.ReLU(),
            nn.Linear(64, out_dim)
        ])
        self.refine_layers = nn.Sequential(*refine_layers)

        self.confidence_mlp = nn.Sequential(
            nn.Linear(out_dim, 1),
            nn.ReLU(),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print("why ---")
        shape = x.shape[:-1]
        x_flat = x.view(-1, x.shape[-1])

        base_pred = self.base_layers(x_flat)  # [N * width * height, out_dim]
        refine_pred = self.refine_layers(x_flat)  # [N * width * height, out_dim]
        
        # 置信度计算
        confidence = self.confidence_mlp(base_pred).squeeze(-1)  # [N * width * height]
        # print("why confidence: ", confidence.shape)

        # 置信度 mask ============> original
        # mask = (confidence < self.threshold).float().unsqueeze(-1)
        # out = mask * refine_pred + (1 - mask) * base_pred
        # print("why refine_pred: ", refine_pred.shape)
        
        
        # # 置信度 mask ============> soft
        # confidence = confidence.unsqueeze(-1)
        # out = confidence * refine_pred + (1 - confidence) * base_pred
        
        
        # 置信度 mask ============> residual soft
        confidence = confidence.unsqueeze(-1)
        out = confidence * refine_pred + base_pred
        return out.view(*shape, -1)

@register('refinedmlp21')
class REFINEDMLP21(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list, threshold=0.5):
        super().__init__()
        self.threshold = threshold

        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden

        # 共享基础 MLP
        # self.base_layers = nn.Sequential(*layers, nn.Linear(lastv, out_dim))
        self.base_layers = nn.Sequential(*layers)
        
        self.coarse_layers = nn.Linear(lastv, out_dim)

        # 细化 MLP（共享前面的层，只修改最后几层）
        # refine_layers = list(layers)  # 共享前面结构
        # refine_layers.extend([
        #     nn.Linear(lastv, 64), nn.ReLU(),
        #     nn.Linear(64, out_dim)
        # ])
        self.refine_layers = nn.Sequential(
            nn.Linear(lastv, 64), nn.ReLU(),
            nn.Linear(64, out_dim)
        )

        self.confidence_mlp = nn.Sequential(
            nn.Linear(out_dim, 1),
            nn.ReLU(),
            nn.Sigmoid()
        )

    def forward(self, x, lr=None):
        # print("why ---")
        shape = x.shape[:-1]
        x_flat = x.view(-1, x.shape[-1])

        base_pred = self.base_layers(x_flat)  
        coarse_pred = self.coarse_layers(base_pred)  # [N * width * height, out_dim]
        refine_pred = self.refine_layers(base_pred)    # [N * width * height, out_dim]
        
        # 置信度计算
        confidence = self.confidence_mlp(coarse_pred).squeeze(-1)  # [N * width * height]
        # print("why confidence: ", confidence.shape)

        # 置信度 mask ============> original
        # mask = (confidence < self.threshold).float().unsqueeze(-1)
        # out = mask * refine_pred + (1 - mask) * base_pred
        # print("why refine_pred: ", refine_pred.shape)
        
        
        # # 置信度 mask ============> soft
        # confidence = confidence.unsqueeze(-1)
        # out = confidence * refine_pred + (1 - confidence) * base_pred
        
        
        # 置信度 mask ============> residual soft
        confidence = confidence.unsqueeze(-1)
        out = confidence * refine_pred + coarse_pred
 
        return out.view(*shape, -1)


@register('refinedmlp3')
class REFINEDMLP3(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list, topk_ratio=0.2):
        super().__init__()
        # 粗粒度路径（2层浅层MLP）
        layers = []
        lastv = in_dim
        for hidden in hidden_list[:2]:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        self.coarse_mlp = nn.Sequential(*layers, nn.Linear(lastv, out_dim))
        
        # 细粒度路径（4层深层MLP）  
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        self.fine_mlp = nn.Sequential(*layers, nn.Linear(lastv, out_dim))
        
        self.topk_ratio = topk_ratio

    def forward(self, x):
        print("why testing")
        # x: [N*H*W, in_dim]
        shape = x.shape[:-1]
        x_flat = x.view(-1, x.shape[-1])
        # print("why x_flat: ", x_flat.shape)     torch.Size([36864, 580])
        # 粗粒度预测
        coarse_out = self.coarse_mlp(x_flat)  # [N*H*W, out_dim]
        # print("why coarse_out: ", coarse_out.shape)  torch.Size([36864, 3])
        # 动态选择高熵区域
        entropy = -torch.sum(coarse_out.softmax(dim=1) * 
                           torch.log(coarse_out.softmax(dim=1)), dim=1)  # [N*H*W]
        # print("why entropy: ", entropy.shape)     torch.Size([36864])
        topk = max(1, int(len(entropy)*self.topk_ratio))
        # print("why topk: ", topk)   7372
        _, indices = torch.topk(entropy, topk)  # 选择最不确定的top-k点
        # print("why indices: ", indices)    
        
        # 细粒度修正
        fine_out = coarse_out.clone()
        fine_out[indices] = self.fine_mlp(x_flat[indices])  # 仅计算关键点
        # print("why fine_out: ", fine_out.shape)
        
        return fine_out.view(*shape, -1)
        
@register('refinedmlp4')
class REFINEDMLP4(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        # 粗粒度路径（2层浅层MLP）
        layers = []
        lastv = in_dim
        for hidden in hidden_list[:2]:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        self.coarse_mlp = nn.Sequential(*layers, nn.Linear(lastv, out_dim))
        
        # 细粒度路径（4层深层MLP）  
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        self.fine_mlp = nn.Sequential(*layers, nn.Linear(lastv, out_dim))
        

    def forward(self, x):
        # x: [N*H*W, in_dim]
        shape = x.shape[:-1]
        x_flat = x.view(-1, x.shape[-1])
        
        # 粗粒度预测
        coarse_out = self.coarse_mlp(x_flat)  # [N*H*W, out_dim]
        
        # 计算熵
        prob = coarse_out.softmax(dim=1)
        entropy = -torch.sum(prob * torch.log(prob + 1e-9), dim=1)  # 避免 log(0)
        
        # 自适应调整 topk_ratio
        entropy_min, entropy_max = entropy.min().item(), entropy.max().item()
        adaptive_topk_ratio = (entropy.mean().item() - entropy_min) / (entropy_max - entropy_min + 1e-9)
        adaptive_topk_ratio = min(1, max(0.05, adaptive_topk_ratio))
        
        # 选择高熵区域
        topk = max(1, int(len(entropy) * adaptive_topk_ratio))
        _, indices = torch.topk(entropy, topk)  # 选择最不确定的top-k点
        
        # 细粒度修正
        fine_out = coarse_out.clone()
        fine_out[indices] = self.fine_mlp(x_flat[indices])  # 仅计算关键点
        
        return fine_out.view(*shape, -1)
