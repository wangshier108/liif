import torch.nn as nn
import copy
from models import register


@register('refinedmlp')
class REFINEDMLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list, threshold=0.5):
        super().__init__()
        layers = []
        refine_layers = []
        lastv = in_dim
        self.threshold = threshold
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        refine_layers = copy.deepcopy(layers)
        layers.append(nn.Linear(lastv, out_dim))
        refine_layers.append(nn.Linear(lastv, 64))
        refine_layers.append(nn.Linear(64, out_dim))
        
        self.base_layers = nn.Sequential(*layers)
        self.refine_layers = nn.Sequential(*refine_layers)
        
        self.confidence_mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print("refine the mlp")
        shape = x.shape[:-1]
        x_flat = x.view(-1, x.shape[-1])

        base_pred = self.base_layers(x_flat)  # [N * width * height, out_dim]
        refine_pred = self.refine_layers(x_flat)  # [N * width * height, out_dim]
        confidence = self.confidence_mlp(x_flat).squeeze(-1)  # [N * width * height]
        
        # 当置信度低于阈值时使用细化预测，否则使用基础预测
        mask = (confidence < self.threshold).float().unsqueeze(-1)  # [N * width * height, 1]
        out = mask * refine_pred + (1 - mask) * base_pred  # [N * width * height, out_dim]
        
        return out.view(*shape, -1)
