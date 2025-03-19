import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord
import why_kd_utils_not_tensor
# import why_kd_util_without_batch

@register('liif')
class LIIF(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.encoder = models.make(encoder_spec)

        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            if self.feat_unfold:
                imnet_in_dim *= 9
            imnet_in_dim += 2 # attach coord
            if self.cell_decode:
                imnet_in_dim += 2
            # print("why mlp_num: ", why_kd_utils_not_tensor.mlp_num)
            self.imnets = nn.ModuleList(models.make(imnet_spec, args={'in_dim': imnet_in_dim}) for _ in range(why_kd_utils_not_tensor.mlp_num))
            # self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            print("why imnet is none")
            self.imnet = None

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret
    
    def multi_query_rgb(self, coord, indice, cell=None):
        # print("why multi cell: ", cell.shape)
        feat = self.feat

        if self.imnets is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            # (n, c, w, h) -> (n, c*9, block_index1, block_index2)，
            # 其中，第二维是按照先块后通道来排列的。而且block_index1和block_index2的数值对应为w和h。
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
            # print("why feat_unfold.shape: ", feat.shape)   #torch.Size([16, 576, 48, 48])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
        # print("why coord.shape: ", coord.shape)         #torch.Size([16, 2304, 2])
        # print("why feat_coord.shape: ", feat_coord.shape)       #torch.Size([16, 2, 48, 48])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                # print("why q_feat.shape: ", q_feat.shape)        #torch.Size([16, 2304, 576])   
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                # print("why q_coord.shape: ", q_coord.shape)     #torch.Size([16, 2304, 2])
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)
                # print("why inp.shape: ", inp.shape)             #torch.Size([16, 2304, 578])

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)
                    # print("why cell_decode[inp.shape]: ", inp.shape)            #torch.Size([16, 2304, 580])

                bs, q = coord.shape[:2]
                pred = self.imnets[indice](inp.view(bs * q, -1)).view(bs, q, -1)
                # print("why pred.shape: ", pred.shape)           #torch.Size([16, 2304, 3])
                preds.append(pred)
                # print("why preds.shape: ", preds.shape)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        # print("why ret.shape: ", ret.shape)           #torch.Size([16, 2304, 3])
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)      
        pred_list = []    
        # print("why coord: ", coord.shape)  
        # return self.query_rgb(coord, cell)
        if(why_kd_utils_not_tensor.is_train):
            # print("why train....")
            for i in range(why_kd_utils_not_tensor.mlp_num):
                sub_coord = coord[:, i, :, :]
                sub_cell = cell[:, i, :, :]
                pred = self.multi_query_rgb(sub_coord, i, sub_cell)
                pred_list.append(pred)
        else:
            print("why test....")
            print("cell.shape: ", cell.shape)
            batch_size, n, features = coord.shape
            s, r = divmod(n, why_kd_utils_not_tensor.mlp_num)
            current = 0
            for i in range(why_kd_utils_not_tensor.mlp_num):
                size = s + (i < r)
                print("size: ", size)
                sub_coord = coord[:, current:current+size, :]
                sub_cell = cell[:, current:current+size, :]
                print("cell.shape: ", cell.shape)
                pred = self.multi_query_rgb(sub_coord, i, sub_cell)
                pred_list.append(pred)
                current += size

        # 沿第二维度拼接（288*8=2304）
        final_pred = torch.cat(pred_list, dim=1)  
        # print("why final_pred: ", final_pred.shape)
        return final_pred