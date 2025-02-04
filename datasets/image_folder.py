import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register


@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, root_path_mask=None, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache
        self.is_mask = False

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        if(root_path_mask):
            self.files_mask = []
            self.is_mask = True
        for filename in filenames:
            file = os.path.join(root_path, filename)
            if(root_path_mask):
                mask = os.path.join(root_path_mask, filename)


            if cache == 'none':
                self.files.append(file)
                if(root_path_mask):
                    self.files_mask.append(mask)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                self.files.append(transforms.ToTensor()(
                    Image.open(file).convert('RGB')))
                if(root_path_mask):
                    self.files_mask.append(transforms.ToTensor()(
                        Image.open(mask)))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]
        if(self.is_mask):
            mask = self.files_mask[idx % len(self.files)]
        # print(f"x: {x.shape}, mask: {mask.shape}")

        if self.cache == 'none':
            if(self.is_mask):
                # print("why imagefloled ")
                return (transforms.ToTensor()(Image.open(x).convert('RGB')), transforms.ToTensor()(
                        Image.open(mask)))
  
            else:
                return transforms.ToTensor()(Image.open(x).convert('RGB'))
            
        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            if(self.is_mask):
                return (x, mask)
            else:
                return (x, None)


@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, root_path_mask, **kwargs):
        self.is_mask = False
        if(root_path_mask):
            self.dataset_1 = ImageFolder(root_path_1, root_path_mask,  **kwargs)
            self.dataset_2 = ImageFolder(root_path_2, None, **kwargs)
            self.is_mask = True
        else:
            self.dataset_1 = ImageFolder(root_path_1, None,  **kwargs)
            self.dataset_2 = ImageFolder(root_path_2, None, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        if(self.is_mask):
            img, mask = self.dataset_1[idx]
            return img, mask, self.dataset_2[idx]
        else:
            return self.dataset_1[idx], None, self.dataset_2[idx]
