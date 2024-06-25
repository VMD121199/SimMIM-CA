# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

import math
import random
import numpy as np

import torch
import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torch.utils.data._utils.collate import default_collate
from torchvision.datasets import ImageFolder
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from PIL import Image
import os

class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask


class SimMIMTransform:
    def __init__(self, config):
        self.transform_img = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ])
 
        if config.MODEL.TYPE == 'swin':
            model_patch_size=config.MODEL.SWIN.PATCH_SIZE
        elif config.MODEL.TYPE == 'vit':
            model_patch_size=config.MODEL.VIT.PATCH_SIZE
        else:
            raise NotImplementedError
        
        self.mask_generator = MaskGenerator(
            input_size=config.DATA.IMG_SIZE,
            mask_patch_size=config.DATA.MASK_PATCH_SIZE,
            model_patch_size=model_patch_size,
            mask_ratio=config.DATA.MASK_RATIO,
        )
    
    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()
        
        return img, mask

class SimMIMTransformCustom(SimMIMTransform):
    def __init__(self, config):
        super().__init__(config)
        custom_default_mean = (0.485, 0.456, 0.406, 0.435)
        custom_default_std = (0.229, 0.224, 0.225, 0.234)
        self.transform_img = T.Compose([
            T.Resize(config.DATA.IMG_SIZE),
            T.CenterCrop(config.DATA.IMG_SIZE),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(custom_default_mean),std=torch.tensor(custom_default_std)),
        ])

class RS_Dataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.imgs = self._load_data()
    
    def _load_data(self):
        imgs = []
        for img in os.listdir(self.root):
            if img.endswith('.png') or img.endswith('.jpg') or img.endswith('.jpeg'):
                imgs.append(os.path.join(self.root, img))
        return imgs
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img

# Early fusion
class RS_multimodal_dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_pairs = self._get_image_pairs()

    def _get_image_pairs(self):
        image_pairs = []
        for filename in os.listdir(self.root_dir):
            if filename.startswith("co") and filename.endswith(".png"):
                rgb_path = os.path.join(self.root_dir, filename)
                ir_filename = filename.replace("co", "ir")
                ir_path = os.path.join(self.root_dir, ir_filename)
                if os.path.exists(ir_path):
                    image_pairs.append((rgb_path, ir_path))
        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        rgb_path, ir_path = self.image_pairs[idx]
        pil2tensor = T.Compose([T.PILToTensor()])
        # Load and convert to RGB
        try:
            rgb_image = pil2tensor(Image.open(rgb_path).convert("RGB"))
            ir_image = pil2tensor(Image.open(ir_path).convert('L'))
            combined_image = torch.cat([rgb_image, ir_image], dim=0) 
        except:
            breakpoint()
        # Combine into 4-channel tensor (Early Fusion)
        combined_image = T.ToPILImage()(combined_image)
        if self.transform:
            combined_image = self.transform(combined_image)
        return combined_image

def collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(default_collate([batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret


def build_loader_simmim(config, logger):
    transform = SimMIMTransform(config)
    transform_custom = SimMIMTransformCustom(config)
    logger.info(f'Pre-train data transform:\n{transform}')
    # dataset = ImageFolder(config.DATA.DATA_PATH, transform)
    dataset = RS_Dataset(config.DATA.DATA_PATH, transform)
    if config.MULTIMODAL:
        dataset = RS_multimodal_dataset(config.DATA.DATA_PATH, transform_custom)
    logger.info(f'Build dataset: train images = {len(dataset)}')
    # sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, config.DATA.BATCH_SIZE, sampler=sampler, num_workers=config.DATA.NUM_WORKERS, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    return dataloader
