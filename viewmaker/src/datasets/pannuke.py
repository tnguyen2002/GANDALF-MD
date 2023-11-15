import os
import copy
import getpass
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import string
import math

import torch
import torch.utils.data as data
from torchvision import transforms, datasets
from src.datasets.root_paths import DATA_ROOTS
from torchvision.datasets import MNIST
import pandas as pd

class PanNuke(data.Dataset):
    NUM_CLASSES = 19
    NUM_CHANNELS = 3
    FILTER_SIZE = 32
    MULTI_LABEL = False

    def __init__(
            self, 
            root=DATA_ROOTS['PanNuke'],
            train=True, 
            image_transforms=None, 
            alternate_label=False
        ):
        super().__init__()
        if not os.path.isdir(root):
            os.makedirs(root)
        
        self.root = root
        self.transform = image_transforms
        self.alternate_label = alternate_label
        if self.alternate_label:
            self.NUM_CLASSES = 6
        if train: 
            self.dataset = np.load(os.path.join(root, 'train.npz'))
            self.images = self.dataset['images']
            self.body_parts = self.dataset['body_parts']
            self.cell_type = self.dataset['cell_types']
        else:
            self.dataset = np.load(os.path.join(root, 'val.npz'))
            self.images = self.dataset['images']
            self.body_parts = self.dataset['body_parts']
            self.cell_type = self.dataset['cell_types']
            
        self.dataset.targets = list(self.body_parts)

    def __getitem__(self, index):
        # pick random number
        neg_index = np.random.choice(np.arange(self.__len__()))

        image =  self.images[index].astype(np.uint8)
        body_part =  self.body_parts[index]
        cell_type =  self.cell_type[index]
        neg_image = self.images[neg_index].astype(np.uint8)

        img_data = Image.fromarray(image)
        img2_data = Image.fromarray(image)
        label = body_part
        alt_label = cell_type
        neg_data = Image.fromarray(neg_image)

        if self.transform:
            img_data = self.transform(img_data)
            img2_data = self.transform(img2_data)
            neg_data = self.transform(neg_data)
        else:
            resize_transforms = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])
            img_data = resize_transforms(img_data)
            img2_data = resize_transforms(img2_data)
            neg_data = resize_transforms(neg_data)
        
        if self.alternate_label:
            label = alt_label

        data = [index, img_data.float(), img2_data.float(), 
                neg_data.float(), label]
        return tuple(data)

    def __len__(self):
        return len(self.body_parts)