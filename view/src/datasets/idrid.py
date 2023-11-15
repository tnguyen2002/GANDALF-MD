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

class IDRiD(data.Dataset):
    NUM_CLASSES = 3
    NUM_CHANNELS = 3
    FILTER_SIZE = 32
    MULTI_LABEL = False

    def __init__(
            self, 
            root=DATA_ROOTS['IDRiD'],
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
        self.split = "train" if train else "val"
        if self.alternate_label:
            self.NUM_CLASSES = 5
        if train: 
            self.df = pd.read_csv(os.path.join(root, 'train.csv'))
        else:
            self.df = pd.read_csv(os.path.join(root, 'val.csv'))
        self.dataset = self.df
        self.dataset.targets = list(self.df['ret_grade'])

    def __getitem__(self, index):
        # pick random number
        neg_index = np.random.choice(np.arange(self.__len__()))
        image_id, ret_grade, rme =  self.df.iloc[index]

        img_data_path = self.root + "/images/" + self.split + "/" + image_id + ".jpg"
        img_data = Image.open(os.path.join(self.root, img_data_path)).convert('RGB')
        img2_data = Image.open(os.path.join(self.root, img_data_path)).convert('RGB')
        neg_image_id = self.df.iloc[neg_index, 0]
        neg_data_path = self.root + "/images/" + self.split + "/" + neg_image_id + ".jpg"
        neg_data = Image.open(os.path.join(self.root, neg_data_path)).convert('RGB')
        # build this wrapper such that we can return index

        label = rme
        alt_label = ret_grade

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
        return len(self.dataset)