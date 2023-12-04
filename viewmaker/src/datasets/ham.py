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

class HAM(data.Dataset):
    NUM_CLASSES = 7
    NUM_CHANNELS = 3
    FILTER_SIZE = 32
    MULTI_LABEL = False

    def __init__(
            self, 
            root=DATA_ROOTS['HAM'],
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
            self.NUM_CLASSES = 3
        if train: 
            self.df = pd.read_csv(os.path.join(root, 'HAM10000_metadata_train.csv'))
        else:
            self.df = pd.read_csv(os.path.join(root, 'HAM10000_metadata_val.csv'))
        self.dataset = self.df
        self.dataset.targets = list(self.df['dx_id'])

    def __getitem__(self, index):
        # pick random number
        neg_index = np.random.choice(np.arange(self.__len__()))
        lesion_id, image_id, dx, dx_type, age, sex, localization, dataset, dx_id, dx_type_id, sex_id, localization_id, dataset_id, split =  self.df.iloc[index]
        
        img_data_path = self.root + "/images/" + image_id + ".jpg"
        img_data = Image.open(os.path.join(self.root, img_data_path)).convert('RGB')
        img2_data = Image.open(os.path.join(self.root, img_data_path)).convert('RGB')
        neg_image_id = self.df.iloc[neg_index, 1]
        neg_data_path = self.root + "/images/" + neg_image_id + ".jpg"
        neg_data = Image.open(os.path.join(self.root, neg_data_path)).convert('RGB')
        # build this wrapper such that we can return index

        label = dx_id
        alt_label = sex_id

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

class HAM_semisupervised(data.Dataset):
    NUM_CLASSES = 1
    NUM_CHANNELS = 3
    FILTER_SIZE = 32
    MULTI_LABEL = False

    def __init__(
            self, 
            root=DATA_ROOTS['ham_generated_and_real_diffaug'], #'/home/anhn/pretrained_data/HAM/Ham_diffaug'
            train=True, 
            image_transforms=None, 
            alternate_label=False
        ):
        super().__init__()
        if not os.path.isdir(root):
            os.makedirs(root)
        
        self.root = root
        self.transform = image_transforms
        self.train = train
        self.df = pd.read_csv(os.path.join(root, 'HAM_semisupervised_train.csv'))
        self.dataset = self.df
        self.dataset.targets = [int(i) for i in self.df['label']]

    def __getitem__(self, index):
        # pick random number
        #train
         # pick random number
        neg_index = np.random.choice(np.arange(self.__len__()))
        image_path, label =  self.df.iloc[index]
        neg_image_path, label =  self.df.iloc[neg_index]
        
        img_data_path = self.root + "/" + image_path
        img_data = Image.open(os.path.join(self.root, img_data_path)).convert('RGB')
        img2_data = Image.open(os.path.join(self.root, img_data_path)).convert('RGB')

        neg_image_id = self.df.iloc[neg_index, 1]
        neg_data_path = self.root + "/" + neg_image_path
        neg_data = Image.open(os.path.join(self.root, neg_data_path)).convert('RGB')
        # build this wrapper such that we can return index

        # label = dx_id
        # alt_label = sex_id

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
    
        data = [index, img_data.float(), img2_data.float(), neg_data.float(), label]
        return tuple(data)

    def __len__(self):
        return len(self.df)