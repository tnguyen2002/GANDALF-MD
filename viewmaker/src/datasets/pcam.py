import os
import copy
import getpass
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import string
import math
import pandas as pd

import torch
import torch.utils.data as data
from torchvision import transforms, datasets
from src.datasets.root_paths import DATA_ROOTS

class PCAM(data.Dataset):
    NUM_CLASSES = 2
    NUM_CHANNELS = 3
    FILTER_SIZE = 32
    MULTI_LABEL = False

    def __init__(
            self, 
            root=DATA_ROOTS['PCAM'],
            train=True, 
            image_transforms=None, 
            alternate_label=False
        ):
        super().__init__()
        if not os.path.isdir(root):
            os.makedirs(root)
        self.alternate_label = alternate_label
        if self.alternate_label:
            self.NUM_CLASSES = 399
        if train:
            split = "train"
            self.df = pd.read_csv('/data5/xiluohe/pcamv1/camelyonpatch_level_2_split_train_meta.csv')
        else:
            split = "val"
            self.df = pd.read_csv('/data5/xiluohe/pcamv1/camelyonpatch_level_2_split_valid_meta.csv')
        self.dataset = datasets.PCAM(
            root, 
            split=split,
            download=True,
            transform=image_transforms,
        )
        self.dataset.targets = list(self.df['tumor_patch'])

    def __getitem__(self, index):
        # pick random number
        neg_index = np.random.choice(np.arange(self.__len__()))
        img_data, label = self.dataset.__getitem__(index)
        img2_data, _ = self.dataset.__getitem__(index)
        neg_data, _ = self.dataset.__getitem__(neg_index)
        # build this wrapper such that we can return index
        #print(img_data.shape)

        _, _, _, tumor, _, _, wsi = self.df.iloc[index]
        alt_label = wsi
        #print(wsi)

        if self.alternate_label:
            label = alt_label
        # if label < 0: print("NEGATIVE LABEL")
        # if label > 0: print("NEGATIVE LABEL")
        data = [index, img_data.float(), img2_data.float(), 
                neg_data.float(), label]
        return tuple(data)

    def __len__(self):
        #print(len(self.dataset))
        return len(self.dataset)