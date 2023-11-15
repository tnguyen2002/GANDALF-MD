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

class CIFAR10(data.Dataset):
    NUM_CLASSES = 10
    NUM_CHANNELS = 3
    FILTER_SIZE = 32
    MULTI_LABEL = False

    def __init__(
            self, 
            root=DATA_ROOTS['cifar10'],
            train=True, 
            image_transforms=None, 
            alternate_label=False
        ):
        super().__init__()
        if not os.path.isdir(root):
            os.makedirs(root)
        self.dataset = datasets.cifar.CIFAR10(
            root, 
            train=train,
            download=True,
            transform=image_transforms,
        )

    def __getitem__(self, index):
        # pick random number
        neg_index = np.random.choice(np.arange(self.__len__()))
        img_data, label = self.dataset.__getitem__(index)
        img2_data, _ = self.dataset.__getitem__(index)
        neg_data, _ = self.dataset.__getitem__(neg_index)
        # build this wrapper such that we can return index
        data = [index, img_data.float(), img2_data.float(), 
                neg_data.float(), label]
        return tuple(data)

    def __len__(self):
        return len(self.dataset)

def draw_mask(shape, x1, y1, x2, y2, width, height, x3=0, y3=0):
    mask = Image.new("RGB", (width, height), (0, 0, 0))
    maskdraw = ImageDraw.Draw(mask, "RGB") 
    if shape == 'rect': maskdraw.rectangle([(x1, y1),(x2, y2)], outline=None, width=1, fill=(255, 255, 255))
    elif shape == 'tri': maskdraw.polygon([(x1,y1),(x2,y2),(x3,y3)], outline=None, fill=(255, 255, 255))
    elif shape == 'ellipse': maskdraw.ellipse([(x1, y1),(x2, y2)], outline=None, width=1, fill=(255, 255, 255))
    return np.array(mask) / 255

def draw_rect(draw, width, height, r, g, b, scale):
    x1, y1 = (1-scale)*width/2 + scale*width/3, (1-scale)*height/2 + scale*height/3
    x2, y2 = (1-scale)*width/2 + scale*2*width/3, (1-scale)*height/2 + scale*2*height/3
    draw.rectangle([(x1, y1),(x2, y2)], outline=None, width=1, fill=(r, g, b, 180))
    return draw_mask('rect', x1, y1, x2, y2, width, height)

def draw_tri(draw, width, height, r, g, b, scale): 
    x1, y1 = width/2, (1-scale)*height/2 + scale*height/3
    x2, y2 = (1-scale)*width/2 + scale*width/3, (1-scale)*height/2 + scale*2*height/3
    x3, y3 = (1-scale)*width/2 + scale*2*width/3, (1-scale)*height/2 + scale*2*height/3
    draw.polygon([(x1,y1),(x2,y2),(x3,y3)], outline=None, fill=(r, g, b, 180))
    return draw_mask('tri', x1, y1, x2, y2, width, height, x3=x3, y3=y3)
    
def draw_ellipse(draw, width, height, r, g, b, scale):
    x1, y1 = (1-scale)*width/2 + scale*width/3, (1-scale)*height/2 + scale*height/3
    x2, y2 = (1-scale)*width/2 + scale*2*width/3, (1-scale)*height/2 + scale*2*height/3
    draw.ellipse([(x1, y1),(x2, y2)], outline=None, width=1, fill=(r, g, b, 180))
    return draw_mask('ellipse', x1, y1, x2, y2, width, height)


#Draws an opaque shape (rectangle, triangle, ellipse) somewhere around the center of the image
def add_shape(im, scale=1, index=0):
    draw = ImageDraw.Draw(im, "RGBA") 
    width, height = im.size

    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)

    shape_fns = [draw_rect, draw_tri, draw_ellipse]
    shape_idx = random.randint(0, 2)
    shape_mask = shape_fns[shape_idx](draw, width, height, r, g, b, scale)
    return im, shape_idx, shape_mask

class CIFAR10Shapes(data.Dataset):
    NUM_CLASSES = 10
    NUM_CHANNELS = 3
    FILTER_SIZE = 32
    MULTI_LABEL = False

    def __init__(
            self, 
            shape_size=1,
            root=DATA_ROOTS['cifar10'],
            train=True, 
            image_transforms=None,
            alternate_label=False, # Whether to return the kind of perturbation (e.g. shape) as a label.
        ):
        super().__init__()
        if not os.path.isdir(root):
            os.makedirs(root)
        self.dataset = datasets.cifar.CIFAR10(
            root, 
            train=train,
            download=True,
            transform=None,
        )
        self.image_transforms = image_transforms
        self.shape_size = shape_size
        self.alternate_label = alternate_label
        if alternate_label:
            self.NUM_CLASSES = 3

    def __getitem__(self, index):
        # pick random number
        
        neg_index = np.random.choice(np.arange(self.__len__()))
        img_data, label = self.dataset.__getitem__(index)
        neg_data, _ = self.dataset.__getitem__(neg_index)

        img_data, shape_label, mask = add_shape(img_data, self.shape_size, index)
        neg_data, _, _ = add_shape(neg_data, self.shape_size, index)

        if self.alternate_label:
            label = shape_label

        # build this wrapper such that we can return index
        data = [index, 
                self.image_transforms(img_data).float(), 
                self.image_transforms(img_data).float(), 
                self.image_transforms(neg_data).float(), label, mask]
        return tuple(data)

    def __len__(self):
        return len(self.dataset)

class CIFAR10ShapesManualFeatureDropout(data.Dataset):
    NUM_CLASSES = 10
    NUM_CHANNELS = 3
    FILTER_SIZE = 32
    MULTI_LABEL = False

    def __init__(
            self, 
            shape_size=1,
            root=DATA_ROOTS['cifar10'],
            train=True, 
            image_transforms=None,
            alternate_label=False, # Whether to return the kind of perturbation (e.g. shape) as a label.
            feat_dropout_prob=0
        ):
        super().__init__()
        if not os.path.isdir(root):
            os.makedirs(root)
        self.dataset = datasets.cifar.CIFAR10(
            root, 
            train=train,
            download=True,
            transform=None,
        )
        self.image_transforms = image_transforms
        self.shape_size = shape_size
        self.alternate_label = alternate_label
        if alternate_label:
            self.NUM_CLASSES = 3

        self.feat_dropout_prob = feat_dropout_prob

    def __getitem__(self, index):
        # pick random number
        
        neg_index = np.random.choice(np.arange(self.__len__()))
        img_data, label = self.dataset.__getitem__(index)
        neg_data, _ = self.dataset.__getitem__(neg_index)

        img_data1, shape_label = add_shape(copy.deepcopy(img_data), self.shape_size, index)

        p = random.random()
        img_data2, _ = add_shape(copy.deepcopy(img_data), self.shape_size, index) if p < self.feat_dropout_prob else (img_data1, shape_label)

        neg_data, _ = add_shape(neg_data, self.shape_size, index)

        if self.alternate_label:
            label = shape_label

        # build this wrapper such that we can return index
        data = [index, 
                self.image_transforms(img_data1).float(), 
                self.image_transforms(img_data2).float(), 
                self.image_transforms(neg_data).float(), label]
        
        return tuple(data)

    def __len__(self):
        return len(self.dataset)

class CIFAR10Letters(data.Dataset):
    NUM_CLASSES = 10
    NUM_CHANNELS = 3
    FILTER_SIZE = 32
    MULTI_LABEL = False
    LETTERS = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

    def __init__(
            self, 
            root=DATA_ROOTS['cifar10'],
            train=True, 
            image_transforms=None,
            alternate_label=False 
        ):
        super().__init__()
        if not os.path.isdir(root):
            os.makedirs(root)
        self.dataset = datasets.cifar.CIFAR10(
            root, 
            train=train,
            download=True,
            transform=None,
        )
        self.image_transforms = image_transforms
        self.alternate_label = alternate_label
        if self.alternate_label:
            self.NUM_CLASSES = 26
    
    #Draws a letter somewhere in each of the four quadrants of the image
    def add_char(self, im, index=0):
        draw = ImageDraw.Draw(im, "RGBA")
        width, height = im.size

        r = random.randint(0, 255) 
        g = random.randint(0, 255) 
        b = random.randint(0, 255) 

        letter_idx = random.randint(0, 25)  
        letter = self.LETTERS[letter_idx]
        
        x1, y1 = random.randint(0, math.floor(
            width/5)), random.randint(0, math.floor(height/5))
        x2, y2 = random.randint(width/2, math.floor(2*width/3)
                                ), random.randint(0, math.floor(height/5))
        x3, y3 = random.randint(0, math.floor(
            width/5)), random.randint(height/2, math.floor(2*height/3))
        x4, y4 = random.randint(width/2, math.floor(2*width/3)
                                ), random.randint(height/2, math.floor(2*height/3))
        
        font = ImageFont.truetype("./scripts/arial.ttf", 12)

        draw.text((x1, y1), letter, (r, g, b, 255), font=font)
        draw.text((x2, y2), letter, (r, g, b, 255), font=font)
        draw.text((x3, y3), letter, (r, g, b, 255), font=font)
        draw.text((x4, y4), letter, (r, g, b, 255), font=font)

        mask = Image.new("RGB", (width, height), (0, 0, 0))
        maskdraw = ImageDraw.Draw(mask, "RGB") 
        maskdraw.text((x1, y1), letter, (255, 255, 255, 255), font=font)
        maskdraw.text((x2, y2), letter, (255, 255, 255, 255), font=font)
        maskdraw.text((x3, y3), letter, (255, 255, 255, 255), font=font)
        maskdraw.text((x4, y4), letter, (255, 255, 255, 255), font=font)


        return im, letter_idx

    def __getitem__(self, index):
        # pick random number
        
        neg_index = np.random.choice(np.arange(self.__len__()))
        img_data, label = self.dataset.__getitem__(index)
        neg_data, _ = self.dataset.__getitem__(neg_index)

        img_data, letter_idx = self.add_char(img_data, index)
        neg_data, _ = self.add_char(neg_data, index)

        if self.alternate_label:
            label = letter_idx

        # build this wrapper such that we can return index
        data = [index, 
                self.image_transforms(img_data).float(), 
                self.image_transforms(img_data).float(), 
                self.image_transforms(neg_data).float(), label]
        return tuple(data)

    def __len__(self):
        return len(self.dataset)

class CIFAR10LettersManualFeatureDropout(data.Dataset):
    NUM_CLASSES = 10
    NUM_CHANNELS = 3
    FILTER_SIZE = 32
    MULTI_LABEL = False
    LETTERS = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

    def __init__(
            self, 
            root=DATA_ROOTS['cifar10'],
            train=True, 
            image_transforms=None,
            alternate_label=False,
            feat_dropout_prob=0
        ):
        super().__init__()
        if not os.path.isdir(root):
            os.makedirs(root)
        self.dataset = datasets.cifar.CIFAR10(
            root, 
            train=train,
            download=True,
            transform=None,
        )
        self.image_transforms = image_transforms
        self.alternate_label = alternate_label
        if self.alternate_label:
            self.NUM_CLASSES = 26
        self.feat_dropout_prob = feat_dropout_prob
    
    #Draws a letter somewhere in each of the four quadrants of the image
    def add_char(self, im, index=0):
        draw = ImageDraw.Draw(im, "RGBA")
        width, height = im.size

        r = random.randint(0, 255) 
        g = random.randint(0, 255) 
        b = random.randint(0, 255) 

        letter_idx = random.randint(0, 25)  
        letter = self.LETTERS[letter_idx]
        
        x1, y1 = random.randint(0, math.floor(
            width/5)), random.randint(0, math.floor(height/5))
        x2, y2 = random.randint(width/2, math.floor(2*width/3)
                                ), random.randint(0, math.floor(height/5))
        x3, y3 = random.randint(0, math.floor(
            width/5)), random.randint(height/2, math.floor(2*height/3))
        x4, y4 = random.randint(width/2, math.floor(2*width/3)
                                ), random.randint(height/2, math.floor(2*height/3))
        
        font = ImageFont.truetype("./scripts/arial.ttf", 12)

        draw.text((x1, y1), letter, (r, g, b, 255), font=font)
        draw.text((x2, y2), letter, (r, g, b, 255), font=font)
        draw.text((x3, y3), letter, (r, g, b, 255), font=font)
        draw.text((x4, y4), letter, (r, g, b, 255), font=font)
        return im, letter_idx

    def __getitem__(self, index):
        # pick random number
        
        neg_index = np.random.choice(np.arange(self.__len__()))
        img_data, label = self.dataset.__getitem__(index)
        neg_data, _ = self.dataset.__getitem__(neg_index)

        img_data1, letter_idx = self.add_char(copy.deepcopy(img_data), index)
        p = random.random()
        img_data2, _ = self.add_char(copy.deepcopy(img_data), index) if p < self.feat_dropout_prob else (img_data1, letter_idx)
        neg_data, _ = self.add_char(neg_data, index)

        if self.alternate_label:
            label = letter_idx

        # build this wrapper such that we can return index
        data = [index, 
                self.image_transforms(img_data1).float(), 
                self.image_transforms(img_data2).float(), 
                self.image_transforms(neg_data).float(), label]
        return tuple(data)

    def __len__(self):
        return len(self.dataset)

class CIFAR10Digits(data.Dataset):
    NUM_CLASSES = 10
    NUM_CHANNELS = 3
    FILTER_SIZE = 32
    MULTI_LABEL = False

    def __init__(
            self, 
            root=DATA_ROOTS['cifar10'],
            train=True, 
            image_transforms=None, 
            alternate_label=False,  # Whether to return digit label.
        ):
        super().__init__()
        if not os.path.isdir(root):
            os.makedirs(root)
        self.dataset = datasets.cifar.CIFAR10(
            root,
            train=train,
            download=True,
            transform=None,
        )
        self.image_transforms = image_transforms
        self.alternate_label = alternate_label  # Num classes stays the same.
        self.mnist = MNIST(root=DATA_ROOTS['meta_mnist'], train=True, download=True, transform=None)

    #Draws a MNIST digit somewhere in each of the four quadrants of the image
    def add_digits(self, im, index=0):
        digit, digit_label = self.mnist[random.randint(0, len(self.mnist) - 1)]
        digit = digit.convert("RGBA")
        digit = digit.resize((10, 10))
        datas = digit.getdata()
        
        newData = []
        for item in datas:
            if item[0] == 0 or item[1] == 0 or item[2] == 0:
                newData.append((255, 255, 255, 0)) # make black background pixels transparent
            else:
                newData.append(item)
        digit.putdata(newData)

        mask = Image.new("RGB", im.size, (0, 0, 0))

        # add 2x2 digits to center of CIFAR10 img
        for x in range(2):
            for y in range(2):
                im.paste(digit, (4+(x*12), 4+(y*12)), mask = digit)
                mask.paste(digit, (4+(x*12), 4+(y*12)), mask = digit)
        mask = np.array(mask)
        thresh = 63
        mask[mask > thresh] = 255
        mask[mask <= thresh] = 0
        mask = mask / 255

        return im, digit_label, mask
        
    def __getitem__(self, index):
        # pick random number
        neg_index = np.random.choice(np.arange(self.__len__()))
        img_data, label = self.dataset.__getitem__(index)
        neg_data, _ = self.dataset.__getitem__(neg_index)

        img_data, digit_label, mask = self.add_digits(img_data, index)
        neg_data, _, _ = self.add_digits(neg_data, index)

        if self.alternate_label:
            label = digit_label

        # build this wrapper such that we can return index
        # data = [index, 
        #         self.image_transforms(img_data).float(), 
        #         self.image_transforms(img_data).float(), 
        #         self.image_transforms(neg_data).float(), label, mask]
        data = [index, 
                self.image_transforms(img_data).float(), 
                self.image_transforms(img_data).float(), 
                self.image_transforms(neg_data).float(), label]
        return tuple(data)

    def __len__(self):
        return len(self.dataset)


class CIFAR10DigitsManualFeatureDropout(data.Dataset):
    NUM_CLASSES = 10
    NUM_CHANNELS = 3
    FILTER_SIZE = 32
    MULTI_LABEL = False

    def __init__(
            self, 
            root=DATA_ROOTS['cifar10'],
            train=True, 
            image_transforms=None, 
            alternate_label=False,  # Whether to return digit label.
            feat_dropout_prob=0
        ):
        super().__init__()
        if not os.path.isdir(root):
            os.makedirs(root)
        self.dataset = datasets.cifar.CIFAR10(
            root,
            train=train,
            download=True,
            transform=None,
        )
        self.image_transforms = image_transforms
        self.alternate_label = alternate_label  # Num classes stays the same.
        self.mnist = MNIST(root=DATA_ROOTS['meta_mnist'], train=True, download=True, transform=None)
        self.feat_dropout_prob = feat_dropout_prob

    #Draws a MNIST digit somewhere in each of the four quadrants of the image
    def add_digits(self, im, index=0):
        digit, digit_label = self.mnist[random.randint(0, len(self.mnist) - 1)]
        digit = digit.convert("RGBA")
        digit = digit.resize((10, 10))
        datas = digit.getdata()
        
        newData = []
        for item in datas:
            if item[0] == 0 or item[1] == 0 or item[2] == 0:
                newData.append((255, 255, 255, 0)) # make black background pixels transparent
            else:
                newData.append(item)
        digit.putdata(newData)

        # add 2x2 digits to center of CIFAR10 img
        for x in range(2):
            for y in range(2):
                im.paste(digit, (4+(x*12), 4+(y*12)), mask = digit)
        return im, digit_label
        
    def __getitem__(self, index):
        # pick random number
        neg_index = np.random.choice(np.arange(self.__len__()))
        img_data, label = self.dataset.__getitem__(index)
        neg_data, _ = self.dataset.__getitem__(neg_index)

        img_data1, digit_label = self.add_digits(copy.deepcopy(img_data), index)
        p = random.random()
        img_data2, _ = self.add_digits(copy.deepcopy(img_data), index) if p < self.feat_dropout_prob else (img_data1, digit_label)
        neg_data, _ = self.add_digits(neg_data, index)

        if self.alternate_label:
            label = digit_label

        # build this wrapper such that we can return index
        data = [index, 
                self.image_transforms(img_data1).float(), 
                self.image_transforms(img_data2).float(), 
                self.image_transforms(neg_data).float(), label]
        return tuple(data)

    def __len__(self):
        return len(self.dataset)

class CIFAR10Corners(data.Dataset):
    '''Creates a four-corners mosaic of different CIFAR images.'''
    
    NUM_CLASSES = 10
    NUM_CHANNELS = 3
    FILTER_SIZE = 32
    MULTI_LABEL = False

    def __init__(
        self,
        root=DATA_ROOTS['cifar10'],
        train=True,
        image_transforms=None,
    ):
        super().__init__()
        if not os.path.isdir(root):
            os.makedirs(root)
        self.dataset = datasets.cifar.CIFAR10(
            root,
            train=train,
            download=True # Don't apply transformations yet
        )
        self.train = train
        self.transforms = image_transforms

    def get_random_cifar(self):
        idx = random.randint(0, len(self.dataset) - 1)
        return self.dataset[idx][0]


    def paste_random_cifar_square(self, base_img, x, y):
        img = self.get_random_cifar()
        img_crop = img.crop((x, y, x + 16, y + 16))
        base_img.paste(img_crop, (x, y))
        return base_img


    def get_cifar_corners(self):
        base_img = self.get_random_cifar()
        base_img = self.paste_random_cifar_square(base_img, 16, 0)
        base_img = self.paste_random_cifar_square(base_img, 16, 16)
        base_img = self.paste_random_cifar_square(base_img, 0, 16)
        return base_img


    def __getitem__(self, index):
        if not self.train:
            img_data, label = self.dataset.__getitem__(index)
            img2_data, _ = self.dataset.__getitem__(index)
            # build this wrapper such that we can return index
            data = [index, self.transforms(img_data).float(), self.transforms(img2_data).float(), label, label]
        else:
            img_data = self.get_cifar_corners()
            img2_data = img_data
            # No labels for pano.
            data = [index, self.transforms(img_data).float(), self.transforms(img2_data).float(), 0, 0]
        return tuple(data)

    def __len__(self):
        return len(self.dataset)
