import os.path as osp
from utils.images import load_image
import matplotlib.pyplot as plt
import os

import cv2
import numpy as np
import torch
import random
from torch.utils.data.dataset import Dataset

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')


## Dataloader for deep learning
class NUDTSIRSTSetLoader(Dataset):
    def __init__(self, base_dir='../data/NUDT-SIRST/', mode='test'):
        super(NUDTSIRSTSetLoader).__init__()
        self.mode = mode

        if mode == 'trainval':
            txtfile = 'train_NUDT-SIRST.txt'
        elif mode == 'test':
            txtfile = 'test_NUDT-SIRST.txt'
        else:
            raise NotImplementedError

        self.list_dir = osp.join(base_dir, 'img_idx', txtfile)
        self.imgs_dir = osp.join(base_dir, 'images')
        self.label_dir = osp.join(base_dir, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name + '.png')
        label_path = osp.join(self.label_dir, name + '.png')

        img = load_image(img_path)
        mask = load_image(label_path)
        img = np.array(img, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0

        if self.mode == 'trainval':
            h = 256
            w = 256
            img = cv2.resize(img, dsize=(h, w), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, dsize=(h, w), interpolation=cv2.INTER_NEAREST)
        else:
            h, w = img.shape

        img, mask = img[np.newaxis, :], mask[np.newaxis, :]
        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))

        if self.mode == 'trainval':
            return img, mask
        else:
            return img, mask, [h, w], self.names[i]

    def __len__(self):
        return len(self.names)


class IRSTD1KSetLoader(Dataset):
    def __init__(self, base_dir='../data/IRSTD-1K/', mode='test'):
        super(IRSTD1KSetLoader).__init__()
        self.mode = mode

        if mode == 'trainval':
            txtfile = 'train_IRSTD-1K.txt'
        elif mode == 'test':
            txtfile = 'test_IRSTD-1K.txt'
        else:
            raise NotImplementedError

        self.list_dir = osp.join(base_dir, 'img_idx', txtfile)
        self.imgs_dir = osp.join(base_dir, 'images')
        self.label_dir = osp.join(base_dir, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name + '.png')
        label_path = osp.join(self.label_dir, name + '.png')

        img = load_image(img_path)
        mask = load_image(label_path)
        img = np.array(img, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0

        h, w = img.shape  # All the pictures are 512 x 512

        img, mask = img[np.newaxis, :], mask[np.newaxis, :]
        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))

        if self.mode == 'trainval':
            return img, mask
        else:
            return img, mask, [h, w], self.names[i]

    def __len__(self):
        return len(self.names)


class SIRSTAugSetLoader(Dataset):
    def __init__(self, base_dir='../data/sirst_aug/', mode='test'):
        super(SIRSTAugSetLoader).__init__()
        self.mode = mode

        if mode == 'trainval':
            txtfile = 'train.txt'
        elif mode == 'test':
            txtfile = 'test.txt'
        else:
            raise NotImplementedError

        self.list_dir = osp.join(base_dir, 'img_idx', txtfile)
        self.imgs_dir = osp.join(base_dir, 'images')
        self.label_dir = osp.join(base_dir, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name + '.png')
        label_path = osp.join(self.label_dir, name + '_mask.png')

        img = load_image(img_path)
        mask = load_image(label_path)
        img = np.array(img, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0

        if self.mode == 'trainval':
            h = 256
            w = 256
            img = cv2.resize(img, dsize=(h, w), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, dsize=(h, w), interpolation=cv2.INTER_NEAREST)
        else:
            h, w = img.shape

        img, mask = img[np.newaxis, :], mask[np.newaxis, :]
        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))

        if self.mode == 'trainval':
            return img, mask
        else:
            return img, mask, [h, w], self.names[i]

    def __len__(self):
        return len(self.names)


class GenericSIRSTSetLoader(Dataset):
    """RDANet 布局：images + masks + img_idx。用于 NUAA-SIRST / SIRST-V2。"""

    def __init__(
        self,
        base_dir,
        mode='test',
        train_txt=None,
        test_txt=None,
        mask_suffix='',
        train_size=256,
    ):
        super(GenericSIRSTSetLoader, self).__init__()
        self.mode = mode
        self.mask_suffix = mask_suffix
        self.train_size = int(train_size)

        if mode == 'trainval':
            txtfile = train_txt or 'train.txt'
        elif mode == 'test':
            txtfile = test_txt or 'test.txt'
        else:
            raise NotImplementedError(mode)

        self.list_dir = osp.join(base_dir, 'img_idx', txtfile)
        self.imgs_dir = osp.join(base_dir, 'images')
        self.label_dir = osp.join(base_dir, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines() if line.strip()]

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name + '.png')
        label_path = osp.join(self.label_dir, name + self.mask_suffix + '.png')

        img = load_image(img_path)
        mask = load_image(label_path)
        img = np.array(img, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0

        if self.mode == 'trainval':
            h = self.train_size
            w = self.train_size
            img = cv2.resize(img, dsize=(h, w), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, dsize=(h, w), interpolation=cv2.INTER_NEAREST)
        else:
            h, w = img.shape

        img, mask = img[np.newaxis, :], mask[np.newaxis, :]
        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))

        if self.mode == 'trainval':
            return img, mask
        return img, mask, [h, w], self.names[i]

    def __len__(self):
        return len(self.names)


def build_dataset(dataset_name, base_dir=None, mode='trainval'):
    """统一入口：官方三集 + Ex04 用的 NUAA / SIRST-V2。"""
    name = dataset_name
    if base_dir is None:
        mapping = {
            'NUDT-SIRST': './data/NUDT-SIRST/',
            'IRSTD-1K': './data/IRSTD-1K/',
            'SIRST-aug': './data/sirst_aug/',
            'NUAA-SIRST': './data/NUAA-SIRST/',
            'SIRST-V2': './data/SIRST-V2/',
        }
        if name not in mapping:
            raise NotImplementedError(name)
        base_dir = mapping[name]

    if name == 'NUDT-SIRST':
        return NUDTSIRSTSetLoader(base_dir=base_dir, mode=mode)
    if name == 'IRSTD-1K':
        return IRSTD1KSetLoader(base_dir=base_dir, mode=mode)
    if name == 'SIRST-aug':
        return SIRSTAugSetLoader(base_dir=base_dir, mode=mode)
    if name in ('NUAA-SIRST', 'SIRST-V2'):
        return GenericSIRSTSetLoader(
            base_dir=base_dir,
            mode=mode,
            train_txt='train_%s.txt' % name,
            test_txt='test_%s.txt' % name,
            mask_suffix='',
            train_size=256,
        )
    raise NotImplementedError(name)
