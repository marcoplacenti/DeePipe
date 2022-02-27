import os
import pandas as pd
import numpy as np

import cv2

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None):
        self.img_dir = data_dir
        self.image_labels = pd.DataFrame(self.make_annotations(self.img_dir),
                                columns=["img_path", "label"])
        self.transform = transform
        self.target_transform = target_transform
        

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_labels.iloc[idx, 0])
        rgb_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        rgb_image = np.transpose(rgb_image, axes=(2, 0, 1))
        image = torch.from_numpy(rgb_image)/255
        label = self.image_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def make_annotations(self, data_dir):
        return [["/".join([dir, img]), idx] \
            for idx, dir in enumerate(os.listdir(data_dir)) \
                for img in os.listdir("/".join([data_dir, dir]))]

    def get_num_classes(self):
        return len(set(self.image_labels['label']))

    


