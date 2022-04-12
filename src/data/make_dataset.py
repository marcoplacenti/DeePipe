import os
import pandas as pd
import numpy as np

from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, data_dir, channels, transform=None, target_transform=None):
        self.img_dir = os.path.realpath(os.path.join(os.getcwd(), data_dir))
        self.image_labels = pd.DataFrame(self.make_annotations(),
                                columns=["img_path", "label"])
        self.transform = transform
        self.target_transform = target_transform

        self.color = (cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR)
        

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_labels.iloc[idx, 0])
        image = cv2.imread(img_path, self.color)
        if len(image.shape) == 2:
            image = np.reshape(image, (image.shape[0], image.shape[1], 1))
        image = np.transpose(image, axes=(2, 0, 1))
        image = torch.from_numpy(image)/255
        label = self.image_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def make_annotations(self):
        annotations = []
        class_idx = -1
        for idx, dir in enumerate(os.listdir(self.img_dir)):
            if os.path.isdir("/".join([self.img_dir, dir])):
                class_idx += 1
                for img in os.listdir("/".join([self.img_dir, dir])):
                    annotations.append(["/".join([dir, img]), class_idx])

        return annotations
            
    def get_num_classes(self):
        return len(set(self.image_labels['label']))

