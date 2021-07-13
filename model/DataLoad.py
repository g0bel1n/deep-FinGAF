import os
import pandas as pd
import numpy as np
import torch.utils.data
from torchvision.io import read_image

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, labels_file, img_dir):
        self.img_labels = pd.read_csv(labels_file, delimiter=";")
        self.img_dir = img_dir
        self.list_dir = [el for el in os.listdir(img_dir) if el.endswith(".jpg") and not el.startswith(".")]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, item):
        if self.img_labels.iloc[item, 0] + ".jpg" in self.list_dir :
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[item, 0] + ".jpg")
            image = read_image(img_path)
            label = 1 if self.img_labels.iloc[item, 1]=='SHORT' else 0
            return image, label
        return None