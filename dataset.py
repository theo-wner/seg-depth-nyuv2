import os
import cv2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import torchvision.transforms.functional as TF
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from plot_utils import visualize_img_depth, visualize_img_label, visualize_img_label_depth
import config


'''
Defines classes for the NYUv2 dataset
'''

class NYUv2DataModule(pl.LightningDataModule):
    """
    Represents the NYUv2 DataModule needed for further simplification
    """
    def __init__(self, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    # Loads the dataset (not needed data already downloaded)
    def prepare_data(self):
        self.train_dataset = NYUv2Dataset(split='train')
        self.val_dataset = NYUv2Dataset(split='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=False)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=False)


class NYUv2Dataset(Dataset):
    """
    Represents the NYUv2 Dataset
    Example for obtaining an image: image, depth = dataset[0]
    """

    # Split can be either 'train' or 'test'
    def __init__(self, split='train'):
        self.root_dir = './data'
        self.split = split
        self.images_dir = os.path.join(self.root_dir, 'image', self.split)
        self.labels_dir = os.path.join(self.root_dir, 'seg40', self.split)
        self.depths_dir = os.path.join(self.root_dir, 'depth', self.split)
        self.filenames = os.listdir(self.images_dir)

    def __len__(self):
        return len(self.filenames)
    
    def _load_image(self, index):
        image_filename = os.path.join(self.images_dir, self.filenames[index])
        image = np.array(Image.open(image_filename))
        return image
    
    def _load_label(self, index):
        label_filename = os.path.join(self.labels_dir, self.filenames[index])
        label = np.array(Image.open(label_filename))
        return label
    
    def _load_depth(self, index):
        depth_filename = os.path.join(self.depths_dir, self.filenames[index])
        depth = np.array(Image.open(depth_filename))
        depth = np.divide(depth, 1000.) # Convert to meters
        return depth
    
    def get_training_augmentation(self):
        train_augmentation = A.Compose([
            A.RandomScale(scale_limit=(-0.5, +0.75), p=1), # Relates to Scaling between 0.5 and 1.75
            A.PadIfNeeded(min_height=480, min_width=640, always_apply=True, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0), mask_value=config.IGNORE_INDEX), # If the image gets smaller than 480x640    
            A.RandomCrop(height=480, width=640, p=1),
            A.HorizontalFlip(p=0.5),
        ])#,
        #additional_targets={'label' : 'image', 'depth' : 'image'})
        return train_augmentation

    def __getitem__(self, index):
        image = self._load_image(index)
        label = self._load_label(index)
        depth = self._load_depth(index)

        # In case of training, apply data augmentation (and ToTensor)
        if self.split == 'train':
            train_augmentation = self.get_training_augmentation()
            transformed = train_augmentation(image=image, masks=[label, depth])
            image, label, depth = transformed['image'], transformed['masks'][0], transformed['masks'][1]
            image = TF.to_tensor(image)
            label = torch.tensor(label, dtype=torch.long).unsqueeze(0)
            depth = torch.tensor(depth, dtype=torch.float).unsqueeze(0)

        # In case of validation, only apply ToTensor
        elif self.split == 'test':
            image = TF.to_tensor(image)
            label = torch.tensor(label, dtype=torch.long).unsqueeze(0)
            depth = torch.tensor(depth, dtype=torch.float).unsqueeze(0)
            
        return image, label, depth
    

if __name__ == '__main__':
    # Test the dataset
    dataset = NYUv2Dataset(split='train')

    for i in range(len(dataset)):
        image, label, depth = dataset[i]
        visualize_img_label_depth(image, label, label, depth, depth, filename='test' + str(i) + '.png')
        if i == 50:
            break



    