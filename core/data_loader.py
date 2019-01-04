"""
Now that we know what the input size must be, we can initialize
the data transforms, image datasets, and the dataloaders.
Notice, the models were pretrained with the hard-coded
normalization values, as described in
https://pytorch.org/docs/master/torchvision/models.html.
"""

from __future__ import print_function

# Data augmentation and normalization for training
# Just normalization for validation
import os

import torch
from torchvision import transforms, datasets


def transform_data(
        data_dir: str,
        input_size: int,
        batch_size: int
):
    img_folders = ['train', 'val']
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ])
    }

    print("Initializing Datasets and Dataloaders...")

    image_datasets = {
        folder: datasets.ImageFolder(
            os.path.join(data_dir, folder),
            data_transforms[folder]
        )
        for folder in img_folders
    }
    dataloaders_dict = {
        folder: torch.utils.data.DataLoader(
            image_datasets[folder],
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        for folder in img_folders
    }

    return dataloaders_dict
