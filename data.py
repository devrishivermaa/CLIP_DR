"""Data loading and preprocessing for APTOS dataset."""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import config


class AptosMapper(Dataset):
    """APTOS Diabetic Retinopathy dataset."""
    
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the csv file with annotations.
            img_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_name = self.labels.iloc[idx, 0] + ".png"
        label = int(self.labels.iloc[idx, 1])
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms():
    """Get train and test transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.CLIP_MEAN, std=config.CLIP_STD),
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.CLIP_MEAN, std=config.CLIP_STD),
    ])
    
    return {"train": train_transform, "test": test_transform}


def get_dataloaders(batch_size=None, num_workers=None):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if num_workers is None:
        num_workers = config.NUM_WORKERS
    
    transforms_dict = get_transforms()
    
    train_dataset = AptosMapper(
        config.TRAIN_CSV_PATH,
        config.TRAIN_IMG_DIR,
        transforms=transforms_dict["train"]
    )
    
    val_dataset = AptosMapper(
        config.VAL_CSV_PATH,
        config.VAL_IMG_DIR,
        transforms=transforms_dict["test"]
    )
    
    test_dataset = AptosMapper(
        config.TEST_CSV_PATH,
        config.TEST_IMG_DIR,
        transforms=transforms_dict["test"]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader