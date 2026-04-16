import os
import PIL.ImageOps
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def solarize(img):
    return PIL.ImageOps.solarize(img, 128)

class TwoViewFolder(datasets.ImageFolder):
    """
    Returns two augmented versions of the same image.
    Used for Joint-Embedding SSL (VICReg, SimCLR, JEPA, etc.).
    """
    def __init__(self, root, transform):
        super().__init__(root, None)
        self.aug = transform
        
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.loader(path)
        return self.aug(img), self.aug(img), label

def get_ssl_transform(img_size=128):
    """
    Standard high-quality augmentations for Self-Supervised Learning.
    Includes ColorJitter, Grayscale, Gaussian Blur, and Solarization.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.RandomApply([transforms.Lambda(solarize)], p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_eval_transform(img_size=128):
    """
    Simple CenterCrop transform for Linear Probe and Evaluation.
    """
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_imagenette_loaders(data_dir, batch_size=256, img_size=128, num_workers=4, is_ssl=False):
    """
    Standardized ImageNette loaders for KerJEPA.
    - is_ssl=True returns TwoViewFolder for pre-training.
    - is_ssl=False returns standard ImageFolder for linear probing.
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if is_ssl:
        train_ds = TwoViewFolder(train_dir, get_ssl_transform(img_size))
    else:
        train_ds = datasets.ImageFolder(train_dir, get_eval_transform(img_size))
        
    val_ds = datasets.ImageFolder(val_dir, get_eval_transform(img_size))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader

if __name__ == "__main__":
    # Example usage / test
    DATA_PATH = '/kaggle/input/datasets/aniladepu/imagenette/imagenette'
    if os.path.exists(DATA_PATH):
        train, val = get_imagenette_loaders(DATA_PATH, is_ssl=True)
        print(f"SSL Pre-training: {len(train)} batches")
    else:
        print(f"Data path {DATA_PATH} not found.")
