import os
import PIL.ImageOps
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def solarize(img):
    return PIL.ImageOps.solarize(img, 128)

def get_imagenet100_loaders(data_dir, batch_size=64, img_size=224, num_workers=4):
    """
    Creates DataLoaders for ImageNet-100 following the wilyzh structure:
    data_dir/
      train/
        nxxxx/
      val/
        nxxxx/
    """
    # Standard SSL/JEPA Augmentations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)), # 256 for 224
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load datasets using ImageFolder
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        # Fallback for wilyzh specific nested structure: ImageNet100/train/
        train_dir = os.path.join(data_dir, 'ImageNet100', 'train')
        val_dir = os.path.join(data_dir, 'ImageNet100', 'val')

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    # Create loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader

def get_imagenette_loaders(data_dir, batch_size=256, img_size=128, num_workers=8):
    """Standard ImageNette loaders for KerJEPA reproduction."""
    # SSL Pre-training Transform
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.RandomApply([transforms.Lambda(solarize)], p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Linear Probe / Evaluation Transform
    val_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader

if __name__ == "__main__":
    # Test ImageNette structure
    DATA_PATH = '/kaggle/input/datasets/aniladepu/imagenette/imagenette'
    if os.path.exists(DATA_PATH):
        train, val = get_imagenette_loaders(DATA_PATH)
        print(f"ImageNette - Train batches: {len(train)}, Val: {len(val)}")
    else:
        print(f"Path {DATA_PATH} not found.")
