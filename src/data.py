import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_transforms(image_size=224, train=True):
    if train:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),  # Slightly more rotation
            transforms.RandomAffine(0, translate=(0.05, 0.05)),  # Small translations
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
    return transform

def make_dataloaders(data_root, batch_size=32, image_size=224, num_workers=4):
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'validation')
    test_dir = os.path.join(data_root, 'test')

    train_ds = datasets.ImageFolder(train_dir, transform=get_transforms(image_size, train=True))
    val_ds = datasets.ImageFolder(val_dir, transform=get_transforms(image_size, train=False))
    test_ds = datasets.ImageFolder(test_dir, transform=get_transforms(image_size, train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = train_ds.classes
    return train_loader, val_loader, test_loader, class_names
