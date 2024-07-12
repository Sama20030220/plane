import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_data(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(root=data_dir, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    return train_loader, len(train_data.classes)
