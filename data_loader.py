# data_loader.py
import torch
from torchvision import datasets, transforms

def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),       # Resize from 28x28 to 224x224
        transforms.Grayscale(num_output_channels=3),  # Duplicate channels to get 3 channels
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Still normalize as usual
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
