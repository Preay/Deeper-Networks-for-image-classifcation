# data_loader.py
import torch
from torchvision import datasets, transforms
import random

def get_data_loaders(batch_size=64, noise_level=0.0):
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

    if noise_level > 0:
        inject_label_noise(train_dataset, noise_level=noise_level)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, test_loader


def inject_label_noise(dataset, noise_level=0.1, num_classes=10):
    """
    Randomly flips a percentage of labels in the dataset.
    """
    targets = torch.tensor(dataset.targets)
    num_noisy = int(noise_level * len(targets))
    noisy_indices = random.sample(range(len(targets)), num_noisy)

    for idx in noisy_indices:
        original_label = targets[idx].item()
        new_label = random.choice([i for i in range(num_classes) if i != original_label])
        targets[idx] = new_label

    dataset.targets = targets
    print(f"🔄 Injected label noise: {num_noisy} samples ({noise_level*100}%)")
