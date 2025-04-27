# main.py
import torch
from data_loader import get_data_loaders
from models.vgg_custom import VGG16Custom
from models.resnet_custom import ResNet18Custom
from train import train_model
from utils import plot_training_curves
import os

def main():
    # Set hyperparameters
    batch_size = 64
    num_epochs = 20
    learning_rate = 0.001

    # Prepare directories
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./results", exist_ok=True)

    # Load data
    train_loader, val_loader = get_data_loaders(batch_size=batch_size)

    ##############################
    # Train VGG16 Model
    ##############################
    print("\n===== Training VGG16 Model =====\n")
    vgg_model = VGG16Custom(num_classes=10)
    vgg_train_losses, vgg_val_losses, vgg_train_accuracies, vgg_val_accuracies = train_model(
        model=vgg_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=num_epochs,
        lr=learning_rate,
        model_name="VGG16",
        log_dir="./logs"
    )

    # Plot and save curves
    plot_training_curves(
        vgg_train_losses,
        vgg_val_losses,
        vgg_train_accuracies,
        vgg_val_accuracies,
        model_name="VGG16",
        results_dir="./results"
    )

    ##############################
    # Train ResNet18 Model
    ##############################
    print("\n===== Training ResNet18 Model =====\n")
    resnet_model = ResNet18Custom(num_classes=10)
    resnet_train_losses, resnet_val_losses, resnet_train_accuracies, resnet_val_accuracies = train_model(
        model=resnet_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=num_epochs,
        lr=learning_rate,
        model_name="ResNet18",
        log_dir="./logs"
    )

    # Plot and save curves
    plot_training_curves(
        resnet_train_losses,
        resnet_val_losses,
        resnet_train_accuracies,
        resnet_val_accuracies,
        model_name="ResNet18",
        results_dir="./results"
    )

    print("\n===== Training Completed! =====\n")
    print("Graphs and logs saved inside 'results/' and 'logs/' folders.")

if __name__ == "__main__":
    main()
