# main.py
import torch
from data_loader import get_data_loaders
from models.vgg_small import MiniVGG
from models.resnet_custom import ResNet18Custom
from train import train_model
from utils import plot_training_curves
import os

def model_exists(model_name):
    return os.path.exists(f"./logs/{model_name}_model.pth")

def main():
    # Hyperparameters
    batch_size = 128
    num_epochs = 5
    learning_rate = 0.001
    noise_level = 0.1  # ✨ Set noise level: 0.0 for clean, 0.1 = 10% noisy labels

    # Prepare directories
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./results", exist_ok=True)

    # Load data with noise
    train_loader, val_loader = get_data_loaders(batch_size=batch_size, noise_level=noise_level)

    ##############################
    # Train VGG16 Model
    ##############################
    model_name = f"VGG16_noise{int(noise_level * 100)}"
    if not model_exists(model_name):
        print(f"\n🚀 Training {model_name}...\n")
        vgg_model = MiniVGG(num_classes=10)
        vgg_train_losses, vgg_val_losses, vgg_train_accuracies, vgg_val_accuracies = train_model(
            model=vgg_model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=num_epochs,
            lr=learning_rate,
            model_name=model_name,
            log_dir="./logs"
        )
        plot_training_curves(
            vgg_train_losses,
            vgg_val_losses,
            vgg_train_accuracies,
            vgg_val_accuracies,
            model_name=model_name,
            results_dir="./results"
        )
    else:
        print(f"✅ Skipping {model_name} — already trained and saved.\n")

    ##############################
    # Train ResNet18 Model
    ##############################
    model_name = f"ResNet18_noise{int(noise_level * 100)}"
    if not model_exists(model_name):
        print(f"\n🚀 Training {model_name}...\n")
        resnet_model = ResNet18Custom(num_classes=10)
        resnet_train_losses, resnet_val_losses, resnet_train_accuracies, resnet_val_accuracies = train_model(
            model=resnet_model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=num_epochs,
            lr=learning_rate,
            model_name=model_name,
            log_dir="./logs"
        )
        plot_training_curves(
            resnet_train_losses,
            resnet_val_losses,
            resnet_train_accuracies,
            resnet_val_accuracies,
            model_name=model_name,
            results_dir="./results"
        )
    else:
        print(f"✅ Skipping {model_name} — already trained and saved.\n")

    print("\n🎉 All done! Training completed or skipped based on existing models.")
    print("📁 Check 'logs/' for saved models and logs, and 'results/' for loss/accuracy plots.\n")


if __name__ == "__main__":
    main()
