# utils.py
import matplotlib.pyplot as plt
import os
import torch

def load_model(model, model_path):
    """
    Loads a model's weights from a given path.
    Args:
        model: The model architecture (should match the saved weights).
        model_path: Path to the saved .pth file
    """
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode
    print(f"✅ Loaded model from {model_path}")
    return model

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, model_name="model", results_dir="./results/"):
    os.makedirs(results_dir, exist_ok=True)

    # Plot Loss
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'{model_name} Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    loss_path = os.path.join(results_dir, f"{model_name}_loss_curve.png")
    plt.savefig(loss_path)
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(10,5))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    acc_path = os.path.join(results_dir, f"{model_name}_accuracy_curve.png")
    plt.savefig(acc_path)
    plt.close()

    print(f"Saved loss and accuracy curves for {model_name} in {results_dir}")

