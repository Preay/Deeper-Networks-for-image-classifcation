# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from datetime import datetime

def train_model(model, train_loader, val_loader, epochs=20, lr=0.001, model_name="model", log_dir="./logs/"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Create log directory if not exists
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{model_name}_train_log.txt")
    
    # Start logging
    with open(log_file, "w") as f:
        f.write(f"Training started at {datetime.now()}\n")
        f.write(f"Model: {model_name}\n\n")

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total

        # Print and Save Logs
        log_line = f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%"
        print(log_line)
        with open(log_file, "a") as f:
            f.write(log_line + "\n")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

    with open(log_file, "a") as f:
        f.write(f"\nTraining finished at {datetime.now()}\n")


    # Save model
    os.makedirs(log_dir, exist_ok=True)
    model_save_path = os.path.join(log_dir, f"{model_name}_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"\n✅ Saved {model_name} model to {model_save_path}")
    
    return train_losses, val_losses, train_accuracies, val_accuracies
