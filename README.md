# Deep CNNs for Image Classification - MNIST Dataset

## Overview
This project trains two deep convolutional neural network models — **VGG16 with Dropout** and **ResNet18** — on the **MNIST** dataset for digit classification.

The goal is to:
- Compare deep architectures (VGG vs ResNet)
- Analyze training dynamics (loss, accuracy)
- Introduce small innovations (Dropout, Adam optimizer)
- Save clean training logs and visualization plots

---

## Project Structure
```plaintext
deepnet_classification_project/
├── data_loader.py         # Load MNIST dataset
├── models/
│   ├── vgg_custom.py      # Modified VGG16 with Dropout
│   └── resnet_custom.py   # Customized ResNet18
├── train.py               # Training script with logging
├── utils.py               # Plotting functions (loss/accuracy)
├── main.py                # Main file to run experiments
├── logs/                  # Training logs (per model)
├── results/               # Loss and accuracy plots
├── README.md              # This file
└── report/                # (for the 6-page final report)


---

## Requirements
To run this project, you need:

- Python 3.7+
- PyTorch
- torchvision
- matplotlib
- tqdm
- (optional) tensorboard

Install the required libraries using pip:
```bash
pip install torch torchvision matplotlib tqdm
