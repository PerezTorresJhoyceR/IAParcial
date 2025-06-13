#!/usr/bin/env python3
"""
Food-101 Dataset Classifier with MLP
===================================
This script trains an MLP model to classify food images from the Food-101 dataset.
It includes data loading, model definition, training, and evaluation.
The script compares different activation functions: ReLU, SiLU, and LeakyReLU.
"""

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Custom dataset class for Food-101
class Food101Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Root directory of the Food-101 dataset
            split: 'train' or 'test'
            transform: Optional transform to be applied on an image
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Load class names
        with open(os.path.join(root_dir, 'meta', 'classes.txt'), 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Create class to index mapping
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Load image paths and labels
        with open(os.path.join(root_dir, 'meta', f'{split}.txt'), 'r') as f:
            self.image_paths = [line.strip() for line in f.readlines()]
        
        self.labels = [self.class_to_idx[path.split('/')[0]] for path in self.image_paths]
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'images', f'{self.image_paths[idx]}.jpg')
        
        # Use PIL Image for better compatibility
        from PIL import Image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image in case of error
            image = Image.new('RGB', (224, 224), color='black')
            
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# MLP Model Architecture
class MLPClassifier(nn.Module):
    def __init__(self, input_size, num_classes=101, activation='relu'):
        super(MLPClassifier, self).__init__()
        self.flatten = nn.Flatten()
        
        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # MLP layers
        self.layers = nn.Sequential(
            nn.Linear(input_size, 2048),
            self.activation,
            nn.BatchNorm1d(2048),
            nn.Dropout(0.4),
            
            nn.Linear(2048, 1024),
            self.activation,
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            self.activation,
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x

# Function to train the model
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, device='cuda'):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    patience = 10  # Early stopping patience
    no_improve = 0
    
    # For tracking training progress
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data with tqdm progress bar
            loader = dataloaders[phase]
            for inputs, labels in tqdm(loader, desc=f"{phase} batch"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Update history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                # Update learning rate based on validation metrics
                scheduler.step(epoch_acc)
            
            # Save the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                torch.save(model.state_dict(), f'best_food101_model_{model_name}.pth')
                print(f'Model saved with accuracy: {best_acc:.4f}')
                no_improve = 0
            elif phase == 'val':
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1} as validation accuracy hasn't improved for {patience} epochs")
                    # Break out of the inner loop
                    # To break out completely, we'll check a flag
                    stop_early = True
                
        print()
        
        # Check if early stopping was triggered
        if 'stop_early' in locals() and stop_early:
            break
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation accuracy: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Save training history
    with open(f'training_history_{model_name}.json', 'w') as f:
        json.dump(history, f)
        
    return model, history

# Function to evaluate model
def evaluate_model(model, dataloader, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# Function to visualize predictions
def visualize_predictions(model, dataloader, class_names, device='cuda', num_images=8):
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(15, 10))
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 4, images_so_far)
                ax.axis('off')
                ax.set_title(f'Predicted: {class_names[preds[j]]}\nActual: {class_names[labels[j]]}')
                
                # Convert tensor to numpy for plotting
                img = inputs.cpu().data[j].permute(1, 2, 0).numpy()
                # Denormalize
                img = img * 0.5 + 0.5
                img = np.clip(img, 0, 1)
                ax.imshow(img)
                
                if images_so_far == num_images:
                    plt.savefig(f'prediction_examples_{model_name}.png')
                    return
    plt.tight_layout()
    plt.savefig(f'prediction_examples_{model_name}.png')

def train_and_evaluate(activation_function, input_size, num_classes, dataloaders, device):
    global model_name
    model_name = f"mlp_{activation_function}"
    
    print(f"\n{'='*50}")
    print(f"Training MLP with {activation_function} activation")
    print(f"{'='*50}\n")
    
    # Create model
    model = MLPClassifier(input_size=input_size, num_classes=num_classes, activation=activation_function)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, verbose=True)
    
    # Train model
    print("Starting model training...")
    model, history = train_model(
        model, 
        dataloaders, 
        criterion, 
        optimizer, 
        scheduler,
        num_epochs=25,
        device=device
    )
    
    # Evaluate model on test set
    print("Evaluating model on test set...")
    test_accuracy = evaluate_model(model, dataloaders['test'], device=device)
    
    # Visualize some predictions
    print("Visualizing predictions...")
    visualize_predictions(model, dataloaders['test'], dataloaders['train'].dataset.dataset.classes, device=device)
    
    return model, history, test_accuracy

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set paths
    data_dir = 'food-101'
    
    # Define transformations - No augmentation, just resize and normalize
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Downsample to make MLP training feasible
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Create datasets
    train_dataset = Food101Dataset(root_dir=data_dir, split='train', transform=transform)
    test_dataset = Food101Dataset(root_dir=data_dir, split='test', transform=transform)
    
    # Get a subset for faster experimentation - comment out for full training
    subset_size = min(10000, len(train_dataset))
    test_subset_size = min(2000, len(test_dataset))
    
    train_indices = torch.randperm(len(train_dataset))[:subset_size]
    test_indices = torch.randperm(len(test_dataset))[:test_subset_size]
    
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
    
    # Split training data into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders with reduced batch size for MLP
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    # Calculate input size for MLP
    input_size = 64 * 64 * 3  # 64x64 RGB images flattened
    num_classes = 101  # Food-101 has 101 food categories
    
    # Train and evaluate models with different activations
    activation_functions = ['relu', 'silu', 'leaky_relu']
    results = {}
    
    for activation in activation_functions:
        _, history, test_acc = train_and_evaluate(
            activation, 
            input_size, 
            num_classes, 
            dataloaders, 
            device
        )
        results[activation] = {
            'test_accuracy': test_acc,
            'history': history
        }
    
    # Compare results
    print("\nComparison of activation functions:")
    print("-" * 40)
    for activation in activation_functions:
        print(f"{activation.upper()}: Test accuracy = {results[activation]['test_accuracy']:.2f}%")
    
    # Plot comparative results
    plt.figure(figsize=(15, 10))
    
    # Plot training loss
    plt.subplot(2, 2, 1)
    for activation in activation_functions:
        plt.plot(results[activation]['history']['train_loss'], label=f'{activation}')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation loss
    plt.subplot(2, 2, 2)
    for activation in activation_functions:
        plt.plot(results[activation]['history']['val_loss'], label=f'{activation}')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot training accuracy
    plt.subplot(2, 2, 3)
    for activation in activation_functions:
        plt.plot(results[activation]['history']['train_acc'], label=f'{activation}')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(2, 2, 4)
    for activation in activation_functions:
        plt.plot(results[activation]['history']['val_acc'], label=f'{activation}')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('activation_function_comparison.png')
    print("Comparison plot saved as 'activation_function_comparison.png'")

if __name__ == "__main__":
    main() 