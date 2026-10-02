import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data import SkinLesionDataset, get_transforms
from src.model import SkinCancerModel
import argparse

def train_model(epochs=5, batch_size=32, mock=False, csv_file=None, image_dirs=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    transform = get_transforms(phase='train')
    
    # Default image dir if None
    if image_dirs is None:
        image_dirs = ['data/train']

    print(f"Loading data from: {image_dirs}")
    print(f"Using metadata: {csv_file}")
    
    dataset = SkinLesionDataset(root_dirs=image_dirs, csv_file=csv_file, transform=transform, mode='train', mock=mock)
    
    if len(dataset) == 0:
        print("Error: Dataset is empty. Found 0 images. Check your paths and CSV.")
        return

    print(f"Found {len(dataset)} images.")
    
    # Calculate Class Weights for Imbalance Handling
    # Getting labels list might be expensive if very large, but fine for 10k
    if not mock:
        from collections import Counter
        import numpy as np
        
        counts = Counter(dataset.labels)
        print(f"Class Distribution: {counts}")
        
        # Mapping back to class names for info
        class_names = dataset.classes
        for idx, count in counts.items():
            print(f"  {class_names[idx]}: {count}")

        # Weights = Total / (NumClasses * Count)
        total_samples = len(dataset)
        num_classes = len(dataset.classes)
        weights = [total_samples / (num_classes * counts[i]) for i in range(num_classes)]
        class_weights = torch.FloatTensor(weights).to(device)
        print(f"Class Weights: {weights}")
    else:
        num_classes = 2 # Mock default
        class_weights = None

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    # Important: Update num_classes
    num_classes = len(dataset.classes) if not mock else 2
    model = SkinCancerModel(num_classes=num_classes).to(device)
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels)
            total_samples += labels.size(0)
            
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct_predictions.double() / total_samples
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        
    # Save model
    torch.save(model.state_dict(), "skin_cancer_model.pth")
    print("Model saved to skin_cancer_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Use mock data")
    parser.add_argument("--csv", type=str, help="Path to HAM10000_metadata.csv")
    parser.add_argument("--image_dirs", nargs='+', help="List of folders containing images")
    parser.add_argument("--epochs", type=int, default=1) 
    
    args = parser.parse_args()
    
    train_model(epochs=args.epochs, mock=args.mock, csv_file=args.csv, image_dirs=args.image_dirs)
