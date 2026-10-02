import torch
from torch.utils.data import DataLoader
from src.data import SkinLesionDataset, get_transforms
from src.model import SkinCancerModel
import argparse

def evaluate(csv_file, image_dirs, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data - use val transforms (no augmentation)
    transform = get_transforms(phase='val')
    
    dataset = SkinLesionDataset(root_dirs=image_dirs, csv_file=csv_file, transform=transform, mode='val')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = SkinCancerModel(num_classes=7).to(device)
    try:
        model.load_state_dict(torch.load("skin_cancer_model.pth", map_location=device))
        print("Loaded trained model.")
    except:
        print("Could not load model file.")
        return

    model.eval()
    
    correct_predictions = 0
    total_samples = 0
    
    print("Evaluating...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Metrics
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Get class names from dataset
    class_names = dataset.classes
    
    print("\n" + "="*30)
    print("CLASSIFICATION REPORT")
    print("="*30)
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    
    print("\n" + "="*30)
    print("CONFUSION MATRIX")
    print("="*30)
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    # Save Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("\nSaved confusion matrix to: confusion_matrix.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--image_dirs", nargs='+', required=True)
    args = parser.parse_args()
    
    evaluate(csv_file=args.csv, image_dirs=args.image_dirs)
