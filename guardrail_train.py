import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms, models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm

# ── Custom Dataset for Guardrail ──
class GuardrailDataset(Dataset):
    def __init__(self, mri_dir, non_mri_dir, transform=None):
        self.transform = transform
        self.samples = []
        self.classes = ['non_mri', 'mri']
        
        # Load Non-MRI (Class 0)
        if os.path.exists(non_mri_dir):
            for fname in os.listdir(non_mri_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                    self.samples.append((os.path.join(non_mri_dir, fname), 0))
        else:
            print(f"Warning: Non-MRI directory {non_mri_dir} not found.")

        # Load MRI (Class 1)
        if os.path.exists(mri_dir):
            for fname in os.listdir(mri_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                    self.samples.append((os.path.join(mri_dir, fname), 1))
        else:
            print(f"Warning: MRI directory {mri_dir} not found.")

        print(f"Loaded {len(self.samples)} samples. Non-MRI (0): {sum(1 for _, l in self.samples if l == 0)}, MRI (1): {sum(1 for _, l in self.samples if l == 1)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            # EfficientNet-B0 expects 3 channels
            img = Image.open(path).convert("RGB")
        except Exception as e:
            # Fallback to black image in case of load failure
            img = Image.new("RGB", (224, 224), color=0)
            print(f"Error loading {path}: {e}")
        
        if self.transform:
            img = self.transform(img)
        return img, label

# ── TransformedSubset Wrapper for multiprocessing pickling ──
class TransformedSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.subset)

# ── Configuration ────────────────────────────────────────────────────────────
MRI_DIR = r"C:\TumorOI\dataset\guardrail_dataset\mri_images"
NON_MRI_DIR = r"C:\TumorOI\dataset\guardrail_dataset\random_images_gray"
MODELS_DIR = r"C:\TumorOI\models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Augmentations for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        normalize
    ])

    # Basic normalization for validation/test
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    full_dataset = GuardrailDataset(mri_dir=MRI_DIR, non_mri_dir=NON_MRI_DIR)
    
    total_size = len(full_dataset)
    if total_size == 0:
        raise ValueError("Dataset is empty. Verify that the image folders exist and contain images.")

    # Split: 80% Train, 10% Val, 10% Test
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Wrap subsets with correct transformations
    train_dataset = TransformedSubset(train_subset, train_transform)
    val_dataset = TransformedSubset(val_subset, val_test_transform)
    test_dataset = TransformedSubset(test_subset, val_test_transform)

    # Create Dataloaders (num_workers=0 to avoid Windows multiprocessing issues)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Dataset split summary:")
    print(f"  Training samples   : {len(train_dataset)}")
    print(f"  Validation samples : {len(val_dataset)}")
    print(f"  Testing samples    : {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, full_dataset.classes

def build_model():
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    
    # Replace classifier head for 2 classes (non_mri, mri)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 2)
    )
    return model.to(DEVICE)

def train_epoch(model, dataloader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="Training Batch", leave=False):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        # Mixed precision training
        with torch.amp.autocast(device_type=DEVICE.type):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        with torch.amp.autocast(device_type=DEVICE.type):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc

@torch.no_grad()
def evaluate_model(model, dataloader, classes):
    model.eval()
    all_preds = []
    all_labels = []

    for inputs, labels in dataloader:
        inputs = inputs.to(DEVICE)
        with torch.amp.autocast(device_type=DEVICE.type):
            outputs = model(inputs)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    report_text = classification_report(all_labels, all_preds, target_names=classes)
    print("\n" + "="*60 + "\nGuardrail Test Set Performance Report:\n" + "="*60)
    print(report_text)

    # Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Guardrail Classifier - Confusion Matrix')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    
    cm_path = os.path.join(MODELS_DIR, "guardrail_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")
    
    return report

def plot_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'bo-', label='Training Acc')
    ax2.plot(epochs, history['val_acc'], 'ro-', label='Validation Acc')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    history_plot_path = os.path.join(MODELS_DIR, "guardrail_training_history.png")
    plt.savefig(history_plot_path)
    plt.close()
    print(f"Training history plot saved to {history_plot_path}")

def main():
    print(f"Using device: {DEVICE}")
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    
    print("\nBuilding model...")
    model = build_model()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler()

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0.0
    best_model_path = os.path.join(MODELS_DIR, "best_guardrail.pth")

    print("\nStarting Training...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 20)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")

        # Save best model
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best validation accuracy: {val_acc*100:.2f}%. Model saved.")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes.")
    
    # Plot history
    plot_history(history)
    
    # Load best weights and evaluate on test set
    print("\nLoading best model weights for test set evaluation...")
    model.load_state_dict(torch.load(best_model_path))
    test_metrics = evaluate_model(model, test_loader, classes)
    
    # Save metrics JSON
    metrics_summary = {
        "training_history": history,
        "test_performance": test_metrics,
        "total_training_time_minutes": total_time / 60
    }
    
    metrics_path = os.path.join(MODELS_DIR, "guardrail_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Metrics JSON saved to {metrics_path}")

if __name__ == "__main__":
    main()
