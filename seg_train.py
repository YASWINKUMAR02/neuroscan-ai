"""
Brain Tumor MRI Segmentation
Architecture : U-Net with EfficientNet-B0 encoder (segmentation_models_pytorch)
Dataset      : Segmentation-dataset/images + masks  (grayscale 512x512, binary masks 0/255)
Metric       : Dice Score + IoU
"""

import os, json, time, random
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm import tqdm

# -- Reproducibility ----------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -- Paths --------------------------------------------------------------------
IMG_DIR   = r"C:\TumorOI\dataset\Segmentation-dataset\images"
MASK_DIR  = r"C:\TumorOI\dataset\Segmentation-dataset\masks"
SAVE_DIR  = r"C:\TumorOI\seg_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# -- Hyperparameters ----------------------------------------------------------
IMG_SIZE     = 256          # resize to 256x256 to fit GPU memory
BATCH_SIZE   = 16
EPOCHS       = 25
LR           = 1e-4
VAL_SPLIT    = 0.15         # 15% validation
TEST_SPLIT   = 0.10         # 10% test
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS  = 2
# -----------------------------------------------------------------------------

print(f"Using device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU          : {torch.cuda.get_device_name(0)}")


# =============================================================================
# 1.  Dataset
# =============================================================================

class TumorSegDataset(Dataset):
    """
    Loads paired (image, mask) PNG files.
    Images are grayscale MRI  -> replicated to 3-channel for EfficientNet.
    Masks are binary {0, 255} -> normalised to {0, 1} float.
    """
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image (grayscale -> 3-channel RGB for EfficientNet)
        img  = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        img  = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)   # (H,W,3) uint8

        # Load mask: {0,255} -> {0,1}
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)          # (H,W) float32

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img  = augmented["image"]   # tensor (3,H,W) float
            mask = augmented["mask"]    # tensor (H,W)   float

        # Add channel dim to mask -> (1,H,W)
        mask = mask.unsqueeze(0)
        return img, mask


def get_transforms(split: str):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                               rotate_limit=15, p=0.4),
            A.ElasticTransform(alpha=120, sigma=6, p=0.3),
            A.GridDistortion(p=0.2),
            A.RandomBrightnessContrast(p=0.4),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:   # val / test
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


def build_dataloaders():
    # Collect all paired file paths (sorted by filename number)
    img_files  = sorted(os.listdir(IMG_DIR),
                        key=lambda x: int(os.path.splitext(x)[0]))
    mask_files = sorted(os.listdir(MASK_DIR),
                        key=lambda x: int(os.path.splitext(x)[0]))

    img_paths  = [os.path.join(IMG_DIR,  f) for f in img_files]
    mask_paths = [os.path.join(MASK_DIR, f) for f in mask_files]

    total = len(img_paths)
    assert total == len(mask_paths), "Image/mask count mismatch!"
    print(f"\nTotal samples : {total}")

    # Split indices
    indices   = list(range(total))
    random.shuffle(indices)

    n_test  = int(total * TEST_SPLIT)
    n_val   = int(total * VAL_SPLIT)
    n_train = total - n_val - n_test

    train_idx = indices[:n_train]
    val_idx   = indices[n_train: n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    print(f"  Train        : {len(train_idx)}")
    print(f"  Validation   : {len(val_idx)}")
    print(f"  Test         : {len(test_idx)}")

    def make_dataset(idx_list, split):
        imgs  = [img_paths[i]  for i in idx_list]
        masks = [mask_paths[i] for i in idx_list]
        return TumorSegDataset(imgs, masks, transform=get_transforms(split))

    train_ds = make_dataset(train_idx, "train")
    val_ds   = make_dataset(val_idx,   "val")
    test_ds  = make_dataset(test_idx,  "test")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, test_loader


# =============================================================================
# 2.  Model
# =============================================================================

def build_model():
    model = smp.Unet(
        encoder_name    = "efficientnet-b0",
        encoder_weights = "imagenet",       # ImageNet pretrained
        in_channels     = 3,
        classes         = 1,
        activation      = None,             # raw logits (BCEWithLogitsLoss)
    )
    return model.to(DEVICE)


# =============================================================================
# 3.  Loss & Metrics
# =============================================================================

class CombinedLoss(nn.Module):
    """BCE + Dice loss -- handles both structure and overlap."""
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce    = nn.BCEWithLogitsLoss()
        self.dice   = DiceLoss(mode="binary", from_logits=True)
        self.w_bce  = bce_weight

    def forward(self, pred, target):
        return self.w_bce * self.bce(pred, target) + \
               (1 - self.w_bce) * self.dice(pred, target)


def dice_score(pred_logits, target, threshold=0.5):
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    smooth = 1e-6
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union        = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


def iou_score(pred_logits, target, threshold=0.5):
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    smooth = 1e-6
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union        = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) \
                   - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


# =============================================================================
# 4.  Train / Val loops
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, total_dice, total_iou = 0, 0, 0

    for imgs, masks in tqdm(loader, desc="  Train", leave=False):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=DEVICE.type):
            preds = model(imgs)
            loss  = criterion(preds, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_dice += dice_score(preds.detach(), masks)
        total_iou  += iou_score(preds.detach(), masks)

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss, total_dice, total_iou = 0, 0, 0

    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        with torch.amp.autocast(device_type=DEVICE.type):
            preds = model(imgs)
            loss  = criterion(preds, masks)

        total_loss += loss.item()
        total_dice += dice_score(preds, masks)
        total_iou  += iou_score(preds, masks)

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


# =============================================================================
# 5.  Visualise predictions
# =============================================================================

@torch.no_grad()
def save_prediction_grid(model, loader, save_path, n_samples=8):
    model.eval()
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    imgs_list, masks_list, preds_list = [], [], []
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        with torch.amp.autocast(device_type=DEVICE.type):
            preds = torch.sigmoid(model(imgs))
        imgs_list.append(imgs.cpu()); masks_list.append(masks.cpu())
        preds_list.append(preds.cpu())
        if sum(len(x) for x in imgs_list) >= n_samples:
            break

    imgs_t  = torch.cat(imgs_list)[:n_samples]
    masks_t = torch.cat(masks_list)[:n_samples]
    preds_t = torch.cat(preds_list)[:n_samples]

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, n_samples * 4))
    fig.suptitle("U-Net + EfficientNet-B0 -- Predictions\n"
                 "Left: MRI Image | Middle: Ground Truth | Right: Prediction",
                 fontsize=14, fontweight='bold')

    for i in range(n_samples):
        # Denormalize image
        img = imgs_t[i].permute(1, 2, 0).numpy()
        img = np.clip(img * std + mean, 0, 1)
        img_gray = img.mean(axis=2)

        gt   = masks_t[i, 0].numpy()
        pred = (preds_t[i, 0].numpy() > 0.5).astype(np.float32)

        axes[i, 0].imshow(img_gray, cmap='gray')
        axes[i, 0].set_title("MRI Image"); axes[i, 0].axis('off')

        axes[i, 1].imshow(gt, cmap='Reds', vmin=0, vmax=1)
        axes[i, 1].set_title("Ground Truth"); axes[i, 1].axis('off')

        axes[i, 2].imshow(pred, cmap='Reds', vmin=0, vmax=1)
        axes[i, 2].set_title("Prediction"); axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Prediction grid saved -> {save_path}")


# =============================================================================
# 6.  Plot training curves
# =============================================================================

def plot_history(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train')
    axes[0].plot(epochs, history['val_loss'],   'r-o', label='Val')
    axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch')
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs, history['train_dice'], 'b-o', label='Train')
    axes[1].plot(epochs, history['val_dice'],   'r-o', label='Val')
    axes[1].set_title('Dice Score'); axes[1].set_xlabel('Epoch')
    axes[1].legend(); axes[1].grid(True)

    axes[2].plot(epochs, history['train_iou'], 'b-o', label='Train')
    axes[2].plot(epochs, history['val_iou'],   'r-o', label='Val')
    axes[2].set_title('IoU Score'); axes[2].set_xlabel('Epoch')
    axes[2].legend(); axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Training curves saved -> {save_path}")


# =============================================================================
# 7.  Main
# =============================================================================

def main():
    # -- Data --
    train_loader, val_loader, test_loader = build_dataloaders()

    # -- Model --
    print("\nBuilding U-Net + EfficientNet-B0 ...")
    model     = build_model()
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params/1e6:.2f}M")

    # -- Training setup --
    criterion = CombinedLoss(bce_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler    = torch.amp.GradScaler()

    best_val_dice   = 0.0
    best_model_path = os.path.join(SAVE_DIR, "best_unet_effb0.pth")

    history = {k: [] for k in
               ['train_loss','train_dice','train_iou',
                'val_loss',  'val_dice',  'val_iou']}

    print(f"\nStarting training for {EPOCHS} epochs ...")
    print("=" * 55)
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}  (lr={scheduler.get_last_lr()[0]:.2e})")
        print("-" * 40)

        tr_loss, tr_dice, tr_iou = train_epoch(
            model, train_loader, criterion, optimizer, scaler)
        vl_loss, vl_dice, vl_iou = validate(
            model, val_loader, criterion)

        scheduler.step()

        for key, val in zip(
            ['train_loss','train_dice','train_iou',
             'val_loss',  'val_dice',  'val_iou'],
            [tr_loss, tr_dice, tr_iou,
             vl_loss, vl_dice, vl_iou]):
            history[key].append(val)

        print(f"  Train -> Loss: {tr_loss:.4f}  Dice: {tr_dice:.4f}  IoU: {tr_iou:.4f}")
        print(f"  Val   -> Loss: {vl_loss:.4f}  Dice: {vl_dice:.4f}  IoU: {vl_iou:.4f}")

        if vl_dice > best_val_dice:
            best_val_dice = vl_dice
            torch.save(model.state_dict(), best_model_path)
            print(f"  Best val Dice: {vl_dice:.4f} -- model saved.")

    total_time = time.time() - start_time
    print(f"\n{'='*55}")
    print(f"Training complete in {total_time/60:.2f} min.")
    print(f"Best validation Dice: {best_val_dice:.4f}")

    # -- Plots --
    plot_history(history,
                 os.path.join(SAVE_DIR, "seg_training_history.png"))

    # -- Test evaluation --
    print("\nLoading best weights for test evaluation ...")
    model.load_state_dict(torch.load(best_model_path))
    te_loss, te_dice, te_iou = validate(model, test_loader, criterion)
    print(f"\n{'='*55}")
    print(f"  TEST  -> Loss: {te_loss:.4f}  Dice: {te_dice:.4f}  IoU: {te_iou:.4f}")
    print(f"{'='*55}")

    # -- Prediction visualisation --
    save_prediction_grid(
        model, test_loader,
        os.path.join(SAVE_DIR, "prediction_samples.png"))

    # -- Save metrics JSON --
    metrics = {
        "training_history": history,
        "test_performance": {
            "loss": te_loss,
            "dice": te_dice,
            "iou":  te_iou,
        },
        "best_val_dice": best_val_dice,
        "total_training_time_minutes": total_time / 60,
        "config": {
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "encoder": "efficientnet-b0",
            "architecture": "U-Net",
        }
    }
    metrics_path = os.path.join(SAVE_DIR, "seg_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\n  Metrics saved -> {metrics_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
