# 🧠 ProHealth TumorOI OS: Brain Tumor MRI Classification & Segmentation

An end-to-end, medical imaging deep learning pipeline built to classify brain tumor types and segment tumorous tissue from Magnetic Resonance Imaging (MRI) scans. The project features a premium, modern clinical dashboard interface built with Streamlit, backed by PyTorch models for classification and segmentation.

---

## 🚀 Key Features

- **Automated Workflow Pipeline**: Upload an MRI scan and the application automatically runs a multi-step analysis workflow:
  1. **Image Validation**: Checks if the upload is a valid grayscale brain MRI via hybrid guardrails.
  2. **Classification**: Evaluates whether a tumor is present and classifies it into one of four classes (`Glioma`, `Meningioma`, `Pituitary`, `No Tumor`).
  3. **Segmentation**: If a tumor is detected, identifies the exact spatial region and renders a pixel-level overlay.
  4. **Explainable AI (XAI)**: Generates Grad-CAM class activation maps showing model focal areas.
  5. **Standardized PDF Report Generation**: Instant export and download of a clinical diagnostic report with tri-view imaging, quantitative morphology, and patient demographics.
- **Interactive Segmentation Overlay**: Displays the tumor contour and probability overlay on top of the original MRI scan.
- **Robust Verification Check**: Includes an automated verification script to check dependencies, CUDA runtime compatibility, and GPU execution support.
- **Deep Learning Architectures**:
  - **Classifier**: Fine-tuned **EfficientNet-B0** classification model achieving **95.3% test accuracy**.
  - **Segmenter**: **U-Net** architecture with an **EfficientNet-B0 encoder** achieving a **81.4% Dice Score**.


---

## 📂 Project Structure

```directory
c:/TumorOI/
├── .streamlit/
│   └── config.toml               # Streamlit custom premium UI theme
├── dataset/                      # (User-provided) Brain MRI datasets
│   ├── preprocessed/
│   │   ├── Training/             # Class-wise folders for training
│   │   └── Testing/              # Class-wise folders for testing
│   └── Segmentation-dataset/
│       ├── images/               # Grayscale MRI scan images (512x512)
│       └── masks/                # Co-registered binary masks (0/255)
├── models/                       # Classifier checkpoints & metrics
│   ├── best_efficientnet_b0.pth  # Classifier PyTorch state dict
│   ├── metrics.json              # Detailed test evaluations
│   ├── confusion_matrix.png      # Classifier confusion matrix visual
│   └── training_history.png      # Training vs validation loss/accuracy
├── seg_models/                   # Segmentation checkpoints & metrics
│   ├── best_unet_effb0.pth       # U-Net PyTorch state dict
│   ├── seg_metrics.json          # Detailed validation & test IoU/Dice
│   ├── prediction_samples.png    # Example predicted masks vs ground truth
│   └── seg_training_history.png  # Dice & loss optimization history
├── app.py                        # Streamlit clinical dashboard application
├── train.py                      # Classifier training & evaluation script
├── seg_train.py                  # U-Net segmentation training script
├── verify.py                     # Python environment & CUDA verification script
├── requirements.txt              # Project package dependencies
└── README.md                     # Documentation
```

---

## 🛠️ Installation & Setup

### 1. Prerequisite Packages
Verify that you have Python 3.10+ installed.

### 2. Configure Virtual Environment (Recommended)
Set up a clean virtual environment to prevent package version conflicts:

```powershell
# Create environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1
```

### 3. Install PyTorch with CUDA support and Dependencies
Install all requirements using the pre-configured [requirements.txt](file:///c:/TumorOI/requirements.txt), which pulls PyTorch with **CUDA 12.1** runtime support for optimal hardware acceleration:

```powershell
pip install -r requirements.txt
```

*Note: For the segmentation network, the `segmentation-models-pytorch` and `albumentations` packages are also required. Install them with:*
```powershell
pip install segmentation-models-pytorch albumentations
```

### 4. Run the Environment Verification Check
Run [verify.py](file:///c:/TumorOI/verify.py) to ensure all core libraries are correctly installed and that PyTorch detects your GPU:

```powershell
python verify.py
```

---

## 📊 Dataset Configuration

Ensure your datasets are structured as follows:

### Classification Dataset
Place images under [dataset/preprocessed/](file:///c:/TumorOI/dataset/preprocessed/):
- **`Training/`** & **`Testing/`** folders, each containing subdirectories named:
  - `glioma`
  - `meningioma`
  - `notumor`
  - `pituitary`

### Segmentation Dataset
Place grayscale PNG files under [dataset/Segmentation-dataset/](file:///c:/TumorOI/dataset/Segmentation-dataset/):
- **`images/`**: Source MRI images.
- **`masks/`**: Ground-truth binary segmentation masks (0 for background, 255 for tumor area) corresponding precisely to the filename numbers of the source images.

---

## 🏋️ Training the Models

### 1. Classification Model Training
The [train.py](file:///c:/TumorOI/train.py) script fine-tunes a pre-trained **EfficientNet-B0** network using mixed-precision training (`torch.amp.autocast`) to speed up processing:

- **Features**: Data augmentation (Random Rotation, Flipping), train/validation splitting, dynamic loss scale optimization, and automated generation of confusion matrix and training logs.
- **Execution**:
  ```powershell
  python train.py
  ```
- **Outputs**:
  - Saved model weights to [models/best_efficientnet_b0.pth](file:///c:/TumorOI/models/best_efficientnet_b0.pth)
  - Logged performance metrics to [models/metrics.json](file:///c:/TumorOI/models/metrics.json)
  - Plots generated in [models/training_history.png](file:///c:/TumorOI/models/training_history.png) and [models/confusion_matrix.png](file:///c:/TumorOI/models/confusion_matrix.png)

### 2. Segmentation Model Training
The [seg_train.py](file:///c:/TumorOI/seg_train.py) script trains a **U-Net** architecture using a pretrained **EfficientNet-B0 encoder**:

- **Features**: Heavy pixel-level augmentations using `albumentations` (Elastic Transform, Grid Distortion, Gauss Noise), custom Dice Loss optimization, and evaluation of Dice Coefficients and Mean Intersection over Union (mIoU).
- **Execution**:
  ```powershell
  python seg_train.py
  ```
- **Outputs**:
  - Saved model weights to [seg_models/best_unet_effb0.pth](file:///c:/TumorOI/seg_models/best_unet_effb0.pth)
  - Logged performance metrics to [seg_models/seg_metrics.json](file:///c:/TumorOI/seg_models/seg_metrics.json)
  - Diagnostic test visuals to [seg_models/prediction_samples.png](file:///c:/TumorOI/seg_models/prediction_samples.png)

---

## 🖥️ Running the Streamlit App

Launch the interactive clinical dashboard:

```powershell
streamlit run app.py
```

### Dashboard Workflow:
1. **Clinical File Uploader**: Upload any Brain MRI (JPG, PNG, JPEG).
2. **Quality Heuristics Validation**: The system checks image characteristics (color saturation, average brightness, black/dark background ratio) to ensure it resembles a grayscale MRI scan and flags any invalid format.
3. **Multi-Class Classifier Execution**: Instantly provides class likelihoods for Glioma, Meningioma, Pituitary, or No Tumor.
4. **Interactive Tumor Segmenter**: If a tumor class is predicted, the application passes the image to the U-Net model, displays a pixel-level overlay, highlights the exact spatial region, and lists critical metrics like the Dice Coefficient and mIoU.

---

## 📈 Model Performance Metrics

### Classifier Evaluation
*Evaluated on the testing set (1,600 total scans; 400 per class)*

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Glioma** | 99.4% | 82.5% | 90.2% | 400 |
| **Meningioma** | 88.8% | 99.3% | 93.7% | 400 |
| **No Tumor** | 95.0% | 100.0% | 97.4% | 400 |
| **Pituitary** | 99.5% | 99.5% | 99.5% | 400 |
| **Accuracy** | - | - | **95.3%** | 1,600 |
| **Macro Average** | **95.7%** | **95.3%** | **95.2%** | 1,600 |

- **Total Training Time**: ~8.7 minutes (10 epochs on GPU)

---

### Segmenter Evaluation
*Evaluated on the holdout test set (Dice & IoU coefficients)*

- **Test Dice Score**: **81.39%**
- **Test Mean IoU**: **72.67%**
- **Validation Dice Score (Best)**: **79.74%**
- **Total Training Time**: ~25.6 minutes (25 epochs on GPU)

---

## 🛡️ Medical Disclaimer
This software is intended for research, educational, and developer demonstration purposes only. It is not approved for clinical diagnostics or medical decision-making. Always consult a certified medical professional for health concerns.
