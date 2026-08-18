# NeuroScan AI: Enterprise Brain Tumor MRI Classification & Volumetric Suite

**NeuroScan AI (v2.1)** is an end-to-end clinical diagnostic workstation and volumetric neuroimaging intelligence platform. Built with PyTorch, Streamlit, and MySQL/SQLite, it couples state-of-the-art computer vision models with 3D Marching Cubes volumetric rendering, clinical Retrieval-Augmented Generation (RAG), salted **bcrypt** password cryptography, and a hospital-grade diagnostic interface.

---

## 📖 About The Project

### The Clinical Problem
Brain tumors (such as **Gliomas**, **Meningiomas**, and **Pituitary adenomas**) are among the most critical neuro-oncological conditions requiring rapid diagnosis, accurate classification, and precise spatial localization. Traditional radiological workflows often face significant challenges:
- **Inter-Observer Variability**: Manual delineation of irregular tumor boundaries on MRI slices is subjective and labor-intensive.
- **2D to 3D Cognitive Load**: Clinicians must mentally reconstruct 2D axial MRI slices into a 3D mental model to evaluate tumor volume, mass effect, and surgical approach angles.
- **Reporting Turnaround**: Generating standardized, quantifiable diagnostic documentation and patient summaries can take hours per case.

### The NeuroScan AI Solution
**NeuroScan AI** provides an integrated, AI-assisted radiological workstation that transforms raw Brain MRI scans into actionable, quantified clinical intelligence in seconds:

1. **Intelligent Ingestion & Medical Guardrails**: Validates that uploaded scans are genuine brain MRIs, rejecting non-medical or corrupted images before analysis.
2. **Deep Learning Classification (95.3% Accuracy)**: Fine-tuned **EfficientNet-B0** instantly detects tumor presence and classifies it into **Glioma**, **Meningioma**, **Pituitary**, or **No Tumor**, complete with **Grad-CAM** visual transparency.
3. **Pixel-Level Lesion Segmentation (81.4% Dice)**: A **U-Net** deep learning network delineates the precise anatomical tumor boundaries and calculates lesion surface area.
4. **3D Volumetric Mesh & Penetration Analysis**: Using the **Marching Cubes** isosurface algorithm, the system extracts interactive 3D surface meshes of the skull and tumor, computing tumor volume in cubic centimeters ($cm^3$) and measuring multi-slice anatomical penetration depth.
5. **Clinical RAG & Multimodal AI Co-Pilot**: An embedded Retrieval-Augmented Generation assistant answers clinician questions with grounded medical guidelines and multimodal scan reasoning.
6. **DICOM-Grade PDF Diagnostic Reports**: One-click generation of exportable, audit-ready clinical summary reports containing tri-view imaging, patient demographics, and quantitative metrics.
7. **Hospital-Grade Security & Multi-Tenant RBAC**: Salted **bcrypt** password hashing, 7-point password policy enforcement, and role-isolated portals for **Clinicians**, **Patients**, and **Administrators**.

---

## 🚀 Key Capabilities

### 1. Dual-Stage Deep Learning Pipeline
- **Hybrid Input Guardrails**: Evaluates grayscale saturation, brightness histograms, and tissue boundary ratios to prevent invalid or corrupted scan ingestion.
- **EfficientNet-B0 Classifier**: Fine-tuned classification network identifying **Glioma**, **Meningioma**, **Pituitary**, and **No Tumor** (**95.3% test accuracy**).
- **U-Net EfficientNet-B0 Segmenter**: Pixel-level tumor boundary delineation with **81.4% Dice Score** and **72.7% mIoU**.
- **Explainable AI (XAI)**: High-resolution Grad-CAM focal activation heatmaps for radiological transparency.

### 2. 3D Volumetric Morphology & Interactive Orbit Engine
- **Marching Cubes Isosurface Extraction**: Generates 3D spatial meshes of anatomical brain contours and co-registered tumor bodies from multi-slice axial MRI data.
- **Interactive WebGL Visualizer**: Real-time 3D orbit rotation with dynamic opacity layers, penetration depth metrics, and tumor volumetric measurement ($cm^3$).

### 3. Cryptographic Security & Password Policy
- **Salted bcrypt Hashing (`rounds=12`)**: Never stores plaintext passwords; protects against rainbow tables and GPU brute-force attacks. Multi-scheme verification with Argon2 and PBKDF2 fallbacks.
- **Transparent Auto-Upgrade Migration**: Automatically identifies legacy SHA-256 accounts upon sign-in and transparently upgrades them to salted bcrypt hashes in place.
- **7-Point Clinical Password Policy**:
  - Minimum 8 characters
  - At least 1 uppercase letter (`A-Z`)
  - At least 1 lowercase letter (`a-z`)
  - At least 1 numeric digit (`0-9`)
  - At least 1 special symbol (`!@#$%^&*`)
  - Blacklist check against common passwords (`password123`, `admin`, etc.)
  - Identity leak prevention (no username or full name substrings)

### 4. Multi-Tier Role-Based Access Control (RBAC)
- **👨‍⚕️ Clinician / Doctor Workstation**: Diagnostic image upload, 3D mesh rendering, DICOM PDF export, and patient portal onboarding.
- **🧑 Patient Portal**: Secure, read-only personal longitudinal scan history, MRN tracking, and diagnostic summaries.
- **🛡️ Systems Administrator**: User provisioning, role assignment, system health monitoring, and activity audit logging.

### 5. Clinical RAG & Multimodal AI Assistant
- Local knowledge-base retrieval engine indexed with clinical neuro-oncology guidelines.
- Multimodal query answering powered by Google Gemini and localized vector search.

---

## 📂 Project Structure

```directory
neuroscan-ai/
├── .streamlit/
│   └── config.toml               # Streamlit UI theme configuration
├── dataset/                      # Brain MRI datasets
│   ├── preprocessed/             # 4-class classification images (Training & Testing)
│   └── Segmentation-dataset/     # 512x512 MRI scans and ground-truth binary masks
├── models/                       # Classifier checkpoints & performance logs
│   ├── best_efficientnet_b0.pth  # Fine-tuned EfficientNet-B0 state dict
│   ├── metrics.json              # Classification metrics (Precision/Recall/F1)
│   └── confusion_matrix.png      # Confusion matrix visual
├── seg_models/                   # Segmentation checkpoints & metrics
│   ├── best_unet_effb0.pth       # Fine-tuned U-Net state dict
│   ├── seg_metrics.json          # Segmentation test evaluation metrics
│   └── prediction_samples.png    # Validation mask predictions
├── app.py                        # Main clinical workstation application
├── database.py                   # MySQL / SQLite backend with bcrypt auth
├── volume_engine.py              # 3D Marching Cubes & Plotly mesh generator
├── rag_engine.py                 # Retrieval-Augmented Generation engine
├── train.py                      # Classifier training pipeline (AMP enabled)
├── seg_train.py                  # U-Net segmentation training pipeline
├── verify.py                     # Environment & CUDA hardware check
├── requirements.txt              # Production dependency specifications
└── README.md                     # Documentation
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.10 or 3.11 installed.
- (Optional) NVIDIA GPU with CUDA 12.1+ for accelerated inference and 3D rendering.

### 2. Environment Setup
```powershell
# Clone repository & navigate to project directory
git clone https://github.com/your-org/neuroscan-ai.git
cd neuroscan-ai

# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Verify GPU & Environment
```powershell
python verify.py
```

---

## 🖥️ Launching the Application

Start the local workstation server:

```powershell
python -m streamlit run app.py --server.port 8501
```

Open your browser to: **[http://localhost:8501](http://localhost:8501)**

### Default System Accounts:

| Role | Username | Default Password | Access Level |
|---|---|---|---|
| **Clinician** | `doctor` | `brain123` | Full diagnostic suite, 3D viewer, patient onboarding, PDF reports |
| **Patient** | `patient` | `patient123` | Read-only access to personal scan records and findings |
| **Administrator**| `admin` | `neuro2025` | Clinician provisioning, role management, and audit trail |

---

## 📈 Model Performance Metrics

### EfficientNet-B0 Classifier (1,600 Test Scans)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Glioma** | 99.4% | 82.5% | 90.2% | 400 |
| **Meningioma** | 88.8% | 99.3% | 93.7% | 400 |
| **No Tumor** | 95.0% | 100.0% | 97.4% | 400 |
| **Pituitary** | 99.5% | 99.5% | 99.5% | 400 |
| **Overall Accuracy** | — | — | **95.3%** | 1,600 |
| **Macro Average** | **95.7%** | **95.3%** | **95.2%** | 1,600 |

### U-Net Segmenter (Holdout Test Set)

- **Test Dice Similarity Coefficient**: **81.39%**
- **Test Mean IoU (Jaccard Index)**: **72.67%**
- **Validation Dice Score (Best Epoch)**: **79.74%**

---

## 🔒 Security & HIPAA/GDPR Compliance

1. **Zero Plaintext Passwords**: Every credential is salt-hashed with `bcrypt` (work factor 12).
2. **SQL Injection Defense**: 100% of database operations use parameterized queries (`%s` / `?`).
3. **Audit Trail**: Real-time logging of authentication attempts, patient registrations, and diagnostic exports.
4. **Data Isolation**: Multi-tenant separation between doctor, patient, and administrator profiles.

---

## 🛡️ Medical Disclaimer
*NeuroScan AI is intended for research, educational, and developer evaluation purposes only. It is not an FDA/CE-cleared medical device and should not be used as a substitute for professional medical diagnosis or clinical decision-making.*
