import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import json
import os
import sys
import traceback
import cv2
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ProHealth Brain Tumor MRI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session State Init ────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "landing"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "users" not in st.session_state:
    # Demo accounts: {username: password}
    st.session_state.users = {
        "doctor":  "brain123",
        "admin":   "neuro2025",
        "demo":    "demo",
    }
if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""
if "patient_age" not in st.session_state:
    st.session_state.patient_age = 25
if "patient_gender" not in st.session_state:
    st.session_state.patient_gender = "Not specified"

# ── CSS (DICOM Clinical Workstation Theme + Landing + Auth) ────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ── Base resets ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Outfit', sans-serif;
    color: #E6EDF3;
}
.stApp {
    background-color: #0D1117 !important;
}

/* ── Hide Streamlit chrome ── */
[data-testid="stHeader"] { display: none !important; }
header { visibility: hidden; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 0.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* ── Top Bar ── */
.top-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0;
    margin-bottom: 0.85rem;
    border-bottom: 1px solid #21262D;
}
.logo-container {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'Outfit', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #00D4FF;
    letter-spacing: 0.02em;
}
.top-bar-right {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #8B949E;
    display: flex;
    gap: 1.5rem;
    align-items: center;
}
.status-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #3FB950;
    margin-right: 5px;
    animation: pulse-green 2s infinite;
}
@keyframes pulse-green {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Cards ── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #161B22 !important;
    border: 1px solid #21262D !important;
    border-radius: 10px !important;
    padding: 0.85rem !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.4) !important;
    margin-bottom: 0.6rem !important;
}

/* ── Card titles ── */
.pro-card-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #21262D;
}

/* ── File uploader ── */
div[data-testid="stFileUploader"] {
    background-color: #0D1117 !important;
    border: 1px dashed #30363D !important;
    border-radius: 8px !important;
    padding: 0.4rem !important;
}

/* ── Pipeline steps ── */
.pipe-step-card {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    padding: 0.45rem 0.6rem;
    border-radius: 7px;
    margin-bottom: 0.35rem;
    background-color: #0D1117;
    border: 1px solid #21262D;
    transition: all 0.2s ease;
}
.pipe-step-card.active {
    background-color: #0D1F33;
    border-color: #00D4FF;
}
.pipe-step-card.done {
    background-color: #0D1F18;
    border-color: #3FB950;
}
.pipe-step-badge {
    width: 20px; height: 20px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.68rem;
    flex-shrink: 0;
}
.pipe-step-badge.active { background-color: #00D4FF; color: #0D1117; }
.pipe-step-badge.done   { background-color: #3FB950; color: #0D1117; }
.pipe-step-badge.todo   { background-color: #21262D; color: #8B949E; }
.pipe-step-title {
    font-size: 0.78rem;
    font-weight: 500;
    color: #E6EDF3;
}

/* ── Confidence ring ── */
.circle-progress-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.75rem;
}
.circle-progress {
    position: relative;
    width: 58px; height: 58px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #21262D;
}
.circle-val {
    position: relative;
    z-index: 10;
    font-size: 0.88rem;
    font-weight: 700;
    color: #E6EDF3;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Progress bars ── */
.bar-wrap {
    width: 100%;
    background-color: #21262D;
    height: 5px;
    border-radius: 99px;
    overflow: hidden;
    margin-top: 0.2rem;
}
.bar-fill {
    height: 100%;
    border-radius: 99px;
}

/* ── Viewport labels ── */
.viewport-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #8B949E;
    margin-top: 0.4rem;
    display: flex;
    justify-content: space-between;
}

/* ── st.metric overrides ── */
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    color: #E6EDF3 !important;
}
[data-testid="stMetricLabel"] {
    text-transform: uppercase !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.06em !important;
    color: #8B949E !important;
}

/* ── Misc ── */
.eyebrow {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #8B949E;
    margin-bottom: 0.4rem;
}
.disclaimer-text {
    font-size: 0.68rem;
    color: #484F58;
    margin-top: 1rem;
    line-height: 1.5;
    text-align: center;
    border-top: 1px solid #21262D;
    padding-top: 0.6rem;
}

/* ── st.info / st.error ── */
[data-testid="stAlert"] {
    background-color: #161B22 !important;
    border-color: #21262D !important;
    border-radius: 8px !important;
    color: #8B949E !important;
}

/* ════════════════════════════════════════════════
   LANDING PAGE STYLES
════════════════════════════════════════════════ */

.landing-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 0;
    margin-bottom: 0;
    border-bottom: 1px solid #21262D;
}
.landing-logo {
    font-family: 'Outfit', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #00D4FF;
    letter-spacing: 0.03em;
}
.landing-nav-links {
    display: flex;
    gap: 1.2rem;
    align-items: center;
}

.hero-section {
    background: linear-gradient(135deg, #0D1117 0%, #0D1F33 40%, #0D1117 100%);
    border: 1px solid #21262D;
    border-radius: 20px;
    padding: 4rem 3rem;
    margin: 1.5rem 0 2rem 0;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero-section::before {
    content: '';
    position: absolute;
    top: -60px; left: 50%;
    transform: translateX(-50%);
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(0,212,255,0.12) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-tag {
    display: inline-block;
    background: rgba(0,212,255,0.1);
    color: #00D4FF;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.3rem 1rem;
    border-radius: 99px;
    border: 1px solid rgba(0,212,255,0.25);
    margin-bottom: 1.2rem;
}
.hero-headline {
    font-family: 'Outfit', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #E6EDF3;
    line-height: 1.2;
    margin: 0 auto 1rem auto;
    max-width: 680px;
}
.hero-headline span {
    background: linear-gradient(90deg, #00D4FF, #5AC8FA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    color: #8B949E;
    font-size: 1.05rem;
    max-width: 520px;
    margin: 0 auto 2rem auto;
    line-height: 1.6;
}
.hero-brain {
    font-size: 5rem;
    display: block;
    margin-bottom: 1.5rem;
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-12px); }
}
.hero-btn-primary {
    display: inline-block;
    background: linear-gradient(135deg, #00D4FF, #0099BB);
    color: #0D1117 !important;
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 0.92rem;
    padding: 0.75rem 2.2rem;
    border-radius: 99px;
    text-decoration: none;
    border: none;
    cursor: pointer;
    transition: all 0.2s ease;
    box-shadow: 0 4px 20px rgba(0,212,255,0.35);
    margin-right: 0.8rem;
}
.hero-btn-secondary {
    display: inline-block;
    background: transparent;
    color: #8B949E !important;
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 0.92rem;
    padding: 0.75rem 2.2rem;
    border-radius: 99px;
    text-decoration: none;
    border: 1px solid #30363D;
    cursor: pointer;
    transition: all 0.2s ease;
}
.hero-stats {
    display: flex;
    justify-content: center;
    gap: 3rem;
    margin-top: 2.5rem;
    padding-top: 2rem;
    border-top: 1px solid #21262D;
}
.hero-stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #00D4FF;
    display: block;
}
.hero-stat-lbl {
    font-size: 0.72rem;
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

/* ── Services Section ── */
.section-eyebrow {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #00D4FF;
    font-weight: 600;
    text-align: center;
    margin-bottom: 0.5rem;
}
.section-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #E6EDF3;
    text-align: center;
    margin-bottom: 0.5rem;
}
.section-sub {
    color: #8B949E;
    font-size: 0.92rem;
    text-align: center;
    margin-bottom: 2rem;
}

.service-card {
    background: linear-gradient(145deg, #161B22, #0D1117);
    border: 1px solid #21262D;
    border-radius: 14px;
    padding: 1.5rem 1.2rem;
    text-align: center;
    transition: all 0.25s ease;
    height: 100%;
    position: relative;
    overflow: hidden;
}
.service-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--card-color, #00D4FF), transparent);
    opacity: 0.6;
}
.service-card:hover {
    border-color: #30363D;
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
}
.service-icon {
    font-size: 2.2rem;
    margin-bottom: 0.8rem;
    display: block;
}
.service-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #E6EDF3;
    margin-bottom: 0.5rem;
}
.service-desc {
    font-size: 0.82rem;
    color: #8B949E;
    line-height: 1.55;
    margin-bottom: 1rem;
}
.service-badge {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    padding: 0.2rem 0.65rem;
    border-radius: 99px;
    text-transform: uppercase;
}

/* ════════════════════════════════════════════════
   LOGIN / SIGNUP PAGE STYLES
════════════════════════════════════════════════ */

.auth-wrapper {
    display: flex;
    justify-content: center;
    align-items: flex-start;
    padding-top: 1rem;
}
.auth-card {
    background: linear-gradient(145deg, #161B22, #0D1117);
    border: 1px solid #21262D;
    border-radius: 18px;
    padding: 2.5rem 2rem;
    width: 100%;
    box-shadow: 0 8px 40px rgba(0,0,0,0.5);
}
.auth-icon {
    width: 56px; height: 56px;
    border-radius: 50%;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.25);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    margin: 0 auto 1.2rem auto;
}
.auth-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #E6EDF3;
    text-align: center;
    margin-bottom: 0.3rem;
}
.auth-sub {
    font-size: 0.82rem;
    color: #8B949E;
    text-align: center;
    margin-bottom: 1.8rem;
}
.auth-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.3rem;
}

/* ── Input overrides for auth form ── */
.auth-section div[data-testid="stTextInput"] input {
    background-color: #0D1117 !important;
    border: 1px solid #30363D !important;
    border-radius: 8px !important;
    color: #E6EDF3 !important;
    font-family: 'Inter', sans-serif !important;
}
.auth-section div[data-testid="stTextInput"] input:focus {
    border-color: #00D4FF !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important;
}

/* ── st.button overrides ── */
.stButton > button {
    background: linear-gradient(135deg, #00D4FF, #0099BB) !important;
    color: #0D1117 !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.5rem !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
.stButton > button:hover {
    box-shadow: 0 4px 16px rgba(0,212,255,0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── Patient info card ── */
.patient-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 99px;
    padding: 0.2rem 0.75rem;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    color: #00D4FF;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASSES          = ["glioma", "meningioma", "notumor", "pituitary"]
MODEL_PATH       = r"C:\TumorOI\models\best_efficientnet_b0.pth"
METRICS_PATH     = r"C:\TumorOI\models\metrics.json"
SEG_MODEL_PATH   = r"C:\TumorOI\seg_models\best_unet_effb0.pth"
SEG_METRICS_PATH = r"C:\TumorOI\seg_models\seg_metrics.json"
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEG_IMG_SIZE     = 256

TUMOR_CLASSES = {"glioma", "meningioma", "pituitary"}

GUARDRAIL_MODEL_PATH = r"C:\TumorOI\models\best_guardrail.pth"
GUARDRAIL_THRESHOLD  = 0.85

# ── Model Loaders ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_guardrail_model():
    if not os.path.exists(GUARDRAIL_MODEL_PATH):
        return None, "Model file not found"
    try:
        model = efficientnet_b0(weights=None)
        in_feat = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_feat, 2)
        )
        state = torch.load(GUARDRAIL_MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state)
        model.to(DEVICE).eval()
        return model, None
    except Exception as e:
        return None, str(e)

@st.cache_resource(show_spinner=False)
def load_classifier():
    model = efficientnet_b0(weights=None)
    in_feat = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_feat, len(CLASSES))
    )
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model

@st.cache_resource(show_spinner=False)
def load_segmentation_model():
    try:
        import segmentation_models_pytorch as smp
        model = smp.Unet(
            encoder_name    = "efficientnet-b0",
            encoder_weights = None,
            in_channels     = 3,
            classes         = 1,
            activation      = None,
        )
        state = torch.load(SEG_MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state)
        model.to(DEVICE).eval()
        return model, None
    except Exception as e:
        return None, traceback.format_exc()

@st.cache_data(show_spinner=False)
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None

@st.cache_data(show_spinner=False)
def load_seg_metrics():
    if os.path.exists(SEG_METRICS_PATH):
        with open(SEG_METRICS_PATH) as f:
            return json.load(f)
    return None

# ── Transforms ────────────────────────────────────────────────────────────────
clf_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Helper Functions ──────────────────────────────────────────────────────────
def is_mri_image(pil_img: Image.Image, guardrail_model=None):
    img_rgb  = np.array(pil_img.convert("RGB"))
    img_gray = np.array(pil_img.convert("L"))
    img_hsv  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mean_sat = img_hsv[:, :, 1].mean()
    if mean_sat > 60:
        return False, "The image contains color saturation (brain MRI scans must be grayscale)."
    mean_brightness = img_gray.mean()
    if mean_brightness > 200:
        return False, "The image average brightness is too high."
    dark_ratio = (img_gray < 40).sum() / img_gray.size
    if dark_ratio < 0.15:
        return False, "The image lacks standard dark background contrast."
    bright_ratio = (img_gray > 60).sum() / img_gray.size
    if bright_ratio < 0.05:
        return False, "The image is too dark (insufficient scan data)."
        
    # Run deep learning guardrail classifier
    if guardrail_model is not None:
        try:
            tensor = clf_transform(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                outputs = guardrail_model(tensor)
                probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            # Index 1 is MRI, Index 0 is Non-MRI
            mri_prob = float(probs[1])
            if mri_prob < GUARDRAIL_THRESHOLD:
                return False, "The image is not recognized as a brain MRI scan."
            return True, f"MRI scan format verified (Confidence: {mri_prob*100:.1f}%)"
        except Exception as e:
            return True, "MRI scan format verified (Defaulted to heuristics)"
            
    return True, "MRI scan format verified"

def predict_class(model, pil_img: Image.Image):
    tensor = clf_transform(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    return CLASSES[idx], probs

def predict_segmentation(seg_model, pil_img: Image.Image):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    img_rgb  = np.array(pil_img.convert("RGB"))
    orig_h, orig_w = img_rgb.shape[:2]
    img_resized = cv2.resize(img_rgb, (SEG_IMG_SIZE, SEG_IMG_SIZE))
    img_norm = (img_resized.astype(np.float32) / 255.0 - mean) / std
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        logits   = seg_model(tensor)
        prob_map = torch.sigmoid(logits)[0, 0].cpu().numpy()
    prob_map_orig = cv2.resize(prob_map, (orig_w, orig_h))
    binary_mask   = (prob_map_orig > 0.5).astype(np.uint8)
    return binary_mask, prob_map_orig

def overlay_mask_on_image(pil_img: Image.Image, binary_mask: np.ndarray, alpha: float = 0.45):
    img_rgb = np.array(pil_img.convert("RGB"))
    overlay = img_rgb.copy()
    overlay[binary_mask == 1] = [0, 102, 255] # ProHealth Blue overlay color
    blended = cv2.addWeighted(img_rgb, 1 - alpha, overlay, alpha, 0)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (0, 102, 255), 2)
    return Image.fromarray(blended)

def compute_tumor_area(binary_mask: np.ndarray, pil_img: Image.Image):
    """Pixel count and estimated physical area assuming ~0.5mm/pixel MRI resolution."""
    pixel_count = int(binary_mask.sum())
    total_pixels = binary_mask.size
    coverage_pct = 100.0 * pixel_count / total_pixels
    # Typical brain MRI: ~0.5mm per pixel at standard resolution
    mm_per_pixel = 0.5
    area_mm2 = pixel_count * (mm_per_pixel ** 2)
    area_cm2 = area_mm2 / 100.0
    return pixel_count, total_pixels, coverage_pct, area_mm2, area_cm2

def compute_shape_analysis(binary_mask: np.ndarray):
    """Returns circularity, compactness, and solidity of the largest tumor contour."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    # Circularity: 1.0 = perfect circle, <1 = less circular
    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
    # Compactness: inverse of circularity normalised to 0-1 (1 = most compact)
    compactness = min(circularity, 1.0)  # same formula, alias for clarity in UI
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    # Overall shape label driven by circularity
    if circularity >= 0.60: shape_label, shape_color = "Regular",   "#34C759"
    else:                   shape_label, shape_color = "Irregular",  "#FF3B30"
    return {
        "circularity":  circularity,
        "compactness":  compactness,
        "solidity":     solidity,
        "shape_label":  shape_label,
        "shape_color":  shape_color,
    }

def compute_confidence_stats(prob_map: np.ndarray, binary_mask: np.ndarray):
    """Statistics of the model's raw confidence scores inside the segmented region."""
    tumor_probs = prob_map[binary_mask == 1]
    if tumor_probs.size == 0:
        return None
    return {
        "mean": float(tumor_probs.mean()),
        "max":  float(tumor_probs.max()),
        "min":  float(tumor_probs.min()),
        "std":  float(tumor_probs.std()),
        "high_conf_pct": float((tumor_probs >= 0.75).sum() / tumor_probs.size * 100),
    }

# ── Grad-CAM Helpers ─────────────────────────────────────────────────────────
def generate_gradcam(model, pil_img: Image.Image, class_idx: int):
    """Produce a Grad-CAM saliency map for the given class using the last conv block."""
    model.eval()
    # EfficientNet-B0: hook the last feature block
    target_layer = model.features[-1]
    gradients, activations = [], []

    def _save_grad(grad):   gradients.append(grad)
    def _fwd_hook(m, inp, out):
        activations.append(out)
        out.register_hook(_save_grad)

    handle = target_layer.register_forward_hook(_fwd_hook)
    tensor = clf_transform(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    tensor.requires_grad_(True)
    output = model(tensor)
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0, class_idx] = 1.0
    output.backward(gradient=one_hot)
    handle.remove()

    # Pool gradients across spatial dims and weight activations
    grads  = gradients[0][0]            # (C, H, W)
    acts   = activations[0][0]          # (C, H, W)
    weights = grads.mean(dim=(1, 2))    # (C,)
    cam = (weights[:, None, None] * acts).sum(0)  # (H, W)
    cam = torch.relu(cam).cpu().detach().numpy()
    # Normalize to 0-1
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)
    # Resize to original image dimensions
    orig_w, orig_h = pil_img.size
    cam_resized = cv2.resize(cam, (orig_w, orig_h))
    return cam_resized

def overlay_gradcam(pil_img: Image.Image, cam: np.ndarray, alpha: float = 0.5):
    """Blend Grad-CAM heatmap with original MRI using a clinical red-yellow colormap."""
    img_rgb = np.array(pil_img.convert("RGB"))
    cam_uint8 = (cam * 255).astype(np.uint8)
    heatmap   = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap   = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended   = cv2.addWeighted(img_rgb, 1 - alpha, heatmap, alpha, 0)
    return Image.fromarray(blended), Image.fromarray(heatmap)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

def render_landing_page():
    """Full-width landing page with hero banner and service cards."""

    # ── Navigation Bar ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="landing-nav">
        <div class="landing-logo">🧠&nbsp; NeuroScan AI</div>
        <div class="landing-nav-links">
            <span style="font-size:0.85rem; color:#8B949E;">Brain Tumor Analysis Platform</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Hero Section ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-section">
        <span class="hero-brain">🧠</span>
        <div class="hero-tag">AI-Powered Medical Imaging</div>
        <h1 class="hero-headline">
            Discover hope beyond brain tumors with <span>NeuroScan AI</span>
        </h1>
        <p class="hero-sub">
            An end-to-end deep learning platform for brain MRI classification
            and segmentation — delivering radiologist-grade insights in seconds.
        </p>
        <div class="hero-stats">
            <div>
                <span class="hero-stat-val">95.3%</span>
                <span class="hero-stat-lbl">Classification Accuracy</span>
            </div>
            <div>
                <span class="hero-stat-val">81.4%</span>
                <span class="hero-stat-lbl">Segmentation Dice Score</span>
            </div>
            <div>
                <span class="hero-stat-val">4</span>
                <span class="hero-stat-lbl">Tumor Classes Detected</span>
            </div>
            <div>
                <span class="hero-stat-val">&lt;100ms</span>
                <span class="hero-stat-lbl">Inference Latency</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hero CTA Buttons
    h1, h2, h3 = st.columns([2, 1, 2])
    with h2:
        if st.button("🚀 Try Now — Free", use_container_width=True, key="hero_try_now"):
            st.session_state.page = "login"
            st.rerun()

    st.markdown("<div style='margin-bottom:0.5rem'></div>", unsafe_allow_html=True)

    # ── Services Section ──────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin: 2.5rem 0 1rem 0;">
        <p class="section-eyebrow">What We Offer</p>
        <h2 class="section-title">Our Excellent Services</h2>
        <p class="section-sub">
            Cutting-edge AI tools designed to assist radiologists and researchers
            with comprehensive brain tumor diagnostics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Service card data
    services = [
        ("🔍", "Brain Tumor Detection",
         "Accurately detects and localizes brain tumors from MRI scans using a fine-tuned EfficientNet-B0 classifier with 95.3% accuracy.",
         "EfficientNet-B0", "#00D4FF"),
        ("🎯", "Brain Tumor Segmentation",
         "Produces pixel-level segmentation masks using U-Net with an EfficientNet encoder — achieving 81.4% Dice score on holdout tests.",
         "U-Net · Dice 81.4%", "#5AC8FA"),
        ("📊", "Class Prediction",
         "Predicts one of four tumor classes: Glioma, Meningioma, Pituitary, or No Tumor — with per-class confidence probabilities.",
         "4-Class · 95.3% Acc", "#34C759"),
        ("📐", "Brain Tumor Area",
         "Calculates the cross-sectional lesion area in mm² and cm² based on pixel-level segmentation at standard MRI resolution.",
         "0.5 mm/pixel resolution", "#FF9500"),
        ("🔬", "Shape Analysis",
         "Computes morphological features including circularity, compactness, and solidity to characterize tumor geometry.",
         "Morphology · Grad-CAM", "#FF3B30"),
        ("🩺", "XAI · Grad-CAM",
         "Generates explainable AI saliency maps highlighting which regions of the MRI the classifier weighted most for its prediction.",
         "Explainable AI", "#BF5AF2"),
    ]

    row1 = st.columns(3, gap="medium")
    row2 = st.columns(3, gap="medium")
    rows = [row1, row2]

    for i, (icon, title, desc, badge, color) in enumerate(services):
        col = rows[i // 3][i % 3]
        with col:
            st.markdown(f"""
            <div class="service-card" style="--card-color:{color};">
                <span class="service-icon">{icon}</span>
                <div class="service-title">{title}</div>
                <div class="service-desc">{desc}</div>
                <span class="service-badge" style="background:rgba(255,255,255,0.06);
                      color:{color}; border:1px solid {color}44;">{badge}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin:2rem 0;'></div>", unsafe_allow_html=True)

    # ── Bottom CTA Strip ──────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0D1F33,#0D1117);
                border:1px solid #21262D; border-radius:16px;
                padding:2rem; text-align:center; margin-bottom:1rem;">
        <h3 style="font-family:'Outfit',sans-serif; font-size:1.4rem;
                   font-weight:700; color:#E6EDF3; margin:0 0 0.5rem 0;">
            Ready to analyze your MRI scan?
        </h3>
        <p style="color:#8B949E; font-size:0.9rem; margin:0 0 1.5rem 0;">
            Sign in or create a free account to access the full diagnostic suite.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 1, 2])
    with c2:
        if st.button("Get Started →", use_container_width=True, key="bottom_cta"):
            st.session_state.page = "login"
            st.rerun()

    st.markdown("""
    <div class="disclaimer-text">
        🔒 Research &amp; educational use only &nbsp;·&nbsp;
        Not for primary clinical diagnosis &nbsp;·&nbsp;
        Patient data processed locally
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────

def render_login_page():
    """Centered login / sign-up card."""

    # Back to home link
    back_col, _, _ = st.columns([1, 3, 1])
    with back_col:
        if st.button("← Back to Home", key="back_home"):
            st.session_state.page = "landing"
            st.rerun()

    st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)

    # Center the card
    _, center_col, _ = st.columns([1, 1.4, 1])
    with center_col:
        st.markdown("""
        <div style="text-align:center; margin-bottom:1.5rem;">
            <div class="auth-icon">🔐</div>
            <h2 class="auth-title">Welcome to NeuroScan AI</h2>
            <p class="auth-sub">Sign in to your clinical account or create a new one</p>
        </div>
        """, unsafe_allow_html=True)

        # Tab switcher
        tab = st.radio(
            "Select action",
            ["🔑  Login", "📝  Sign Up"],
            horizontal=True,
            label_visibility="collapsed",
            key="auth_tab",
        )

        st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)

        # ── LOGIN TAB ─────────────────────────────────────────────────────────
        if tab == "🔑  Login":
            with st.container(border=True):
                st.markdown("<div class='auth-section'>", unsafe_allow_html=True)

                st.markdown("<div class='auth-label'>Username</div>", unsafe_allow_html=True)
                login_user = st.text_input(
                    "Username", placeholder="Enter your username",
                    label_visibility="collapsed", key="login_user"
                )

                st.markdown("<div class='auth-label' style='margin-top:0.8rem;'>Password</div>", unsafe_allow_html=True)
                login_pass = st.text_input(
                    "Password", placeholder="Enter your password",
                    type="password", label_visibility="collapsed", key="login_pass"
                )

                st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)

                if st.button("Login →", key="do_login", use_container_width=True):
                    if not login_user or not login_pass:
                        st.error("Please fill in both fields.")
                    elif login_user in st.session_state.users and \
                         st.session_state.users[login_user] == login_pass:
                        st.session_state.logged_in = True
                        st.session_state.username  = login_user
                        st.session_state.page      = "dashboard"
                        st.success(f"Welcome back, {login_user}! Redirecting…")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password.")

                st.markdown("</div>", unsafe_allow_html=True)

            # Demo account hint
            st.markdown("""
            <div style="background:rgba(0,212,255,0.05); border:1px solid rgba(0,212,255,0.15);
                        border-radius:8px; padding:0.75rem 1rem; margin-top:0.6rem;">
                <p style="font-size:0.72rem; color:#8B949E; margin:0 0 0.3rem 0;
                           font-weight:600; text-transform:uppercase; letter-spacing:0.06em;">
                    Demo Credentials
                </p>
                <p style="font-family:'JetBrains Mono',monospace; font-size:0.78rem;
                           color:#00D4FF; margin:0;">
                    Username: <b>demo</b> &nbsp;·&nbsp; Password: <b>demo</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

        # ── SIGN UP TAB ───────────────────────────────────────────────────────
        else:
            with st.container(border=True):
                st.markdown("<div class='auth-section'>", unsafe_allow_html=True)

                st.markdown("<div class='auth-label'>New Username</div>", unsafe_allow_html=True)
                new_user = st.text_input(
                    "New Username", placeholder="Choose a username",
                    label_visibility="collapsed", key="reg_user"
                )

                st.markdown("<div class='auth-label' style='margin-top:0.8rem;'>Password</div>", unsafe_allow_html=True)
                new_pass = st.text_input(
                    "Password", placeholder="Create a password",
                    type="password", label_visibility="collapsed", key="reg_pass"
                )

                st.markdown("<div class='auth-label' style='margin-top:0.8rem;'>Confirm Password</div>", unsafe_allow_html=True)
                confirm_pass = st.text_input(
                    "Confirm Password", placeholder="Re-enter your password",
                    type="password", label_visibility="collapsed", key="reg_confirm"
                )

                st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)

                if st.button("Create Account →", key="do_register", use_container_width=True):
                    if not new_user or not new_pass or not confirm_pass:
                        st.error("Please fill in all fields.")
                    elif new_user in st.session_state.users:
                        st.error("❌ Username already exists. Please choose another.")
                    elif new_pass != confirm_pass:
                        st.error("❌ Passwords do not match.")
                    elif len(new_pass) < 4:
                        st.error("❌ Password must be at least 4 characters.")
                    else:
                        st.session_state.users[new_user] = new_pass
                        st.session_state.logged_in = True
                        st.session_state.username  = new_user
                        st.session_state.page      = "dashboard"
                        st.success(f"✅ Account created! Welcome, {new_user}!")
                        st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-text" style="margin-top:2rem;">
        🔒 Research &amp; educational use only &nbsp;·&nbsp;
        Patient data processed locally &nbsp;·&nbsp; Not for clinical diagnosis
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────

def render_dashboard_page():
    """Main clinical dashboard — full existing AI pipeline with Patient Info panel."""

    # ── Load Models ────────────────────────────────────────────────────────────
    with st.spinner("Initializing models..."):
        clf_model   = load_classifier()
        seg_model, seg_err = load_segmentation_model()
        guardrail_model, guardrail_err = load_guardrail_model()
        metrics     = load_metrics()
        seg_metrics = load_seg_metrics()

    # ── Top Bar Header ─────────────────────────────────────────────────────────
    patient_display = st.session_state.patient_name if st.session_state.patient_name else "No Patient"
    username_display = st.session_state.username or "User"

    st.markdown(f"""
    <div class="top-bar">
        <div class="logo-container">
            🧠&nbsp; NeuroScan AI
        </div>
        <div class="top-bar-right">
            <span><span class="status-dot"></span>Models Online</span>
            <span>EfficientNet-B0 &nbsp;·&nbsp; U-Net</span>
            <span class="patient-chip">👤 {patient_display}</span>
            <span style="color:#00D4FF; font-weight:600;">@{username_display}</span>
            <span style="color:#30363D;">v2.0</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Layout Grid ────────────────────────────────────────────────────────────
    col_left, col_center, col_right = st.columns([1.1, 2, 1.2], gap="large")

    # States & Variables
    step = 0
    uploaded = None
    pil_img = None
    is_mri = False
    mri_reason = ""
    predicted_class = None
    probs = None
    has_tumor = False
    binary_mask = None
    prob_map = None
    gradcam_img = None
    gradcam_raw = None

    # LEFT COLUMN: Patient Info + Pipeline + File Upload
    with col_left:
        st.markdown("<p style='font-size:0.68rem; text-transform:uppercase; letter-spacing:0.1em; color:#8B949E; margin-bottom:0.75rem;'>Patient &amp; Workflow</p>", unsafe_allow_html=True)

        # ── Patient Info Card ─────────────────────────────────────────────────
        with st.container(border=True):
            st.markdown("<div class='pro-card-title'>👤 Patient Information</div>", unsafe_allow_html=True)

            p_name = st.text_input(
                "Patient Name",
                value=st.session_state.patient_name,
                placeholder="Enter patient name…",
                key="pi_name",
                label_visibility="visible",
            )
            st.session_state.patient_name = p_name

            p_col1, p_col2 = st.columns(2)
            with p_col1:
                p_age = st.number_input(
                    "Age",
                    min_value=1, max_value=120,
                    value=st.session_state.patient_age,
                    key="pi_age",
                )
                st.session_state.patient_age = p_age
            with p_col2:
                p_gender = st.selectbox(
                    "Gender",
                    ["Not specified", "Male", "Female", "Other"],
                    index=["Not specified", "Male", "Female", "Other"].index(
                        st.session_state.patient_gender
                    ),
                    key="pi_gender",
                )
                st.session_state.patient_gender = p_gender

            # Patient summary chip row
            if p_name:
                st.markdown(f"""
                <div style="margin-top:0.6rem; display:flex; gap:0.5rem; flex-wrap:wrap;">
                    <span style="background:rgba(63,185,80,0.1); color:#3FB950;
                                 border:1px solid rgba(63,185,80,0.25); border-radius:99px;
                                 font-size:0.68rem; padding:0.2rem 0.6rem; font-family:'JetBrains Mono',monospace;">
                        ✓ {p_name}
                    </span>
                    <span style="background:rgba(0,212,255,0.08); color:#00D4FF;
                                 border:1px solid rgba(0,212,255,0.2); border-radius:99px;
                                 font-size:0.68rem; padding:0.2rem 0.6rem; font-family:'JetBrains Mono',monospace;">
                        Age {p_age} · {p_gender}
                    </span>
                </div>
                """, unsafe_allow_html=True)

        # ── MRI Upload Card ───────────────────────────────────────────────────
        with st.container(border=True):
            st.markdown("<div class='pro-card-title'>Upload MRI Scan</div>", unsafe_allow_html=True)
            uploaded = st.file_uploader("Upload MRI", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    # Automatic Pipeline Execution
    if uploaded:
        step = 1
        pil_img = Image.open(uploaded)
        is_mri, mri_reason = is_mri_image(pil_img, guardrail_model)
        if is_mri:
            step = 2
            predicted_class, probs = predict_class(clf_model, pil_img)
            has_tumor = predicted_class in TUMOR_CLASSES
            # Grad-CAM — always run for any valid MRI
            try:
                class_idx   = CLASSES.index(predicted_class)
                cam         = generate_gradcam(clf_model, pil_img, class_idx)
                gradcam_img, gradcam_raw = overlay_gradcam(pil_img, cam, alpha=0.5)
            except Exception:
                gradcam_img, gradcam_raw = None, None
            step = 3
            if has_tumor and seg_model:
                binary_mask, prob_map = predict_segmentation(seg_model, pil_img)
                step = 4
            else:
                step = 4 # Completed (no tumor skips segmentation)

    # Render pipeline cards in col_left
    with col_left:
        with st.container(border=True):
            st.markdown("<div class='pro-card-title'>Analysis Workflow</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="pipe-step-card {'done' if step > 0 else 'active'}">
                <div class="pipe-step-badge {'done' if step > 0 else 'active'}">{"✓" if step > 0 else "1"}</div>
                <div class="pipe-step-title">Upload scan</div>
            </div>
            <div class="pipe-step-card {'done' if step > 1 else 'active' if step == 1 else ''}">
                <div class="pipe-step-badge {'done' if step > 1 else 'active' if step == 1 else 'todo'}">{"✓" if step > 1 else "2"}</div>
                <div class="pipe-step-title">MRI guardrail</div>
            </div>
            <div class="pipe-step-card {'done' if step > 2 else 'active' if step == 2 else ''}">
                <div class="pipe-step-badge {'done' if step > 2 else 'active' if step == 2 else 'todo'}">{"✓" if step > 2 else "3"}</div>
                <div class="pipe-step-title">Classification</div>
            </div>
            <div class="pipe-step-card {'done' if step > 3 else 'active' if step == 3 else ''}">
                <div class="pipe-step-badge {'done' if step > 3 else 'active' if step == 3 else 'todo'}">{"✓" if step > 3 else "4"}</div>
                <div class="pipe-step-title">Segmentation</div>
            </div>
            """, unsafe_allow_html=True)

        # Logout button at bottom of left col
        st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)
        if st.button("🔓 Logout", key="logout_btn", use_container_width=True):
            st.session_state.logged_in  = False
            st.session_state.username   = ""
            st.session_state.page       = "landing"
            st.session_state.patient_name   = ""
            st.session_state.patient_age    = 25
            st.session_state.patient_gender = "Not specified"
            st.rerun()

    # CENTER COLUMN: Viewport Grid
    with col_center:
        st.markdown("<p style='font-size:0.68rem; text-transform:uppercase; letter-spacing:0.1em; color:#8B949E; margin-bottom:0.75rem;'>AI Imaging Viewport</p>", unsafe_allow_html=True)

        if pil_img:
            if is_mri:
                if has_tumor and binary_mask is not None:
                    # First Row: Original MRI and Pathology Highlight Overlay (larger 2-column layout)
                    view_c1, view_c2 = st.columns(2, gap="medium")
                    with view_c1:
                        with st.container(border=True):
                            st.markdown("<div class='pro-card-title'>Original MRI</div>", unsafe_allow_html=True)
                            st.image(pil_img, use_container_width=True)
                    with view_c2:
                        with st.container(border=True):
                            st.markdown("<div class='pro-card-title'>Pathology Overlay</div>", unsafe_allow_html=True)
                            overlay_img = overlay_mask_on_image(pil_img, binary_mask, alpha=0.45)
                            st.image(overlay_img, use_container_width=True)
                else:
                    # Healthy scan: show original MRI only
                    with st.container(border=True):
                        st.markdown("<div class='pro-card-title'>Original MRI Scan</div>", unsafe_allow_html=True)
                        st.image(pil_img, use_container_width=True)
                    st.info("No pathology detected. Segmentation skipped.")

                # ── Diagnosis findings card (full width below images) ───────────────
                st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown("<div class='pro-card-title'>Diagnosis</div>", unsafe_allow_html=True)

                    class_info_map = {
                        "glioma":      ("Glioma Findings",       "#00D4FF", "#FF3B30", "Malignant",        "#2A1818"),
                        "meningioma":  ("Meningioma Findings",   "#00D4FF", "#FF9500", "Typically Benign", "#2A2118"),
                        "pituitary":   ("Pituitary Findings",    "#00D4FF", "#5AC8FA", "Typically Benign", "#18222A"),
                        "notumor":     ("No Tumor Detected",     "#00D4FF", "#34C759", "Healthy",          "#182A1A")
                    }
                    lbl, theme_color, stroke_color, status_lbl, status_bg = class_info_map[predicted_class]
                    conf = probs[CLASSES.index(predicted_class)] * 100
                    
                    # If classification confidence is below 70%, flag as uncertain
                    if conf < 70.0:
                        status_lbl = "Uncertain Prediction"
                        stroke_color = "#FF9500"  # Warning Orange
                        status_bg = "#2A2118"
                        
                    # Display warning banner FIRST if confidence is low
                    if conf < 70.0:
                        st.warning("⚠️ Uncertain Prediction — Radiologist Review Recommended")
                    
                    # Split label to color name (e.g. Meningioma) and leave suffix (e.g. Findings) white
                    if " Findings" in lbl:
                        lbl_name = lbl.replace(" Findings", "")
                        lbl_suffix = " Findings"
                    elif " Detected" in lbl:
                        lbl_name = lbl.replace(" Detected", "")
                        lbl_suffix = " Detected"
                    else:
                        lbl_name = lbl
                        lbl_suffix = ""

                    # Patient context row
                    if st.session_state.patient_name:
                        st.markdown(f"""
                        <div style="display:flex; gap:0.5rem; margin-bottom:0.8rem; flex-wrap:wrap;">
                            <span style="background:rgba(0,212,255,0.08); color:#8B949E;
                                         border:1px solid #21262D; border-radius:6px;
                                         font-size:0.72rem; padding:0.25rem 0.6rem;">
                                Patient: <b style="color:#E6EDF3;">{st.session_state.patient_name}</b>
                            </span>
                            <span style="background:rgba(0,212,255,0.08); color:#8B949E;
                                         border:1px solid #21262D; border-radius:6px;
                                         font-size:0.72rem; padding:0.25rem 0.6rem;">
                                Age: <b style="color:#E6EDF3;">{st.session_state.patient_age}</b>
                            </span>
                            <span style="background:rgba(0,212,255,0.08); color:#8B949E;
                                         border:1px solid #21262D; border-radius:6px;
                                         font-size:0.72rem; padding:0.25rem 0.6rem;">
                                Gender: <b style="color:#E6EDF3;">{st.session_state.patient_gender}</b>
                            </span>
                        </div>
                        """, unsafe_allow_html=True)

                    # Render circular progress indicator
                    st.markdown(f"""
                    <div class="circle-progress-container">
                        <div>
                            <h3 style="font-family: 'Outfit', sans-serif; font-weight: 700; color: #E6EDF3; margin: 0; font-size: 1.3rem;"><span style="color: {stroke_color}; text-shadow: 0 0 12px {stroke_color}33;">{lbl_name}</span>{lbl_suffix}</h3>
                            <div style="display: inline-block; background-color: {status_bg}; color: {stroke_color}; font-size: 0.72rem; font-weight: 700; padding: 2px 10px; border-radius: 99px; margin-top: 0.4rem; text-transform: uppercase; letter-spacing: 0.05em; border: 1px solid {stroke_color}44;">{status_lbl}</div>
                            <p style="color: #8B949E; font-size: 0.8rem; margin: 0; margin-top: 0.4rem;">Classification Confidence</p>
                        </div>
                        <div class="circle-progress" style="background: conic-gradient({stroke_color} calc({conf} * 1%), #21262D 0);">
                            <div class="circle-val">{conf:.0f}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Probabilities block
                    st.markdown("<p style='font-size: 0.82rem; font-weight: 600; color: #E6EDF3; margin-top: 1.2rem; margin-bottom: 0.5rem;'>Other Class Distribution</p>", unsafe_allow_html=True)
                    for c, p in sorted(zip(CLASSES, probs), key=lambda x: -x[1]):
                        if c == predicted_class: continue
                        p_pct = p * 100
                        c_lbl = class_info_map[c][0].replace(" Findings", "").replace(" Detected", "")
                        bar_color = class_info_map[c][2]
                        status_lbl_other = class_info_map[c][3]
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem; margin-bottom: 0.25rem;">
                            <span style="color: {bar_color}; font-weight: 600;">{c_lbl} <span style="color: #8B949E; font-size: 0.72rem; font-weight: normal;">({status_lbl_other})</span></span>
                            <span style="font-family:'JetBrains Mono',monospace; font-weight: 700; color: #E6EDF3;">{p_pct:.1f}%</span>
                        </div>
                        <div class="bar-wrap" style="height: 5px; margin-bottom: 0.75rem;">
                            <div class="bar-fill" style="width: {p_pct}%; background-color: {bar_color};"></div>
                        </div>
                        """, unsafe_allow_html=True)

                # ── Tumor Area Calculation Card ────────────────────────────────────────
                st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown("<div class='pro-card-title'>🔬 Tumor Area</div>", unsafe_allow_html=True)
                    if has_tumor and binary_mask is not None:
                        px_count, total_px, cov_pct, area_mm2, area_cm2 = compute_tumor_area(binary_mask, pil_img)
                        c1, c2 = st.columns(2)
                        c1.metric("Area (mm²)",    f"{area_mm2:,.1f}")
                        c2.metric("Area (cm²)",    f"{area_cm2:.2f}")
                        c1.metric("Tumor Pixels",  f"{px_count:,}")
                        c2.metric("Coverage",      f"{cov_pct:.2f}%")
                        st.markdown("""
                        <div style="font-size:0.72rem; color:#8B949E; margin-top:0.4rem;">
                            ℹ️ Estimated at 0.5 mm/pixel (standard MRI resolution)
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='color:#8B949E; font-size:0.82rem; text-align:center; padding:0.75rem 0;'>No tumor detected — area analysis unavailable.</div>", unsafe_allow_html=True)

                # ── Tumor Shape Analysis Card ──────────────────────────────────────────
                st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown("<div class='pro-card-title'>📐 Shape Analysis</div>", unsafe_allow_html=True)
                    if has_tumor and binary_mask is not None:
                        shape = compute_shape_analysis(binary_mask)
                        if shape:
                            s_lbl  = shape["shape_label"]
                            s_col  = shape["shape_color"]
                            circ   = shape["circularity"]
                            comp   = shape["compactness"]
                            sol    = shape["solidity"]

                            st.markdown(f"""
                            <div style="margin-bottom:1rem;">
                                <p style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;
                                          color:#8B949E; margin:0 0 0.3rem 0;">Tumor Shape</p>
                                <div style="display:inline-flex; align-items:center; gap:0.5rem;">
                                    <div style="width:10px; height:10px; border-radius:50%;
                                                background:{s_col};"></div>
                                    <span style="font-family:'Outfit',sans-serif; font-size:1.4rem;
                                                 font-weight:700; color:{s_col};">{s_lbl}</span>
                                </div>
                            </div>
                            <p style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;
                                       color:#8B949E; margin:0 0 0.6rem 0;">Features</p>
                            """, unsafe_allow_html=True)

                            def _feat_row(label, value, bar_pct, color):
                                st.markdown(f"""
                                <div style="margin-bottom:0.65rem;">
                                    <div style="display:flex; justify-content:space-between;
                                                align-items:center; margin-bottom:0.2rem;">
                                        <span style="font-size:0.82rem; color:#E6EDF3; font-weight:500;">{label}</span>
                                        <span style="font-family:'JetBrains Mono',monospace; font-size:0.82rem;
                                                     font-weight:700; color:#E6EDF3;">{value:.3f}</span>
                                    </div>
                                    <div class="bar-wrap">
                                        <div class="bar-fill" style="width:{min(bar_pct,100):.1f}%;
                                             background:{color};"></div>
                                    </div>
                                </div>""", unsafe_allow_html=True)

                            circ_col = "#34C759" if circ >= 0.60 else "#FF3B30"
                            comp_col = "#34C759" if comp >= 0.60 else "#FF3B30"
                            sol_col  = "#34C759" if sol  >= 0.90 else ("#FF9500" if sol >= 0.75 else "#FF3B30")

                            _feat_row("Circularity",  circ, circ * 100, circ_col)
                            _feat_row("Compactness",  comp, comp * 100, comp_col)
                            _feat_row("Solidity",     sol,  sol  * 100, sol_col)
                        else:
                            st.markdown("<div style='color:#8B949E;font-size:0.82rem;'>No contour found in mask.</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='color:#8B949E; font-size:0.82rem; text-align:center; padding:0.75rem 0;'>No tumor detected — shape analysis unavailable.</div>", unsafe_allow_html=True)

                w, h = pil_img.size
                st.markdown(f"""
                <div class="viewport-label">
                    <span>Dimensions: {w} × {h} px</span>
                    <span>File: {uploaded.name}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                with st.container(border=True):
                    st.markdown("<div class='pro-card-title'>Uploaded Image (Rejected)</div>", unsafe_allow_html=True)
                    st.image(pil_img, use_container_width=True)
                st.error("Invalid Scan: The uploaded image is not recognized as a valid brain MRI. Please upload a clear brain MRI scan to proceed.")
        else:
            st.markdown("""
            <div style="background-color:#161B22; border:1px solid #21262D; border-radius:12px;
                        padding:6rem 2rem; text-align:center; height:100%;">
                <span style="font-size:3rem;">📂</span>
                <h4 style="font-family:'Outfit',sans-serif; font-weight:600; color:#E6EDF3;
                           margin-top:1rem;">Diagnostics Queue Empty</h4>
                <p style="color:#8B949E; font-size:0.88rem;">Upload a brain MRI scan from the left panel to begin analysis.</p>
            </div>
            """, unsafe_allow_html=True)

    # RIGHT COLUMN: Readouts & Results
    with col_right:
        st.markdown("<p style='font-size:0.68rem; text-transform:uppercase; letter-spacing:0.1em; color:#8B949E; margin-bottom:0.75rem;'>Clinical Analysis</p>", unsafe_allow_html=True)

        if step >= 2 and probs is not None:
            # ── XAI Grad-CAM Card (top) ──────────────────────────────────────────
            with st.container(border=True):
                st.markdown("<div class='pro-card-title'>🔍 XAI · Grad-CAM</div>", unsafe_allow_html=True)
                if gradcam_img:
                    st.image(gradcam_img, use_container_width=True)
                    if gradcam_raw is not None:
                        cam_gray = np.array(gradcam_raw.convert("L")).astype(np.float32) / 255.0
                        h, w = cam_gray.shape
                        cy, cx = np.unravel_index(cam_gray.argmax(), cam_gray.shape)
                        vq = "Upper" if cy < h // 2 else "Lower"
                        hq = "Left"  if cx < w // 2 else "Right"
                        focus_pct = float(cam_gray[cam_gray >= 0.7].size / cam_gray.size * 100)
                        st.markdown(f"""
                        <div style="margin-top:0.5rem;">
                            <p style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;
                                      color:#8B949E; margin:0 0 0.3rem 0;">Classifier Focus Region</p>
                            <p style="font-family:'Outfit',sans-serif; font-size:0.95rem; font-weight:700;
                                      color:#E6EDF3; margin:0;">{vq}-{hq} Region</p>
                            <p style="font-size:0.78rem; color:#8B949E; margin:0.25rem 0 0.6rem 0;">
                                Peak activation at ({cx}px, {cy}px)
                            </p>
                            <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:0.2rem;">
                                <span style="color:#8B949E;">High-attention area (≥70%)</span>
                                <span style="font-weight:700; color:#00D4FF;">{focus_pct:.1f}%</span>
                            </div>
                            <div class="bar-wrap">
                                <div class="bar-fill" style="width:{min(focus_pct,100):.1f}%; background:#00D4FF;"></div>
                            </div>
                            <p style="font-size:0.70rem; color:#8B949E; margin-top:0.5rem; line-height:1.4;">
                                Red/yellow regions indicate areas the classifier weighted most heavily
                                when predicting <strong style="color:{stroke_color}; text-transform: uppercase; letter-spacing: 0.05em;">{predicted_class}</strong>.
                            </p>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='color:#8B949E;font-size:0.82rem;text-align:center;padding:0.75rem 0;'>Grad-CAM unavailable for this scan.</div>", unsafe_allow_html=True)

            # ── Segmentation Model Metrics Card ───────────────────────────────────
            with st.container(border=True):
                st.markdown("<div class='pro-card-title'>Segmentation Metrics</div>", unsafe_allow_html=True)
                if step == 4 and seg_metrics and seg_model and has_tumor and binary_mask is not None:
                    dice = seg_metrics["test_performance"]["dice"]
                    iou  = seg_metrics["test_performance"]["iou"]
                    cov  = 100.0 * binary_mask.sum() / binary_mask.size
                    px   = int(binary_mask.sum())
                    c1, c2 = st.columns(2)
                    c1.metric("Dice Similarity", f"{dice:.3f}")
                    c2.metric("IoU Score",       f"{iou:.3f}")
                    c1.metric("Tumor Coverage",  f"{cov:.1f}%")
                    c2.metric("Active Pixels",   f"{px:,}")
                else:
                    c1, c2 = st.columns(2)
                    c1.metric("Dice Similarity", "--")
                    c2.metric("IoU Score",       "--")
                    c1.metric("Tumor Coverage",  "--")
                    c2.metric("Active Pixels",   "--")

            # ── Segmentation Confidence Map Card ──────────────────────────────────
            with st.container(border=True):
                st.markdown("<div class='pro-card-title'>📊 Confidence Map</div>", unsafe_allow_html=True)
                if has_tumor and binary_mask is not None and prob_map is not None:
                    conf = compute_confidence_stats(prob_map, binary_mask)
                    if conf:
                        mean_pct = conf["mean"] * 100
                        max_pct  = conf["max"]  * 100
                        hc       = conf["high_conf_pct"]

                        if mean_pct >= 80:   conf_lbl, conf_col = "High",     "#34C759"
                        elif mean_pct >= 55: conf_lbl, conf_col = "Moderate", "#FF9500"
                        else:                conf_lbl, conf_col = "Low",       "#FF3B30"

                        st.markdown(f"""
                        <div style="margin-bottom:1rem;">
                            <p style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;
                                      color:#8B949E; margin:0 0 0.3rem 0;">Confidence Level</p>
                            <div style="display:inline-flex; align-items:center; gap:0.5rem;">
                                <div style="width:10px; height:10px; border-radius:50%;
                                            background:{conf_col};"></div>
                                <span style="font-family:'Outfit',sans-serif; font-size:1.4rem;
                                             font-weight:700; color:{conf_col};">{conf_lbl}</span>
                            </div>
                        </div>
                        <p style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;
                                   color:#8B949E; margin:0 0 0.6rem 0;">Features</p>
                        """, unsafe_allow_html=True)

                        def _conf_row(label, value_str, bar_pct, color):
                            st.markdown(f"""
                            <div style="margin-bottom:0.65rem;">
                                <div style="display:flex; justify-content:space-between;
                                            align-items:center; margin-bottom:0.2rem;">
                                    <span style="font-size:0.82rem; color:#E6EDF3; font-weight:500;">{label}</span>
                                    <span style="font-family:'JetBrains Mono',monospace; font-size:0.82rem;
                                                 font-weight:700; color:#E6EDF3;">{value_str}</span>
                                </div>
                                <div class="bar-wrap">
                                    <div class="bar-fill" style="width:{min(bar_pct,100):.1f}%;
                                         background:{color};"></div>
                                </div>
                            </div>""", unsafe_allow_html=True)

                        mean_col = "#34C759" if mean_pct >= 80 else ("#FF9500" if mean_pct >= 55 else "#FF3B30")
                        peak_col = "#34C759" if max_pct  >= 90 else ("#FF9500" if max_pct  >= 70 else "#FF3B30")
                        hc_col   = "#34C759" if hc       >= 70 else ("#FF9500" if hc       >= 40 else "#FF3B30")

                        _conf_row("Mean Confidence",         f"{mean_pct:.1f}%", mean_pct, mean_col)
                        _conf_row("Peak Confidence",         f"{max_pct:.1f}%",  max_pct,  peak_col)
                        _conf_row("High-conf Region (≥75%)", f"{hc:.1f}%",       hc,       hc_col)
                else:
                    st.markdown("<div style='color:#8B949E; font-size:0.82rem; text-align:center; padding:0.75rem 0;'>No tumor detected — confidence map unavailable.</div>", unsafe_allow_html=True)

        else:
            with st.container(border=True):
                st.markdown("<div style='color:#8B949E; font-size:0.85rem; text-align:center; padding:2rem 0;'>Awaiting scan input to run diagnosis...</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer-text">
            🔒 Research &amp; educational use only &nbsp;·&nbsp; Not for primary clinical diagnosis &nbsp;·&nbsp; Patient data processed locally
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# APP ROUTER
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.page == "landing":
    render_landing_page()

elif st.session_state.page == "login":
    render_login_page()

else:  # "dashboard"
    if st.session_state.logged_in:
        render_dashboard_page()
    else:
        st.session_state.page = "login"
        st.rerun()
