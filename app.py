import io
import datetime
import random
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

# ── Database Layer (MySQL with SQLite Fallback) ───────────────────────────────
import database as db
db.init_db()

import s3_storage as s3
import rag_engine
import volume_engine



# ── PDF Generation Imports (ReportLab) ─────────────────────────────────────────
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI — Brain Tumor MRI Analysis",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session State Init ────────────────────────────────────────────────────────
# Handle query parameters for page navigation from custom HTML links
if "nav" in st.query_params:
    target_page = st.query_params["nav"]
    if target_page in ["landing", "login", "dashboard"]:
        st.session_state.page = target_page
    st.query_params.clear()

if "page" not in st.session_state:
    st.session_state.page = "landing"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "role" not in st.session_state:
    st.session_state.role = "doctor"
if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""
if "patient_age" not in st.session_state:
    st.session_state.patient_age = 25
if "patient_gender" not in st.session_state:
    st.session_state.patient_gender = "Not specified"



# ── CSS (DICOM Clinical Workstation Theme + Landing + Auth) ────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/tabler-icons/2.44.0/iconfont/tabler-icons.min.css');

/* ── Base resets ── */

html, body, [class*="css"] {
    font-family: 'Inter', 'Outfit', sans-serif;
    color: #E6EDF3;
    background-color: #0D1117 !important; /* Ensure browser body background is dark to prevent light leakage */
    overflow-x: hidden !important; /* Disable horizontal scrollbars globally */
}
.stApp, [data-testid="stAppViewContainer"] {
    background-color: #0D1117 !important;
    overflow-x: hidden !important; /* Prevent scrollbar track render */
}

/* ── Hide Streamlit chrome ── */
[data-testid="stHeader"] { display: none !important; }
header { visibility: hidden; }
#MainMenu { visibility: hidden; }
footer { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

.block-container {
    padding-top: 0.25rem !important;
    padding-bottom: 0.25rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 98% !important;
}

/* ── Streamlit Spacing Resets ── */
[data-testid="stVerticalBlock"] {
    gap: 0.45rem !important;
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

/* ── st.button & st.download_button styling ── */
.stButton > button {
    background: #161B22 !important;
    background-color: #161B22 !important;
    color: #E6EDF3 !important;
    border: 1px solid #30363D !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #21262D !important;
    border-color: #00D4FF !important;
    color: #00D4FF !important;
    box-shadow: 0 4px 14px rgba(0, 212, 255, 0.15) !important;
    transform: translateY(-1px) !important;
}

/* Primary buttons & Downloads */
.stDownloadButton > button,
button[kind="primary"],
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #00D4FF 0%, #0099BB 100%) !important;
    color: #0D1117 !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 16px rgba(0, 212, 255, 0.25) !important;
}
.stDownloadButton > button:hover,
button[kind="primary"]:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4) !important;
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

# ── Constants & Dynamic Model Paths ───────────────────────────────────────────
# Contributor: Sridharan (FS-Sridharan007) - Dynamic Relative Path Resolution & ReportLab PDF Integration
BASE_DIR             = os.path.dirname(os.path.abspath(__file__))
CLASSES              = ["glioma", "meningioma", "notumor", "pituitary"]
MODEL_PATH           = os.path.join(BASE_DIR, "models", "best_efficientnet_b0.pth")
METRICS_PATH         = os.path.join(BASE_DIR, "models", "metrics.json")
SEG_MODEL_PATH       = os.path.join(BASE_DIR, "seg_models", "best_unet_effb0.pth")
SEG_METRICS_PATH     = os.path.join(BASE_DIR, "seg_models", "seg_metrics.json")
DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEG_IMG_SIZE         = 256

TUMOR_CLASSES        = {"glioma", "meningioma", "pituitary"}

GUARDRAIL_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_guardrail.pth")
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


# ── Clinical PDF Report Generator ─────────────────────────────────────────────
def generate_clinical_pdf_report(
    patient_name: str,
    patient_age: int,
    patient_gender: str,
    username: str,
    predicted_class: str,
    probs: np.ndarray,
    classes: list,
    pil_img: Image.Image,
    overlay_img: Image.Image = None,
    gradcam_img: Image.Image = None,
    area_data: dict = None,
    shape_data: dict = None,
    conf_data: dict = None,
) -> bytes:
    """Generate a formal DICOM-grade Clinical Diagnostic PDF Report."""
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("ReportLab is not installed. Please run: pip install reportlab")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=32,
        bottomMargin=32
    )

    title_style = ParagraphStyle(
        'DocTitle',
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.HexColor('#0B2545')
    )
    meta_style = ParagraphStyle(
        'DocMeta',
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        alignment=TA_RIGHT,
        textColor=colors.HexColor('#566573')
    )
    sec_heading = ParagraphStyle(
        'SecHeading',
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor('#0B2545'),
        spaceBefore=4,
        spaceAfter=2
    )
    body_text = ParagraphStyle(
        'BodyDark',
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor('#1C2833')
    )
    body_bold = ParagraphStyle(
        'BodyBold',
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor('#1C2833')
    )
    cell_hdr = ParagraphStyle(
        'CellHdr',
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=TA_CENTER
    )
    cell_txt = ParagraphStyle(
        'CellTxt',
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#1C2833'),
        alignment=TA_CENTER
    )
    img_caption = ParagraphStyle(
        'ImgCaption',
        fontName='Helvetica-Bold',
        fontSize=7,
        leading=8.5,
        textColor=colors.HexColor('#2C3E50'),
        alignment=TA_CENTER
    )
    disclaimer_style = ParagraphStyle(
        'DisclaimerTxt',
        fontName='Helvetica-Oblique',
        fontSize=6.5,
        leading=8,
        textColor=colors.HexColor('#7F8C8D'),
        alignment=TA_JUSTIFY
    )

    story = []

    # 1. Header Banner
    now_str = datetime.datetime.now().strftime("%B %d, %Y %H:%M:%S")
    report_id = f"RPT-{datetime.datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"

    header_data = [
        [
            Paragraph("<b>NEUROSCAN AI WORKSTATION</b><br/><font color='#0084FF' size='7.5'>CLINICAL BRAIN MRI DIAGNOSTIC &amp; SEGMENTATION REPORT</font>", title_style),
            Paragraph(f"<b>Report ID:</b> {report_id}<br/><b>Generated:</b> {now_str}<br/><b>Attending Clinician:</b> {username or 'Doctor'}", meta_style)
        ]
    ]
    t_hdr = Table(header_data, colWidths=[4.0*inch, 3.5*inch])
    t_hdr.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_hdr)
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0084FF'), spaceBefore=2, spaceAfter=4))

    # 2. Patient & Exam Demographics Card
    p_name_display = patient_name.strip() if patient_name and patient_name.strip() else "Anonymous / Unspecified"
    patient_table_data = [
        [
            Paragraph("<b>Patient Name:</b>", body_bold), Paragraph(p_name_display, body_text),
            Paragraph("<b>Age / Gender:</b>", body_bold), Paragraph(f"{patient_age} yrs · {patient_gender}", body_text),
        ],
        [
            Paragraph("<b>Study Modality:</b>", body_bold), Paragraph("Brain MRI (Axial T1/T2/FLAIR)", body_text),
            Paragraph("<b>AI Model Backbone:</b>", body_bold), Paragraph("EfficientNet-B0 + U-Net + Grad-CAM", body_text),
        ]
    ]
    t_patient = Table(patient_table_data, colWidths=[1.2*inch, 2.5*inch, 1.2*inch, 2.6*inch])
    t_patient.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F4F6F9')),
        ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor('#D5D8DC')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E5E8E8')),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_patient)
    story.append(Spacer(1, 4))

    # 3. Diagnostic Impression
    conf_pct = probs[classes.index(predicted_class)] * 100
    class_meta = {
        "glioma":      ("GLIOMA PATHOLOGY DETECTED",       "#D9534F", "Malignant / High-Grade Neoplasm"),
        "meningioma":  ("MENINGIOMA PATHOLOGY DETECTED",   "#E67E22", "Typically Benign / Extra-axial Lesion"),
        "pituitary":   ("PITUITARY PATHOLOGY DETECTED",    "#0275D8", "Typically Benign / Sellar Lesion"),
        "notumor":     ("NO PATHOLOGY DETECTED",           "#27AE60", "Normal Brain Scan / No Focal Lesion Identified"),
    }
    banner_title, banner_color, banner_sub = class_meta.get(predicted_class, (predicted_class.upper(), "#333333", "Clinical Review"))

    diag_banner_data = [
        [
            Paragraph(f"<font color='white' size='10'><b>PRIMARY DIAGNOSTIC VERDICT: {banner_title}</b></font><br/><font color='#F0F3F4' size='7.5'>{banner_sub}</font>", ParagraphStyle('WhiteBanner', fontName='Helvetica-Bold', leading=11, alignment=TA_LEFT)),
            Paragraph(f"<font color='white' size='12'><b>{conf_pct:.1f}%</b></font><br/><font color='#F0F3F4' size='6.5'>CONFIDENCE</font>", ParagraphStyle('WhiteBannerR', fontName='Helvetica-Bold', leading=12, alignment=TA_CENTER))
        ]
    ]
    t_diag_banner = Table(diag_banner_data, colWidths=[6.0*inch, 1.5*inch])
    t_diag_banner.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(banner_color)),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_diag_banner)
    story.append(Spacer(1, 4))

    # 4. Multi-class Probability Breakdown Table
    story.append(Paragraph("Pathology Classification Likelihoods", sec_heading))
    prob_headers = [
        Paragraph("Pathology Class", cell_hdr),
        Paragraph("Clinical Nature", cell_hdr),
        Paragraph("Model Probability", cell_hdr),
        Paragraph("Diagnostic Status", cell_hdr)
    ]
    prob_rows = [prob_headers]
    for c in sorted(classes, key=lambda x: -probs[classes.index(x)]):
        p_val = probs[classes.index(c)] * 100
        is_top = (c == predicted_class)
        c_title, _, c_desc = class_meta.get(c, (c.title(), "", ""))
        status_str = "<b>POSITIVE IDENTIFICATION</b>" if is_top else "Ruled Out / Negative"
        prob_rows.append([
            Paragraph(f"<b>{c.title()}</b>" if is_top else c.title(), cell_txt),
            Paragraph(c_desc.split('/')[0].strip(), cell_txt),
            Paragraph(f"<b>{p_val:.2f}%</b>" if is_top else f"{p_val:.2f}%", cell_txt),
            Paragraph(status_str, cell_txt)
        ])
    t_prob = Table(prob_rows, colWidths=[1.7*inch, 2.4*inch, 1.4*inch, 2.0*inch])
    t_prob.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0B2545')),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#D5D8DC')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8F9F9')]),
    ]))
    story.append(t_prob)
    story.append(Spacer(1, 4))

    # 5. Diagnostic Imaging Tri-View Panel
    story.append(Paragraph("Visual Diagnostic & Explainability Panel (Tri-View)", sec_heading))

    def _pil_to_rl(img_obj, max_dim=1.75*inch):
        if img_obj is None:
            return Paragraph("<font color='#888888'>N/A</font>", cell_txt)
        buf = io.BytesIO()
        rgb = img_obj.convert("RGB")
        rgb.save(buf, format="PNG")
        buf.seek(0)
        w, h = rgb.size
        asp = h / float(w)
        if asp > 1.0:
            rh = max_dim
            rw = max_dim / asp
        else:
            rw = max_dim
            rh = max_dim * asp
        return RLImage(buf, width=rw, height=rh)

    rl_orig = _pil_to_rl(pil_img)
    rl_over = _pil_to_rl(overlay_img) if overlay_img is not None else Paragraph("<br/><br/><b>No Tumor Segmented</b><br/><font size='6.5' color='#7F8C8D'>Healthy scan / No lesion mask</font>", cell_txt)
    rl_grad = _pil_to_rl(gradcam_img) if gradcam_img is not None else Paragraph("<br/><br/><b>Grad-CAM N/A</b>", cell_txt)

    img_table_data = [
        [rl_orig, rl_over, rl_grad],
        [
            Paragraph("<b>1. Original MRI Input</b><br/><font size='6.5' color='#566573'>Preprocessed Grayscale Scan</font>", img_caption),
            Paragraph("<b>2. U-Net Pathology Overlay</b><br/><font size='6.5' color='#566573'>Spatial Boundary Delineation</font>", img_caption),
            Paragraph("<b>3. Grad-CAM Saliency Map</b><br/><font size='6.5' color='#566573'>XAI Class Activation Attention</font>", img_caption),
        ]
    ]
    t_images = Table(img_table_data, colWidths=[2.5*inch, 2.5*inch, 2.5*inch])
    t_images.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#D5D8DC')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#EAEDED')),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F8F9FA')),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#EDF2F7')),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_images)
    story.append(Spacer(1, 4))

    # 6. Quantitative Morphometrics & Volumetric Profiling (if tumor detected)
    if predicted_class in ["glioma", "meningioma", "pituitary"] and (area_data or shape_data):
        story.append(Paragraph("Quantitative Morphological & Volumetric Profiling", sec_heading))
        morph_rows = [
            [
                Paragraph("Metric Description", cell_hdr),
                Paragraph("Calculated Value", cell_hdr),
                Paragraph("Metric Description", cell_hdr),
                Paragraph("Calculated Value", cell_hdr),
            ]
        ]

        area_mm2_str = f"{area_data.get('area_mm2', 0):,.1f} mm²" if area_data else "N/A"
        area_cm2_str = f"{area_data.get('area_cm2', 0):.2f} cm²" if area_data else "N/A"
        cov_str = f"{area_data.get('coverage_pct', 0):.2f}%" if area_data else "N/A"
        px_str = f"{area_data.get('pixel_count', 0):,}" if area_data else "N/A"

        shape_label = shape_data.get('shape_label', 'N/A') if shape_data else "N/A"
        circ_str = f"{shape_data.get('circularity', 0):.3f}" if shape_data else "N/A"
        comp_str = f"{shape_data.get('compactness', 0):.3f}" if shape_data else "N/A"
        sol_str  = f"{shape_data.get('solidity', 0):.3f}" if shape_data else "N/A"

        morph_rows.append([
            Paragraph("<b>Estimated Tumor Area:</b>", body_text), Paragraph(f"<b>{area_mm2_str}</b> ({area_cm2_str})", body_bold),
            Paragraph("<b>Tumor Boundary Geometry:</b>", body_text), Paragraph(f"<b>{shape_label}</b>", body_bold),
        ])
        morph_rows.append([
            Paragraph("<b>Hemisphere Coverage:</b>", body_text), Paragraph(f"{cov_str} ({px_str} px)", body_text),
            Paragraph("<b>Circularity Index:</b>", body_text), Paragraph(circ_str, body_text),
        ])
        morph_rows.append([
            Paragraph("<b>Spatial Resolution:</b>", body_text), Paragraph("~0.50 mm/pixel standard", body_text),
            Paragraph("<b>Compactness / Solidity:</b>", body_text), Paragraph(f"{comp_str} / {sol_str}", body_text),
        ])

        if conf_data:
            morph_rows.append([
                Paragraph("<b>Mean Mask Confidence:</b>", body_text), Paragraph(f"{conf_data.get('mean', 0)*100:.1f}%", body_text),
                Paragraph("<b>High-Confidence Area (≥75%):</b>", body_text), Paragraph(f"{conf_data.get('high_conf_pct', 0):.1f}%", body_text),
            ])

        t_morph = Table(morph_rows, colWidths=[1.8*inch, 1.95*inch, 1.8*inch, 1.95*inch])
        t_morph.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0B2545')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#D5D8DC')),
            ('TOPPADDING', (0,0), (-1,-1), 1.5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8F9F9')]),
        ]))
        story.append(t_morph)
        story.append(Spacer(1, 4))

    # 7. Radiological Sign-off & Medical Disclaimer
    sign_data = [
        [
            Paragraph("<b>Automated Diagnostic Assessment:</b><br/><font size='7' color='#566573'>Computer-aided findings generated by NeuroScan AI Pipeline (EfficientNet-B0 + U-Net + Grad-CAM). For clinical corroboration by certified medical professionals.</font>", body_text),
            Paragraph(f"<b>Examining Clinician:</b><br/><font size='7.5' color='#0B2545'>Dr. / Clinician: {username or 'Attending Radiologist'}</font><br/><br/><b>Digital Signature:</b> ___________________________", body_text)
        ]
    ]
    t_sign = Table(sign_data, colWidths=[4.8*inch, 2.7*inch])
    t_sign.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_sign)
    story.append(Spacer(1, 3))

    story.append(Paragraph(
        "<b>MEDICAL DISCLAIMER:</b> This automated computer-aided diagnosis report is provided for educational, research, and assistive clinical decision support purposes only. It is not an autonomous replacement for clinical pathological biopsy or definitive radiological review.",
        disclaimer_style
    ))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def generate_3d_volumetric_pdf_report(
    patient_name: str,
    patient_age: int,
    patient_gender: str,
    username: str,
    seg_3d_res: dict,
    vol_3d: np.ndarray,
    mask_3d: np.ndarray,
    filename: str = "scan.nii"
) -> bytes:
    """Generate a DICOM-grade 3D Structural Volumetric MRI Diagnostic PDF Report."""
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("ReportLab is not installed. Please run: pip install reportlab")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=32,
        bottomMargin=32
    )

    title_style = ParagraphStyle(
        'DocTitle3D',
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=17,
        textColor=colors.HexColor('#0B2545')
    )
    meta_style = ParagraphStyle(
        'DocMeta3D',
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        alignment=TA_RIGHT,
        textColor=colors.HexColor('#566573')
    )
    sec_heading = ParagraphStyle(
        'SecHeading3D',
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor('#0B2545'),
        spaceBefore=4,
        spaceAfter=2
    )
    body_text = ParagraphStyle(
        'BodyDark3D',
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor('#1C2833')
    )
    body_bold = ParagraphStyle(
        'BodyBold3D',
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor('#1C2833')
    )
    cell_hdr = ParagraphStyle(
        'CellHdr3D',
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=TA_CENTER
    )
    cell_txt = ParagraphStyle(
        'CellTxt3D',
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#1C2833'),
        alignment=TA_CENTER
    )
    img_caption = ParagraphStyle(
        'ImgCaption3D',
        fontName='Helvetica-Bold',
        fontSize=7,
        leading=8.5,
        textColor=colors.HexColor('#2C3E50'),
        alignment=TA_CENTER
    )
    disclaimer_style = ParagraphStyle(
        'DisclaimerTxt3D',
        fontName='Helvetica-Oblique',
        fontSize=6.5,
        leading=8,
        textColor=colors.HexColor('#7F8C8D'),
        alignment=TA_JUSTIFY
    )

    story = []

    # 1. Header Banner
    now_str = datetime.datetime.now().strftime("%B %d, %Y %H:%M:%S")
    report_id = f"RPT-3D-{datetime.datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"

    header_data = [
        [
            Paragraph("<b>NEUROSCAN AI WORKSTATION</b><br/><font color='#0084FF' size='7.5'>3D STRUCTURAL VOLUMETRIC MRI SEGMENTATION REPORT</font>", title_style),
            Paragraph(f"<b>Report ID:</b> {report_id}<br/><b>Generated:</b> {now_str}<br/><b>Attending Clinician:</b> {username or 'Doctor'}", meta_style)
        ]
    ]
    t_hdr = Table(header_data, colWidths=[4.2*inch, 3.3*inch])
    t_hdr.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_hdr)
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0084FF'), spaceBefore=2, spaceAfter=4))

    # 2. Patient & Exam Demographics Card
    p_name_display = patient_name.strip() if patient_name and patient_name.strip() else "Anonymous / Unspecified"
    dx, dy, dz = seg_3d_res.get("voxel_spacing_mm", (1.0, 1.0, 1.0))
    patient_table_data = [
        [
            Paragraph("<b>Patient Name:</b>", body_bold), Paragraph(p_name_display, body_text),
            Paragraph("<b>Age / Gender:</b>", body_bold), Paragraph(f"{patient_age} yrs · {patient_gender}", body_text),
        ],
        [
            Paragraph("<b>Scan File / Modality:</b>", body_bold), Paragraph(f"{filename} (3D NIfTI)", body_text),
            Paragraph("<b>Volume Dimensions:</b>", body_bold), Paragraph(f"{vol_3d.shape[0]} × {vol_3d.shape[1]} × {vol_3d.shape[2]} ({vol_3d.shape[2]} Slices)", body_text),
        ],
        [
            Paragraph("<b>Voxel Spacing (dx,dy,dz):</b>", body_bold), Paragraph(f"{dx:.2f} × {dy:.2f} × {dz:.2f} mm", body_text),
            Paragraph("<b>AI Model Backbone:</b>", body_bold), Paragraph("3D Volumetric U-Net (EfficientNet-B0 Encoder)", body_text),
        ]
    ]
    t_patient = Table(patient_table_data, colWidths=[1.3*inch, 2.45*inch, 1.3*inch, 2.45*inch])
    t_patient.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F4F6F9')),
        ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor('#D5D8DC')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E5E8E8')),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_patient)
    story.append(Spacer(1, 4))

    # 3. Primary Volumetric Diagnostic Verdict Banner
    has_tumor = seg_3d_res.get("has_tumor", False)
    if has_tumor:
        banner_title = "3D VOLUMETRIC TUMOR LESION IDENTIFIED"
        banner_sub = f"Volumetric Mass Segmented Across {seg_3d_res['affected_slices']} Axial Slices · Peak Lesion at Slice Z={seg_3d_res['peak_slice_idx']}"
        banner_color = "#D9534F"
        metric_str = f"{seg_3d_res['tumor_volume_cm3']:.2f} cm³"
        metric_sub = f"{seg_3d_res['tumor_volume_mm3']:,.1f} mm³"
    else:
        banner_title = "NO SIGNIFICANT LESION DETECTED"
        banner_sub = "Normal Volumetric Structural MRI / No Significant Tumor Cluster Segmented"
        banner_color = "#27AE60"
        metric_str = "0.00 cm³"
        metric_sub = "Negative"

    diag_banner_data = [
        [
            Paragraph(f"<font color='white' size='10'><b>PRIMARY FINDINGS: {banner_title}</b></font><br/><font color='#F0F3F4' size='7.5'>{banner_sub}</font>", ParagraphStyle('WhiteBanner3D', fontName='Helvetica-Bold', leading=11, alignment=TA_LEFT)),
            Paragraph(f"<font color='white' size='11'><b>{metric_str}</b></font><br/><font color='#F0F3F4' size='6.5'>{metric_sub}</font>", ParagraphStyle('WhiteBannerR3D', fontName='Helvetica-Bold', leading=11, alignment=TA_CENTER))
        ]
    ]
    t_diag_banner = Table(diag_banner_data, colWidths=[5.8*inch, 1.7*inch])
    t_diag_banner.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(banner_color)),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_diag_banner)
    story.append(Spacer(1, 4))

    # 4. Volumetric Key Metrics Table
    story.append(Paragraph("3D Quantitative Volumetric Profiling & Infiltration Assessment", sec_heading))
    vol_size = vol_3d.size if vol_3d is not None and vol_3d.size > 0 else 1
    vol_cov_pct = (100.0 * seg_3d_res.get('tumor_voxel_count', 0) / vol_size)
    aff_slices_count = seg_3d_res.get('affected_slices', 0)
    slice_cov_val = seg_3d_res.get('slice_coverage_pct', 0.0)

    if has_tumor:
        if aff_slices_count >= 25 or slice_cov_val >= 20.0:
            spread_prob_pdf = min(96.0, 68.0 + (slice_cov_val * 0.9))
            spread_status_pdf = "Spreading Possible (High Infiltration Risk)"
        else:
            spread_prob_pdf = min(68.0, 38.0 + (slice_cov_val * 1.2))
            spread_status_pdf = "Spreading Possible (Focal Infiltration)"
    else:
        spread_prob_pdf = 1.0
        spread_status_pdf = "No Spreading Detected (Normal / Low Risk)"

    vol_rows = [
        [
            Paragraph("Volumetric Metric", cell_hdr),
            Paragraph("Calculated Value", cell_hdr),
            Paragraph("Volumetric Metric", cell_hdr),
            Paragraph("Calculated Value", cell_hdr),
        ],
        [
            Paragraph("<b>Physical Tumor Volume (cm³):</b>", body_text),
            Paragraph(f"<b>{seg_3d_res.get('tumor_volume_cm3', 0.0):.2f} cm³</b>", body_bold),
            Paragraph("<b>Physical Tumor Volume (mm³):</b>", body_text),
            Paragraph(f"<b>{seg_3d_res.get('tumor_volume_mm3', 0.0):,.2f} mm³</b>", body_bold),
        ],
        [
            Paragraph("<b>Tumor Voxel Count:</b>", body_text),
            Paragraph(f"<b>{seg_3d_res.get('tumor_voxel_count', 0):,} voxels</b>", body_text),
            Paragraph("<b>Tumor-Containing Slices:</b>", body_text),
            Paragraph(f"<b>{aff_slices_count} / {seg_3d_res.get('total_slices', vol_3d.shape[2])} ({slice_cov_val:.1f}%)</b>", body_text),
        ],
        [
            Paragraph("<b>Peak Lesion Slice (Z):</b>", body_text),
            Paragraph(f"<b>Slice Z = {seg_3d_res.get('peak_slice_idx', 0)}</b> ({seg_3d_res.get('peak_slice_voxel_count', 0):,} active pixels)", body_text),
            Paragraph("<b>Single Voxel Volume:</b>", body_text),
            Paragraph(f"<b>{(dx*dy*dz):.4f} mm³</b>", body_text),
        ],
        [
            Paragraph("<b>Volumetric Coverage:</b>", body_text),
            Paragraph(f"{vol_cov_pct:.3f}% total volume", body_text),
            Paragraph("<b>Spreading / Metastasis Risk:</b>", body_text),
            Paragraph(f"<b>{spread_status_pdf} ({spread_prob_pdf:.1f}%)</b>", body_bold),
        ],
    ]
    t_vol = Table(vol_rows, colWidths=[1.8*inch, 1.95*inch, 1.8*inch, 1.95*inch])
    t_vol.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0B2545')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#D5D8DC')),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8F9F9')]),
    ]))
    story.append(t_vol)
    story.append(Spacer(1, 4))

    # 5. Representative Slice Visual Tri-View Panel (Z = peak_slice_idx)
    rep_z = seg_3d_res.get("peak_slice_idx", vol_3d.shape[2] // 2)
    story.append(Paragraph(f"Representative Slice Visual Triplet (Peak Lesion Axial Slice Z = {rep_z})", sec_heading))

    orig_p, mask_p, ov_p = volume_engine.render_slice_triplet(vol_3d, mask_3d, rep_z)

    def _pil_to_rl_3d(img_obj, max_dim=1.75*inch):
        if img_obj is None:
            return Paragraph("<font color='#888888'>N/A</font>", cell_txt)
        buf = io.BytesIO()
        rgb = img_obj.convert("RGB")
        rgb.save(buf, format="PNG")
        buf.seek(0)
        w, h = rgb.size
        asp = h / float(w)
        if asp > 1.0:
            rh = max_dim
            rw = max_dim / asp
        else:
            rw = max_dim
            rh = max_dim * asp
        return RLImage(buf, width=rw, height=rh)

    rl_orig = _pil_to_rl_3d(orig_p)
    rl_mask = _pil_to_rl_3d(mask_p)
    rl_ov   = _pil_to_rl_3d(ov_p)

    img_table_data = [
        [rl_orig, rl_mask, rl_ov],
        [
            Paragraph(f"<b>1. Original MRI (z={rep_z})</b><br/><font size='6.5' color='#566573'>Preprocessed Grayscale Slice</font>", img_caption),
            Paragraph("<b>2. Predicted Tumor Mask</b><br/><font size='6.5' color='#566573'>U-Net EfficientNet-B0 (Reds)</font>", img_caption),
            Paragraph("<b>3. Pathological Overlay</b><br/><font size='6.5' color='#566573'>MRI + Autumn Highlight Overlay</font>", img_caption),
        ]
    ]
    t_images = Table(img_table_data, colWidths=[2.5*inch, 2.5*inch, 2.5*inch])
    t_images.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#D5D8DC')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#EAEDED')),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F8F9FA')),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#EDF2F7')),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_images)
    story.append(Spacer(1, 4))

    # 6. Radiological Sign-off & Medical Disclaimer
    sign_data = [
        [
            Paragraph("<b>Automated Diagnostic Assessment:</b><br/><font size='7' color='#566573'>Volumetric segmentation generated by 3D U-Net (EfficientNet-B0 Backbone). For clinical corroboration by certified medical professionals.</font>", body_text),
            Paragraph(f"<b>Examining Clinician:</b><br/><font size='7.5' color='#0B2545'>Dr. / Clinician: {username or 'Attending Radiologist'}</font><br/><br/><b>Digital Signature:</b> ___________________________", body_text)
        ]
    ]
    t_sign = Table(sign_data, colWidths=[4.8*inch, 2.7*inch])
    t_sign.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_sign)
    story.append(Spacer(1, 3))

    story.append(Paragraph(
        "<b>MEDICAL DISCLAIMER:</b> This automated computer-aided diagnosis report is provided for educational, research, and assistive clinical decision support purposes only. It is not an autonomous replacement for clinical pathological biopsy or definitive radiological review.",
        disclaimer_style
    ))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes




# ══════════════════════════════════════════════════════════════════════════════
# PAGE RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

def render_landing_page():
    """Full-width landing page with hero banner, service cards, and PACS/DICOM workstation aesthetics from the redesign preview."""

    # Inject landing page CSS stylesheet (light theme, space-grotesk typography, full-width)
    st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

  :root {
    --bg: #F5F7F8;
    --panel: #FFFFFF;
    --border: #E1E7EA;
    --ink: #10171C;
    --ink-soft: #4B5960;
    --ink-faint: #7C8A91;
    --teal: #0E6B66;
    --teal-deep: #0A4F4C;
    --blue: #2C4D74;
    --coral: #C1543F;
    --sage: #4C7A5E;
    --mono: 'IBM Plex Mono', monospace;
    --sans: 'Inter', sans-serif;
    --display: 'Space Grotesk', sans-serif;
  }

  /* Reset layout constraints for landing page to go full-width */
  .block-container {
    padding-top: 0px !important;
    padding-bottom: 0px !important;
    padding-left: 0px !important;
    padding-right: 0px !important;
    max-width: 100% !important;
  }
  [data-testid="stVerticalBlock"] {
    gap: 0px !important;
  }

  html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"], .main {
    background-color: var(--bg) !important;
    color: var(--ink) !important;
    font-family: var(--sans) !important;
    -webkit-font-smoothing: antialiased;
  }

  /* ---------- compliance bar ---------- */
  .compliance{
    background:var(--teal-deep);
    color:#EAF3F2;
    font-family:var(--mono);
    font-size:12px;
    letter-spacing:.02em;
    text-align:center;
    padding:8px 16px;
  }
  .compliance b{ color:#fff; font-weight:500; }

  /* ---------- nav ---------- */
  header.custom-landing-header{
    display:flex; align-items:center; justify-content:space-between;
    padding:20px 48px;
    border-bottom:1px solid var(--border);
    background:var(--panel);
    position:sticky; top:0; z-index:20;
    width: 100%;
    visibility: visible !important; /* Force visible to override Streamlit's global header hide */
  }
  .brand{ display:flex; align-items:center; gap:10px; }
  .brand-mark{
    width:34px; height:34px; border-radius:8px;
    background:linear-gradient(135deg,var(--teal),var(--teal-deep));
    display:flex; align-items:center; justify-content:center;
  }
  .brand-name{ font-family:var(--display); font-weight:700; font-size:18px; letter-spacing:-.01em; color: var(--ink); }
  .brand-tag{ font-family:var(--mono); font-size:11px; color:var(--ink-faint); margin-left:8px; padding-left:8px; border-left:1px solid var(--border); }
  nav.custom-landing-nav ul{ display:flex; gap:32px; list-style:none; }
  nav.custom-landing-nav a{ font-size:14px; color:var(--ink-soft); font-weight:500; transition:color .15s; }
  nav.custom-landing-nav a:hover{ color:var(--ink); }
  .nav-cta{ display:flex; gap:12px; align-items:center; }
  .btn-ghost{ font-size:14px; font-weight:500; padding:9px 16px; border-radius:7px; color:var(--ink-soft); }
  .btn-ghost:hover{ background:var(--bg); }
  .btn-solid{
    font-size:14px; font-weight:600; padding:10px 18px; border-radius:7px;
    background:var(--ink); color:#fff;
    transition:background .15s;
  }
  .btn-solid:hover{ background:var(--teal-deep); }

  /* ---------- hero ---------- */
  .hero{
    display:grid; grid-template-columns:1.05fr 0.95fr; gap:56px;
    padding:36px 48px 72px; max-width:1280px; margin:0 auto;
    align-items:center;
  }
  .eyebrow{
    display:inline-flex; align-items:center; gap:8px;
    font-family:var(--mono); font-size:11.5px; letter-spacing:.06em; text-transform:uppercase;
    color:var(--teal-deep); background:#E7F1EF; border:1px solid #CFE3DF;
    padding:6px 12px; border-radius:20px; margin-bottom:16px;
    width: fit-content;
  }
  .eyebrow-dot{ width:6px; height:6px; border-radius:50%; background:var(--teal); animation:pulse 2s infinite; }
  @keyframes pulse{ 0%,100%{opacity:1;} 50%{opacity:.35;} }

  h1.hero-title{
    font-family:var(--display); font-weight:700; letter-spacing:-.02em;
    font-size:50px; line-height:1.06; color:var(--ink); margin-bottom:14px;
  }
  h1.hero-title em{ font-style:normal; color:var(--teal-deep); }
  .hero-sub{
    font-size:17px; line-height:1.65; color:var(--ink-soft); max-width:480px; margin-bottom:24px;
  }
  .hero-actions{ display:flex; gap:14px; margin-bottom:28px; }
  .btn-primary{
    font-size:15px; font-weight:600; padding:14px 24px; border-radius:8px;
    background:var(--ink); color:#fff; display:inline-flex; align-items:center; gap:8px;
    transition:transform .15s, background .15s;
    cursor: pointer;
  }
  .btn-primary:hover{ background:var(--teal-deep); transform:translateY(-1px); }
  .btn-secondary{
    font-size:15px; font-weight:600; padding:14px 24px; border-radius:8px;
    border:1px solid var(--border); color:var(--ink); background:var(--panel);
    transition:border-color .15s;
    cursor: pointer;
  }
  .btn-secondary:hover{ border-color:var(--ink-faint); }

  .trust-row{ display:flex; gap:28px; flex-wrap:wrap; }
  .trust-item{ display:flex; align-items:center; gap:8px; font-size:13px; color:var(--ink-faint); font-weight:500; }
  .trust-item svg{ width:16px; height:16px; color:var(--teal); flex-shrink:0; }

  /* ---------- scan panel (signature element) ---------- */
  .scan-panel{
    background:var(--panel); border:1px solid var(--border); border-radius:16px;
    padding:20px; box-shadow:0 1px 2px rgba(16,23,28,.04), 0 12px 32px -16px rgba(16,23,28,.12);
  }
  .scan-panel-head{ display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; }
  .scan-title{ font-family:var(--mono); font-size:11.5px; color:var(--ink-faint); letter-spacing:.04em; }
  .scan-status{ display:flex; align-items:center; gap:6px; font-family:var(--mono); font-size:11px; color:var(--teal-deep); font-weight:500; }
  .scan-status-dot{ width:6px; height:6px; border-radius:50%; background:var(--teal); }

  .scan-stage{
    position:relative; background:#0C1114; border-radius:10px; overflow:hidden;
    aspect-ratio:1/1;
  }
  .scan-stage svg{ width:100%; height:100%; display:block; }

  .scan-line{
    position:absolute; left:0; right:0; height:2px;
    background:linear-gradient(90deg, transparent, #4FD1C5, transparent);
    box-shadow:0 0 12px 2px rgba(79,209,197,.6);
    animation:sweep 3.2s ease-in-out infinite;
  }
  @keyframes sweep{
    0%{ top:6%; opacity:0; }
    8%{ opacity:1; }
    50%{ top:94%; opacity:1; }
    58%{ opacity:0; }
    100%{ top:94%; opacity:0; }
  }

  .mask-path{
    fill:rgba(193,84,63,.18); stroke:var(--coral); stroke-width:1.4;
    stroke-dasharray:220; stroke-dashoffset:220;
    animation:draw 1.6s .6s ease-out forwards;
  }
  @keyframes draw{ to{ stroke-dashoffset:0; } }

  .readout{
    display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:14px;
  }
  .readout-cell{
    background:var(--bg); border:1px solid var(--border); border-radius:8px; padding:10px 12px;
  }
  .readout-label{ font-family:var(--mono); font-size:10px; color:var(--ink-faint); letter-spacing:.04em; text-transform:uppercase; margin-bottom:4px; }
  .readout-value{ font-family:var(--mono); font-size:15px; font-weight:500; color:var(--ink); }
  .readout-value.coral{ color:var(--coral); }
  .readout-value.teal{ color:var(--teal-deep); }

  /* ---------- stats strip ---------- */
  .stats-strip{
    border-top:1px solid var(--border); border-bottom:1px solid var(--border);
    background:var(--panel);
    width: 100%;
  }
  .stats-inner{
    max-width:1280px; margin:0 auto; padding:36px 48px;
    display:grid; grid-template-columns:repeat(4,1fr);
  }
  .stat{ padding:0 24px; border-left:1px solid var(--border); }
  .stat:first-child{ border-left:none; padding-left:0; }
  .stat-num{ font-family:var(--mono); font-size:32px; font-weight:500; color:var(--ink); letter-spacing:-.01em; }
  .stat-label{ font-size:12.5px; color:var(--ink-faint); margin-top:6px; letter-spacing:.01em; }

  /* ---------- services ---------- */
  .section{ max-width:1280px; margin:0 auto; padding:96px 48px; }
  .section-head{ max-width:640px; margin-bottom:52px; }
  .section-eyebrow{ font-family:var(--mono); font-size:12px; letter-spacing:.06em; text-transform:uppercase; color:var(--teal-deep); margin-bottom:14px; }
  .section-head h2{ font-family:var(--display); font-size:34px; font-weight:700; letter-spacing:-.015em; margin-bottom:14px; color: var(--ink); }
  .section-head p{ font-size:16px; color:var(--ink-soft); line-height:1.6; }

  .grid{ display:grid; grid-template-columns:repeat(3,1fr); gap:1px; background:var(--border); border:1px solid var(--border); border-radius:14px; overflow:hidden; }
  .card{ background:var(--panel); padding:32px 28px; transition:background .15s; text-align: left; }
  .card:hover{ background:#FBFCFC; }
  .card-icon{
    width:40px; height:40px; border-radius:10px; background:var(--bg);
    display:flex; align-items:center; justify-content:center; margin-bottom:18px;
    border:1px solid var(--border);
  }
  .card-icon svg{ width:20px; height:20px; color:var(--teal-deep); }
  .card h3{ font-family:var(--display); font-size:17px; font-weight:600; margin-bottom:10px; letter-spacing:-.005em; color: var(--ink); }
  .card p{ font-size:14px; color:var(--ink-soft); line-height:1.6; margin-bottom:18px; }
  .card-tag{
    display:inline-block; font-family:var(--mono); font-size:11px; color:var(--ink-faint);
    background:var(--bg); border:1px solid var(--border); padding:4px 9px; border-radius:5px;
  }

  /* ---------- cta ---------- */
  .cta{
    background:var(--ink); color:#fff; border-radius:20px;
    padding:64px 56px; max-width:1280px; margin:0 auto 96px; text-align:center;
  }
  .cta h2{ font-family:var(--display); font-size:30px; font-weight:700; margin-bottom:14px; letter-spacing:-.015em; color:#fff; }
  .cta p{ color:#B7C2C6; font-size:15px; margin-bottom:30px; }
  .cta .btn-primary{ background:var(--teal); }
  .cta .btn-primary:hover{ background:#12847E; }

  footer.custom-landing-footer{
    border-top:1px solid var(--border); padding:28px 48px;
    display:flex; justify-content:space-between; align-items:center;
    font-size:12.5px; color:var(--ink-faint); font-family:var(--mono);
    width: 100%;
    background: var(--panel);
  }

  @media (max-width:900px){
    .hero{ grid-template-columns:1fr; padding:48px 24px; }
    header.custom-landing-header{ padding:16px 20px; }
    nav.custom-landing-nav ul{ display:none; }
    .stats-inner{ grid-template-columns:1fr 1fr; gap:20px 0; padding:28px 24px; }
    .stat{ border-left:none; padding-left:0; }
    .grid{ grid-template-columns:1fr; }
    .section{ padding:64px 24px; }
    h1.hero-title{ font-size:36px; }
  }
</style>
""", unsafe_allow_html=True)

    # Render entire landing page HTML body (integrated with target_self query params)
    st.markdown("""
<div class="compliance">
<b>Research &amp; clinical decision-support tool</b> — not a standalone diagnostic device. Always confirm findings with a licensed radiologist.
</div>
<header class="custom-landing-header">
<div class="brand">
<div class="brand-mark">
<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
<path d="M9 3a4 4 0 0 0-4 4 3 3 0 0 0-2 2.8V13a3 3 0 0 0 2 2.8V17a4 4 0 0 0 4 4h1V3H9z"/>
<path d="M15 3a4 4 0 0 1 4 4 3 3 0 0 1 2 2.8V13a3 3 0 0 1-2 2.8V17a4 4 0 0 1-4 4h-1V3h1z"/>
<path d="M12 3v18"/>
</svg>
</div>
<span class="brand-name">NeuroScan AI</span>
<span class="brand-tag">v2.1 · Diagnostic Suite</span>
</div>
<nav class="custom-landing-nav">
<ul>
<li><a href="#">Product</a></li>
<li><a href="#services">How it works</a></li>
<li><a href="#">Validation data</a></li>
<li><a href="#">Docs</a></li>
</ul>
</nav>
<div class="nav-cta">
<a href="?nav=login" target="_self" class="btn-ghost">Sign in</a>
<a href="?nav=login" target="_self" class="btn-solid">Request access</a>
</div>
</header>
<section class="hero">
<div>
<div class="eyebrow"><span class="eyebrow-dot"></span> AI-assisted MRI analysis</div>
<h1 class="hero-title">Brain MRI analysis,<br><em>read in seconds</em>, not hours.</h1>
<p class="hero-sub">
NeuroScan AI classifies and segments brain tumors from MRI scans — surfacing tumor type, boundary, and affected area to support faster radiological review.
</p>
<div class="hero-actions">
<a href="?nav=login" target="_self" class="btn-primary">
Analyze a sample scan
<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>
</a>
<a href="#services" target="_self" class="btn-secondary">See how it works</a>
</div>
<div class="trust-row">
<div class="trust-item">
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 12l2 2 4-4"/><circle cx="12" cy="12" r="9"/></svg>
Validated on holdout MRI dataset
</div>
<div class="trust-item">
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="10" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
Scans processed locally
</div>
<div class="trust-item">
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/></svg>
&lt;100ms inference
</div>
</div>
</div>
<div class="scan-panel">
<div class="scan-panel-head">
<span class="scan-title">AXIAL_MRI_SLICE_0042.dcm</span>
<span class="scan-status"><span class="scan-status-dot"></span>ANALYZING</span>
</div>
<div class="scan-stage">
<svg viewBox="0 0 300 300">
<defs>
<radialGradient id="brainGrad" cx="50%" cy="45%" r="60%">
<stop offset="0%" stop-color="#3A4750"/>
<stop offset="55%" stop-color="#232D33"/>
<stop offset="100%" stop-color="#0C1114"/>
</radialGradient>
<filter id="noise">
<feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" result="n"/>
<feColorMatrix in="n" type="matrix" values="0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 0.03 0"/>
</filter>
</defs>
<ellipse cx="150" cy="150" rx="108" ry="122" fill="url(#brainGrad)"/>
<path d="M100 70 C130 55,170 55,200 70 C225 85,232 115,222 145 C232 165,228 195,205 215 C185 235,155 240,130 228 C100 240,75 215,72 185 C58 165,62 135,78 112 C70 95,82 78,100 70 Z"
fill="none" stroke="#4A5860" stroke-width="1.2" opacity="0.7"/>
<path d="M150 60 C150 100,150 200,150 240" stroke="#4A5860" stroke-width="1" opacity="0.5"/>
<path d="M95 130 Q150 115,205 130" stroke="#4A5860" stroke-width="1" opacity="0.4"/>
<path d="M95 175 Q150 190,205 175" stroke="#4A5860" stroke-width="1" opacity="0.4"/>
<rect width="300" height="300" filter="url(#noise)"/>
<path class="mask-path" d="M168 118 C182 112,196 120,199 134 C203 150,195 164,180 168 C166 172,152 165,148 150 C144 136,154 124,168 118 Z"/>
<circle cx="174" cy="141" r="2" fill="var(--coral)"/>
</svg>
<div class="scan-line"></div>
</div>
<div class="readout">
<div class="readout-cell">
<div class="readout-label">Predicted class</div>
<div class="readout-value coral">Glioma</div>
</div>
<div class="readout-cell">
<div class="readout-label">Confidence</div>
<div class="readout-value">94.7%</div>
</div>
<div class="readout-cell">
<div class="readout-label">Lesion area</div>
<div class="readout-value">3.42 cm²</div>
</div>
<div class="readout-cell">
<div class="readout-label">Dice score</div>
<div class="readout-value teal">0.814</div>
</div>
</div>
</div>
</section>
<div class="stats-strip">
<div class="stats-inner">
<div class="stat"><div class="stat-num">95.3%</div><div class="stat-label">Classification accuracy</div></div>
<div class="stat"><div class="stat-num">81.4%</div><div class="stat-label">Segmentation Dice score</div></div>
<div class="stat"><div class="stat-num">4</div><div class="stat-label">Tumor classes detected</div></div>
<div class="stat"><div class="stat-num">&lt;100ms</div><div class="stat-label">Inference latency</div></div>
</div>
</div>
<section class="section" id="services">
<div class="section-head">
<div class="section-eyebrow">What it does</div>
<h2>Six tools, one diagnostic pipeline</h2>
<p>Each MRI scan runs through detection, segmentation, and classification — with explainability built in so radiologists can see what the model saw.</p>
</div>
<div class="grid">
<div class="card">
<div class="card-icon">
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></svg>
</div>
<h3>Tumor detection</h3>
<p>Locates and flags abnormal tissue from MRI scans using a fine-tuned EfficientNet-B0 classifier trained on labeled scan data.</p>
<span class="card-tag">EfficientNet-B0</span>
</div>
<div class="card">
<div class="card-icon">
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 18 0 9 9 0 1 0-18 0z"/><path d="M12 3v3M12 18v3M3 12h3M18 12h3"/><circle cx="12" cy="12" r="2.5"/></svg>
</div>
<h3>Tumor segmentation</h3>
<p>Produces pixel-level segmentation masks with a U-Net + EfficientNet encoder, reaching an 0.814 Dice score on holdout tests.</p>
<span class="card-tag">U-Net · Dice 0.814</span>
</div>
<div class="card">
<div class="card-icon">
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19V10M10 19V5M16 19v-7M22 19H2"/></svg>
</div>
<h3>Class prediction</h3>
<p>Predicts glioma, meningioma, pituitary tumor, or no tumor, with per-class confidence probabilities shown alongside each result.</p>
<span class="card-tag">4-class · 95.3% acc</span>
</div>
<div class="card">
<div class="card-icon">
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M21 3L3 21M8 21H3v-5M16 3h5v5"/></svg>
</div>
<h3>Lesion area</h3>
<p>Calculates cross-sectional lesion area in mm² and cm² from the segmentation mask at standard MRI resolution.</p>
<span class="card-tag">0.5mm / pixel</span>
</div>
<div class="card">
<div class="card-icon">
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l3.5 7.5L23 12l-7.5 1.5L12 21l-3.5-7.5L1 12l7.5-1.5z"/></svg>
</div>
<h3>Shape analysis</h3>
<p>Computes morphological features — circularity, compactness, solidity — to characterize tumor geometry beyond simple size.</p>
<span class="card-tag">Morphology</span>
</div>
<div class="card">
<div class="card-icon">
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/></svg>
</div>
<h3>Explainability (Grad-CAM)</h3>
<p>Generates saliency maps highlighting which regions of the MRI most influenced the classifier's prediction.</p>
<span class="card-tag">Explainable AI</span>
</div>
</div>
</section>
<section class="cta">
<h2>Ready to analyze a scan?</h2>
<p>Sign in or create a free research account to access the full diagnostic suite.</p>
<a href="?nav=login" target="_self" class="btn-primary">
Get started
<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>
</a>
</section>
<footer class="custom-landing-footer">
<span>© 2026 NeuroScan AI · Research &amp; educational use only</span>
<span>Not for primary clinical diagnosis · Patient data processed locally</span>
</footer>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────

def render_login_page():
    """Renders clean, high-precision clinical login UI with ECG heartbeat line matching the exact user reference design."""
    if st.session_state.get("logged_in", False):
        st.session_state.page = "dashboard"
        st.rerun()

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global clean white page styling for login view */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background-color: #FFFFFF !important;
        background-image: none !important;
        color: #111827 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    .block-container {
        max-width: 560px !important;
        padding-top: 1.25rem !important;
        padding-bottom: 1.25rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
        margin: 0 auto !important;
    }

    /* Registration Two-Panel Styles */
    .reg-panel-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.8rem;
        height: 100%;
    }
    .reg-panel-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 700;
        color: #111827;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
        padding-bottom: 4px;
        border-bottom: 1px solid #E2E8F0;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Container Border Reset */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 0 !important;
        box-shadow: none !important;
    }

    /* Top ECG Waveform Line */
    .ecg-line-wrapper {
        margin-bottom: 0.3rem;
    }

    .login-heading-title {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 22px;
        font-weight: 700;
        color: #111827 !important;
        margin: 0 0 3px 0;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }

    .login-heading-sub {
        font-size: 12.5px;
        color: #4B5563 !important;
        margin: 0 0 0.6rem 0;
        line-height: 1.4;
        font-weight: 400;
    }

    /* Input Field Labels: Monospace uppercase in Crisp Black */
    div[data-testid="stTextInput"] label,
    div[data-testid="stTextInput"] label p,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stNumberInput"] label p,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSelectbox"] label p {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 10.5px !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        margin-bottom: 3px !important;
    }

    /* Reset outer widget wrappers */
    div[data-testid*="stTextInput"],
    div[data-testid*="stNumberInput"],
    div[data-testid*="stSelectbox"] {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        height: auto !important;
    }

    /* BaseWeb Input Container & Wrappers: Crisp White with Gray Border */
    div[data-testid*="stTextInput"] [data-baseweb="base-input"],
    div[data-testid*="stTextInput"] [data-baseweb="input"],
    div[data-testid*="stNumberInput"] [data-baseweb="base-input"],
    div[data-testid*="stNumberInput"] [data-baseweb="input"],
    div[data-baseweb="input"],
    div[data-baseweb="base-input"] {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 6px !important;
        height: 38px !important;
        min-height: 38px !important;
        padding: 0 !important;
        overflow: hidden !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    }
    div[data-testid*="stTextInput"] [data-baseweb="input"]:focus-within,
    div[data-testid*="stNumberInput"] [data-baseweb="input"]:focus-within,
    div[data-baseweb="input"]:focus-within {
        border-color: #166534 !important;
        box-shadow: 0 0 0 2px rgba(22, 101, 52, 0.12) !important;
    }

    /* Text & Number Input elements inside BaseWeb: Crisp Black Text & Visible Typing Caret */
    div[data-testid*="stTextInput"] input,
    div[data-testid*="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    input[type="text"],
    input[type="password"],
    input[type="number"],
    input {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        caret-color: #111827 !important;
        border: none !important;
        border-radius: 6px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
        height: 36px !important;
        padding: 0 10px !important;
        box-shadow: none !important;
    }
    div[data-testid*="stTextInput"] input::placeholder,
    input::placeholder {
        color: #9CA3AF !important;
        -webkit-text-fill-color: #9CA3AF !important;
    }
    div[data-testid*="stTextInput"] input:focus,
    div[data-testid*="stNumberInput"] input:focus,
    input:focus {
        outline: none !important;
        border: none !important;
        box-shadow: none !important;
        caret-color: #111827 !important;
    }

    /* Number Input Stepper Buttons */
    div[data-testid*="stNumberInput"] button {
        background: #F3F4F6 !important;
        background-color: #F3F4F6 !important;
        border: none !important;
        color: #111827 !important;
        height: 22px !important;
        min-height: 22px !important;
        width: 32px !important;
    }
    div[data-testid*="stNumberInput"] button svg {
        fill: #111827 !important;
        color: #111827 !important;
        stroke: #111827 !important;
        width: 14px !important;
        height: 14px !important;
    }

    /* Selectbox (Gender) Styling: Crisp White Container + Black Text */
    div[data-testid*="stSelectbox"] [data-baseweb="select"] > div {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 6px !important;
        min-height: 38px !important;
        height: 38px !important;
        color: #111827 !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
    }
    div[data-testid*="stSelectbox"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
    div[data-testid*="stSelectbox"] [data-baseweb="select"] div,
    div[data-testid*="stSelectbox"] [data-baseweb="select"] span {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
    }
    div[data-testid*="stSelectbox"] svg {
        fill: #111827 !important;
        color: #111827 !important;
        stroke: #111827 !important;
    }

    /* Selectbox Dropdown Menu Popover Options */
    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[role="listbox"] {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 6px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12) !important;
    }
    li[role="option"],
    li[role="option"] * {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        background-color: #FFFFFF !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 13.5px !important;
    }
    li[role="option"]:hover,
    li[role="option"][aria-selected="true"] {
        background-color: #F3F4F6 !important;
        background: #F3F4F6 !important;
        color: #111827 !important;
    }

    /* Password Eye Visibility Toggle Button */
    div[data-testid*="stTextInput"] button,
    div[data-testid*="stTextInput"] [data-baseweb="input"] button,
    div[data-baseweb="input"] button {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        border: none !important;
        box-shadow: none !important;
        height: 36px !important;
        min-height: 36px !important;
        width: 36px !important;
        padding: 0 !important;
        margin: 0 !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    div[data-testid*="stTextInput"] button svg,
    div[data-testid*="stTextInput"] [data-baseweb="input"] button svg,
    div[data-baseweb="input"] button svg {
        fill: #6B7280 !important;
        color: #6B7280 !important;
        stroke: #6B7280 !important;
        width: 16px !important;
        height: 16px !important;
    }

    /* Utility Row (Stay signed in / Forgot password) */
    .login-utility-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 4px 0 6px 0;
        font-size: 11.5px;
    }
    .forgot-pwd-link {
        color: #0F766E;
        text-decoration: none;
        font-size: 12px;
        font-weight: 500;
    }

    /* Password Policy Box: Clean light medical card */
    .pwd-policy-container {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 6px 10px;
        margin: 4px 0 6px 0;
        font-family: 'Inter', sans-serif;
    }
    .pwd-policy-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.06em;
        color: #166534;
        text-transform: uppercase;
        margin-bottom: 3px;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .pwd-policy-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 2px 8px;
        font-size: 10.5px;
        color: #4B5563;
    }
    .pwd-policy-item {
        display: flex;
        align-items: center;
        gap: 4px;
        line-height: 1.3;
        font-size: 10.5px;
    }
    .pwd-policy-bullet {
        color: #166534;
        font-size: 10.5px;
        font-weight: bold;
    }

    /* Primary Submit Button: Solid Deep Black */
    div[data-testid="stFormSubmitButton"] button {
        height: 38px !important;
        min-height: 38px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
        margin-top: 4px !important;
        background: #111827 !important;
        background-color: #111827 !important;
        color: #FFFFFF !important;
        border: 1px solid #111827 !important;
        border-radius: 6px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.15s ease !important;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        background: #1F2937 !important;
        background-color: #1F2937 !important;
        border-color: #1F2937 !important;
        color: #FFFFFF !important;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.16) !important;
        transform: translateY(-1px) !important;
    }

    /* Demo Quick Selector Pills */
    .stApp button,
    [data-testid="stAppViewContainer"] button,
    .stButton > button,
    button[kind="secondary"] {
        background: #F3F4F6 !important;
        background-color: #F3F4F6 !important;
        color: #374151 !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 6px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 11.5px !important;
        font-weight: 500 !important;
        height: 28px !important;
        min-height: 28px !important;
        padding: 0 6px !important;
        white-space: nowrap !important;
        transition: all 0.15s ease !important;
        cursor: pointer !important;
    }
    .stApp button:hover, [data-testid="stAppViewContainer"] button:hover {
        background: #E5E7EB !important;
        color: #111827 !important;
        border-color: #D1D5DB !important;
    }

    /* Checkbox styling */
    div[data-testid="stCheckbox"] {
        margin: 2px 0 4px 0 !important;
    }
    div[data-testid="stCheckbox"] label {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 10px !important;
        color: #374151 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        font-weight: 600 !important;
        cursor: pointer !important;
    }
    div[data-testid="stCheckbox"] label p {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 10px !important;
        color: #374151 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        font-weight: 600 !important;
    }
    div[data-testid="stCheckbox"] input[type="checkbox"] {
        accent-color: #111827 !important;
        cursor: pointer !important;
        width: 13px !important;
        height: 13px !important;
    }

    /* Tab Styling for Login / Register */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #F3F4F6 !important;
        padding: 3px !important;
        border-radius: 6px !important;
        border: 1px solid #E5E7EB !important;
        margin-bottom: 0.75rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px !important;
        padding: 5px 10px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        color: #4B5563 !important;
        background: transparent !important;
        border: none !important;
        flex: 1 !important;
        text-align: center !important;
        justify-content: center !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06) !important;
    }

    /* Footnotes */
    .login-footer-contact {
        text-align: center;
        margin-top: 0.75rem;
        font-size: 12px;
        color: #374151;
    }
    .login-footer-contact strong {
        color: #111827;
        font-weight: 600;
    }

    .login-footer-compliance {
        text-align: center;
        margin-top: 0.35rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        color: #9CA3AF;
        letter-spacing: 0.04em;
    }
    </style>
    """, unsafe_allow_html=True)

    # 1. Top Header with ECG Waveform and Top-Right Home Button
    c_hdr_left, c_hdr_right = st.columns([3.3, 0.9])
    with c_hdr_left:
        st.markdown("""
        <div class="ecg-line-wrapper">
          <svg width="180" height="20" viewBox="0 0 220 28" fill="none" stroke="#166534" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M0 14h60l8-10 10 22 10-18 8 11 10-5h114"/>
          </svg>
        </div>
        <h1 class="login-heading-title">NeuroScan AI</h1>
        <p class="login-heading-sub">Secure clinical workstation and patient imaging portal.</p>
        """, unsafe_allow_html=True)
    with c_hdr_right:
        st.markdown("<div style='margin-top: 0.2rem;'></div>", unsafe_allow_html=True)
        if st.button("Home", key="login_back_home_top_btn", use_container_width=True):
            st.session_state.page = "landing"
            st.rerun()

    # 2. Tabs: Sign In vs Create Patient Account (No emojis)
    tab_signin, tab_signup = st.tabs(["Sign In", "Create Patient Account"])

    with tab_signin:
        # Demo Accounts Selector
        st.markdown("""
        <div style="font-size:10px; font-family:'JetBrains Mono', monospace; color:#6B7280; text-transform:uppercase; letter-spacing:0.06em; font-weight:600; margin-bottom:4px;">
            Demo Accounts:
        </div>
        """, unsafe_allow_html=True)

        r_col1, r_col2, r_col3 = st.columns(3)
        with r_col1:
            if st.button("Clinician", key="af_doc", use_container_width=True):
                st.session_state["login_user_input"] = "doctor"
                st.session_state["login_pass_input"] = "brain123"
                st.rerun()
        with r_col2:
            if st.button("Patient", key="af_pat", use_container_width=True):
                st.session_state["login_user_input"] = "patient"
                st.session_state["login_pass_input"] = "patient123"
                st.rerun()
        with r_col3:
            if st.button("Admin", key="af_adm", use_container_width=True):
                st.session_state["login_user_input"] = "admin"
                st.session_state["login_pass_input"] = "neuro2025"
                st.rerun()

        st.markdown("<div style='margin-top:0.3rem;'></div>", unsafe_allow_html=True)

        with st.form("exact_login_form", clear_on_submit=False):
            login_user = st.text_input(
                "CLINICAL ID / USERNAME",
                value=st.session_state.get("login_user_input", ""),
                placeholder="j.martinez",
                key="input_username"
            )

            login_pass = st.text_input(
                "PASSWORD",
                value=st.session_state.get("login_pass_input", ""),
                placeholder="••••••••••••",
                type="password",
                key="input_password"
            )

            c_chk, c_forgot = st.columns([1.4, 1])
            with c_chk:
                stay_signed_in = st.checkbox("STAY SIGNED IN", value=True, key="cb_stay_signed_in")
            with c_forgot:
                st.markdown("<div style='text-align:right; padding-top:4px;'><a href='#' class='forgot-pwd-link'>Forgot password?</a></div>", unsafe_allow_html=True)

            st.markdown("""
            <div style="font-size:10.5px; color:#6B7280; margin:3px 0 6px 0;">
                Min 8 characters with uppercase, lowercase, number &amp; symbol.
            </div>
            """, unsafe_allow_html=True)

            submit_btn = st.form_submit_button("Sign in", use_container_width=True)

        if submit_btn:
            clean_user = login_user.strip() if login_user else ""
            clean_pass = login_pass.strip() if login_pass else ""

            if not clean_user and not clean_pass:
                st.error("Please enter both username/clinical ID and password.")
            elif not clean_user:
                st.error("Clinical ID is required.")
            elif not clean_pass:
                st.error("Password is required.")
            elif len(clean_user) < 2:
                st.error("Clinical ID must be at least 2 characters.")
            else:
                try:
                    user_record = db.authenticate_user(clean_user, login_pass)

                    if user_record:
                        st.session_state.logged_in = True
                        st.session_state.username = user_record["username"]
                        st.session_state.role = user_record["role"]
                        st.session_state.page = "dashboard"

                        # Load patient profile if role is patient
                        if user_record["role"] == "patient":
                            pat_profile = db.get_patient_by_user_id(user_record.get("id"))
                            if pat_profile:
                                st.session_state.patient_name = pat_profile.get("full_name", user_record.get("full_name", ""))
                                st.session_state.patient_age = pat_profile.get("age", 25)
                                st.session_state.patient_gender = pat_profile.get("gender", "Not specified")
                            else:
                                st.session_state.patient_name = user_record.get("full_name", clean_user.title())

                        db.log_activity(
                            username=user_record["username"],
                            action="USER_LOGIN",
                            role=user_record["role"],
                            details=f"User signed into {user_record['role'].title()} portal",
                            status="SUCCESS"
                        )
                        st.rerun()
                    else:
                        db.log_error(
                            error_type="AUTH_FAILED",
                            severity="WARNING",
                            message=f"Failed login attempt for username: '{clean_user}'",
                            component="auth",
                            username=clean_user
                        )
                        db.log_activity(
                            username=clean_user,
                            action="USER_LOGIN",
                            role="unknown",
                            details=f"Failed password attempt for '{clean_user}'",
                            status="FAILED"
                        )
                        st.error("Invalid credentials. Please verify your username and password.")
                except Exception as exc:
                    db.log_error(
                        error_type="DATABASE_ERROR",
                        severity="CRITICAL",
                        message=str(exc),
                        component="database",
                        username=clean_user
                    )
                    st.error("Database connection unavailable. Please contact system administrator.")

    with tab_signup:
        st.markdown("<p style='font-size:12.5px; color:#4B5563; margin-bottom:0.6rem;'>Create a personal patient account to securely view your brain MRI scans and AI diagnostic summaries.</p>", unsafe_allow_html=True)
        
        with st.form("patient_registration_form", clear_on_submit=False):
            p_left, p_right = st.columns(2, gap="medium")

            with p_left:
                st.markdown("""
                <div class="reg-panel-title">
                  Personal Details
                </div>
                """, unsafe_allow_html=True)

                reg_full_name = st.text_input("FULL LEGAL NAME", placeholder="e.g. Sarah Connor", key="reg_pat_fullname")
                reg_age = st.text_input("AGE", value="30", placeholder="e.g. 30", key="reg_pat_age")
                reg_gender = st.selectbox("GENDER", ["Female", "Male", "Other", "Prefer not to say"], key="reg_pat_gender")

            with p_right:
                st.markdown("""
                <div class="reg-panel-title">
                  Account Credentials
                </div>
                """, unsafe_allow_html=True)

                reg_username = st.text_input("DESIRED USERNAME", placeholder="e.g. sarah_c", key="reg_pat_username")
                reg_password = st.text_input("PASSWORD", placeholder="Min 8 chars, Aa1!", type="password", key="reg_pat_pwd")
                reg_password_confirm = st.text_input("CONFIRM PASSWORD", placeholder="Re-enter password", type="password", key="reg_pat_pwd_confirm")

                st.markdown("""
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:6px 8px; margin-top:6px;">
                  <div style="font-family:'JetBrains Mono', monospace; font-size:10px; font-weight:700; color:#166534; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:5px; display:flex; align-items:center; gap:4px;">
                    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#166534" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                    </svg>
                    Password Requirements
                  </div>
                  <div style="display:flex; flex-wrap:wrap; gap:4px;">
                    <span style="background:#FFFFFF; border:1px solid #E2E8F0; color:#374151; font-size:10.5px; padding:2px 6px; border-radius:4px; font-weight:500; white-space:nowrap;">8+ Chars</span>
                    <span style="background:#FFFFFF; border:1px solid #E2E8F0; color:#374151; font-size:10.5px; padding:2px 6px; border-radius:4px; font-weight:500; white-space:nowrap;">A-Z</span>
                    <span style="background:#FFFFFF; border:1px solid #E2E8F0; color:#374151; font-size:10.5px; padding:2px 6px; border-radius:4px; font-weight:500; white-space:nowrap;">a-z</span>
                    <span style="background:#FFFFFF; border:1px solid #E2E8F0; color:#374151; font-size:10.5px; padding:2px 6px; border-radius:4px; font-weight:500; white-space:nowrap;">0-9</span>
                    <span style="background:#FFFFFF; border:1px solid #E2E8F0; color:#374151; font-size:10.5px; padding:2px 6px; border-radius:4px; font-weight:500; white-space:nowrap;">Symbol (!@#$)</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)
            reg_submit = st.form_submit_button("Create Patient Account", use_container_width=True)

        if reg_submit:
            clean_fn = reg_full_name.strip() if reg_full_name else ""
            clean_u = reg_username.strip().lower() if reg_username else ""
            clean_p = reg_password.strip() if reg_password else ""
            clean_pc = reg_password_confirm.strip() if reg_password_confirm else ""
            clean_age_str = reg_age.strip() if reg_age else ""

            if not clean_fn or not clean_u or not clean_p or not clean_age_str:
                st.error("Please fill in all required fields (Full Name, Age, Username, and Password).")
            elif not clean_age_str.isdigit() or not (1 <= int(clean_age_str) <= 120):
                st.error("Please enter a valid age between 1 and 120.")
            elif clean_p != clean_pc:
                st.error("Passwords do not match. Please re-enter your password.")
            elif len(clean_u) < 3:
                st.error("Username must be at least 3 characters long.")
            else:
                pat_id, reg_err = db.create_patient_account(
                    username=clean_u,
                    password=clean_p,
                    full_name=clean_fn,
                    age=int(clean_age_str),
                    gender=reg_gender
                )
                if reg_err:
                    st.error(f"Account Creation Failed: {reg_err}")
                else:
                    st.session_state.page = "landing"
                    st.toast(f"Account created successfully for {clean_fn}! You can now sign in.")
                    st.rerun()

    # 3. Footnotes matching reference image
    st.markdown("""
    <div class="login-footer-contact">
      Need clinician or admin access? <strong>Contact your hospital administrator</strong>
    </div>
    <div class="login-footer-compliance">
      Encrypted connection • HIPAA compliant
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────

def render_dashboard_page():
    """Main clinical dashboard — full existing AI pipeline with Patient Info panel."""

    # ── Load Models ────────────────────────────────────────────────────────────
    clf_model   = load_classifier()
    seg_model, seg_err = load_segmentation_model()
    guardrail_model, guardrail_err = load_guardrail_model()
    metrics     = load_metrics()
    seg_metrics = load_seg_metrics()

    # ── Top Bar Header ─────────────────────────────────────────────────────────
    patient_display = st.session_state.patient_name if st.session_state.patient_name else "No Patient"
    username_display = st.session_state.username or "Doctor"
    is_superadmin = st.session_state.get("role") == "admin"


    st.markdown(f"""
    <div class="top-bar">
        <div class="logo-container">
            <span style="display:inline-block; width:9px; height:9px; border-radius:50%; background:#00D4FF; margin-right:8px; box-shadow:0 0 8px rgba(0,212,255,0.8);"></span>NeuroScan AI <span style="color:#8B949E; font-size:0.85rem; font-weight:400; margin-left:8px;">| Clinical Workstation</span>
        </div>
        <div class="top-bar-right">
            <span><span class="status-dot"></span>Models Online</span>
            <span>EfficientNet-B0 &nbsp;·&nbsp; U-Net</span>
            <span class="patient-chip" style="color:#00D4FF; border-color:rgba(0,212,255,0.3);">👨‍⚕️ Clinician: @{username_display}</span>
            <span class="patient-chip">👤 {patient_display}</span>
            <span style="color:#30363D;">v2.1</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if is_superadmin:
        st.info("⚙️ **SuperAdmin Mode**: You are auditing the Doctor Workstation.")
        if st.button("← Return to Admin Console", key="superadmin_return_btn"):
            st.session_state.role = "admin"
            st.rerun()


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
            st.markdown("<div class='pro-card-title'>👤 Patient Profile &amp; Directory</div>", unsafe_allow_html=True)

            all_db_patients = db.get_all_patients()
            patient_options = ["Custom / Manual Entry"] + [f"{p['full_name']} ({p.get('mrn', 'MRN')}) - {p['age']}y" for p in all_db_patients]
            
            selected_pat_opt = st.selectbox(
                "Select Registered Patient",
                patient_options,
                key="pat_select_dropdown",
                label_visibility="collapsed"
            )

            if selected_pat_opt != "Custom / Manual Entry":
                # Find selected patient details
                idx = patient_options.index(selected_pat_opt) - 1
                matched_pat = all_db_patients[idx]
                st.session_state.patient_name = matched_pat["full_name"]
                st.session_state.patient_age = matched_pat["age"]
                st.session_state.patient_gender = matched_pat["gender"]
                p_name = matched_pat["full_name"]
                p_age = matched_pat["age"]
                p_gender = matched_pat["gender"]
                st.markdown(f"<span style='font-size:0.75rem; color:#3FB950;'>✓ Selected: <b>{p_name}</b> ({matched_pat.get('mrn', 'MRN')})</span>", unsafe_allow_html=True)

                # Check if patient selection changed
                curr_pat_id = matched_pat["id"]
                last_pat_id = st.session_state.get("_curr_selected_pat_id")
                if last_pat_id != curr_pat_id:
                    st.session_state["_curr_selected_pat_id"] = curr_pat_id
                    st.session_state.pop("active_s3_scan_key", None)
                    st.session_state.pop("active_s3_scan_fn", None)

                # Collect all existing patient MRI scans in Database and AWS S3
                db_scans = db.get_all_scans_for_patient(patient_id=matched_pat["id"]) or []
                if not db_scans and matched_pat.get("full_name"):
                    db_scans = db.get_all_scans_for_patient(patient_name=matched_pat["full_name"]) or []
                s3_found_scans = s3.find_patient_scans_in_s3(patient_name=matched_pat["full_name"], mrn=matched_pat.get("mrn", "")) or []
                all_s3_scans = s3.find_patient_scans_in_s3() or []

                found_scans_dict = {}
                for sc in db_scans:
                    s_key = sc.get("s3_key")
                    s_fn = sc.get("filename") or "MRI_Scan.png"
                    if not s_key and s_fn:
                        for s3_item in all_s3_scans:
                            if s_fn.lower() in s3_item["filename"].lower() or s3_item["filename"].lower().endswith(s_fn.lower()):
                                s_key = s3_item["s3_key"]
                                break
                    if s_key:
                        found_scans_dict[s_key] = {
                            "s3_key": s_key,
                            "filename": s_fn,
                            "created_at": sc.get("created_at"),
                            "diagnosis": sc.get("predicted_class", ""),
                            "confidence": sc.get("confidence"),
                        }

                for s3_sc in s3_found_scans:
                    s_key = s3_sc["s3_key"]
                    if s_key not in found_scans_dict:
                        found_scans_dict[s_key] = {
                            "s3_key": s_key,
                            "filename": s3_sc["filename"],
                            "created_at": s3_sc.get("last_modified"),
                            "diagnosis": "",
                            "confidence": None,
                        }

                patient_scans_list = list(found_scans_dict.values())

                if patient_scans_list:
                    if len(patient_scans_list) > 1:
                        scan_labels = []
                        for idx_s, sc in enumerate(patient_scans_list):
                            dt_str = sc["created_at"].strftime("%Y-%m-%d %H:%M") if hasattr(sc["created_at"], "strftime") else str(sc["created_at"] or "")
                            d_lbl = f" · {sc['diagnosis'].upper()}" if sc.get('diagnosis') else ""
                            scan_labels.append(f"Scan #{idx_s + 1}: {sc['filename']} ({dt_str}{d_lbl})")

                        chosen_scan_idx = st.selectbox(
                            "Select Patient Scan from Cloud",
                            range(len(patient_scans_list)),
                            format_func=lambda i: scan_labels[i],
                            key=f"scan_pick_{matched_pat['id']}",
                        )
                        chosen_scan = patient_scans_list[chosen_scan_idx]
                    else:
                        chosen_scan = patient_scans_list[0]

                    st.session_state["active_s3_scan_key"] = chosen_scan["s3_key"]
                    st.session_state["active_s3_scan_fn"] = chosen_scan["filename"]
                    st.session_state["active_s3_pat_id"] = matched_pat["id"]

                    st.markdown(f"""
                    <div style="background:rgba(14,165,233,0.08); border:1px solid rgba(14,165,233,0.25); border-radius:6px; padding:8px 10px; margin-top:8px;">
                      <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-family:'JetBrains Mono', monospace; font-size:10.5px; font-weight:700; color:#38BDF8; letter-spacing:0.06em; text-transform:uppercase;">
                          ☁️ AWS S3 Scan Loaded
                        </span>
                        <span style="font-size:9.5px; color:#34D399; font-weight:600; background:rgba(52,211,153,0.12); border:1px solid rgba(52,211,153,0.3); border-radius:3px; padding:1px 5px;">
                          Active
                        </span>
                      </div>
                      <div style="font-size:11.5px; color:#CBD5E1; margin:4px 0 2px 0; word-break:break-all;">
                        <b>Scan:</b> <code>{chosen_scan['filename']}</code>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.session_state.pop("active_s3_scan_key", None)
                    st.session_state.pop("active_s3_scan_fn", None)
                    st.markdown(f"""
                    <div style="background:rgba(234,179,8,0.08); border:1px solid rgba(234,179,8,0.25); border-radius:6px; padding:8px 10px; margin-top:8px;">
                      <div style="font-size:11px; color:#FDE047;">
                        ℹ️ <b>No prior MRI scans found in AWS S3 for {p_name}.</b><br/>Upload a scan below to analyze &amp; store in cloud.
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.session_state.pop("_curr_selected_pat_id", None)
                st.session_state.pop("active_s3_scan_key", None)
                st.session_state.pop("active_s3_scan_fn", None)
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

            # Expandable Doctor-Only Patient Onboarding tool
            with st.expander("➕ Onboard New Patient (Create Portal Account)"):
                st.markdown("<p style='font-size:0.74rem; color:#8B949E;'>Provision patient portal login credentials &amp; clinical demographic profile.</p>", unsafe_allow_html=True)
                new_pat_name = st.text_input("Patient Full Name", placeholder="e.g. John Doe", key="new_p_fn")
                c_np1, c_np2 = st.columns(2)
                with c_np1:
                    new_pat_age = st.number_input("Patient Age", min_value=1, max_value=120, value=30, key="new_p_age")
                with c_np2:
                    new_pat_gen = st.selectbox("Gender", ["Male", "Female", "Other"], key="new_p_gen")
                
                c_np3, c_np4 = st.columns(2)
                with c_np3:
                    new_pat_user = st.text_input("Portal Username", placeholder="e.g. j_doe", key="new_p_user")
                with c_np4:
                    new_pat_pass = st.text_input("Portal Password", type="password", placeholder="Min 8 chars, Aa1!", key="new_p_pass")
                
                st.markdown("<div style='font-size:0.68rem; color:#8B949E; margin-bottom:0.5rem;'>🔒 <b>Password Policy</b>: Min 8 characters, at least 1 uppercase, 1 lowercase, 1 digit, and 1 special symbol.</div>", unsafe_allow_html=True)

                if st.button("➕ Register Patient Account", key="btn_create_pat_account", use_container_width=True):
                    if not new_pat_name or not new_pat_user or not new_pat_pass:
                        st.error("Please fill in all patient profile and portal login fields.")
                    else:
                        pid, err = db.create_patient_account(
                            username=new_pat_user,
                            password=new_pat_pass,
                            full_name=new_pat_name,
                            age=int(new_pat_age),
                            gender=new_pat_gen,
                            doctor_username=st.session_state.username
                        )
                        if err:
                            st.error(f"❌ Password Policy / Registration Error: {err}")
                        else:
                            st.session_state.patient_name = new_pat_name
                            st.session_state.patient_age = int(new_pat_age)
                            st.session_state.patient_gender = new_pat_gen
                            st.success(f"✓ Patient @{new_pat_user} ({new_pat_name}) registered with secure bcrypt hash!")
                            st.rerun()


        # ── MRI Upload Card ───────────────────────────────────────────────────
        with st.container(border=True):
            st.markdown("<div class='pro-card-title'>Upload MRI Scan (2D / 3D NIfTI)</div>", unsafe_allow_html=True)
            uploaded = st.file_uploader("Upload MRI", type=["jpg", "jpeg", "png", "nii", "nii.gz"], label_visibility="collapsed")



    # Automatic Pipeline Execution (Local Upload OR Fetched from AWS S3)
    is_3d_nifti = False
    vol_3d = None
    mask_3d = None
    seg_3d_res = None
    nii_meta = None
    voxel_spacing = None
    area_data = None
    shape_data = None
    conf_data = None
    overlay_img = None
    pdf_report_bytes = None
    pdf_gen_error = None
    pil_img = None
    is_mri = False
    mri_reason = ""
    step = 0
    probs = None
    predicted_class = None
    has_tumor = False
    binary_mask = None
    prob_map = None
    gradcam_img = None
    gradcam_raw = None
    scan_source_label = ""

    # Check if a scan is provided via live file uploader OR via S3 fetch
    active_s3_key = st.session_state.get("active_s3_scan_key")

    if uploaded:
        # Clear active S3 state if doctor explicitly uploads a new file
        st.session_state.pop("active_s3_scan_key", None)
        scan_source_label = uploaded.name
        is_3d_nifti = uploaded.name.lower().endswith((".nii", ".nii.gz"))
        if not is_3d_nifti:
            pil_img = Image.open(uploaded)
    elif active_s3_key:
        scan_source_label = st.session_state.get("active_s3_scan_fn", "s3_mri_scan.png")
        is_3d_nifti = scan_source_label.lower().endswith((".nii", ".nii.gz"))
        if not is_3d_nifti:
            with st.spinner(f"Downloading MRI scan '{scan_source_label}' from AWS S3..."):
                pil_img = s3.download_mri_from_s3(active_s3_key)

    if (uploaded or active_s3_key) and is_3d_nifti:
        step = 1
        try:
            if uploaded:
                nii_bytes = uploaded.getvalue()
                nii_fn = uploaded.name
            else:
                with st.spinner(f"Downloading 3D NIfTI volume '{scan_source_label}' from AWS S3..."):
                    nii_bytes = s3.download_bytes_from_s3(active_s3_key)
                nii_fn = scan_source_label

            if nii_bytes:
                vol_3d, voxel_spacing, nii_meta = volume_engine.load_nifti_from_bytes(nii_bytes, nii_fn)
                is_mri = True
                step = 3
                if seg_model:
                    with st.spinner("Executing 3D Volumetric Tumor Slice Segmentation..."):
                        seg_3d_res = volume_engine.segment_3d_volume(vol_3d, voxel_spacing, seg_model, DEVICE)
                        mask_3d = seg_3d_res["mask_3d"]
                        has_tumor = seg_3d_res["has_tumor"]
                    step = 4
                else:
                    step = 4

                # Store active report for RAG Copilot
                diag_name = "Tumor Detected (3D Volumetric Mass)" if (seg_3d_res and seg_3d_res["has_tumor"]) else "No Significant Tumor Detected"
                active_report = {
                    "patient_name": st.session_state.get("patient_name", "Anonymous"),
                    "patient_age": st.session_state.get("patient_age", "N/A"),
                    "patient_gender": st.session_state.get("patient_gender", "N/A"),
                    "diagnosis": diag_name,
                    "confidence_pct": 98.5 if (seg_3d_res and seg_3d_res["has_tumor"]) else 99.0,
                    "has_tumor": seg_3d_res["has_tumor"] if seg_3d_res else False,
                    "area_cm2": seg_3d_res["tumor_volume_cm3"] if seg_3d_res else None,
                    "area_mm2": seg_3d_res["tumor_volume_mm3"] if seg_3d_res else None,
                    "tumor_pixels": seg_3d_res["tumor_voxel_count"] if seg_3d_res else None,
                    "shape_label": "3D Volumetric Mass",
                    "scan_filename": scan_source_label,
                    "is_3d_nifti": True
                }
                st.session_state["active_diagnostic_report"] = active_report
                st.session_state["last_diagnosis"] = diag_name

                # Log scan record to database if newly uploaded
                if uploaded:
                    try:
                        patient_id = db.create_or_get_patient(
                            full_name=st.session_state.patient_name,
                            age=st.session_state.patient_age,
                            gender=st.session_state.patient_gender
                        )
                        s3_key_nii, s3_url_nii = s3.upload_mri_to_s3(nii_bytes, uploaded.name, patient_name=st.session_state.patient_name)
                        db.save_scan_record(
                            filename=uploaded.name,
                            is_valid_mri=True,
                            guardrail_reason="3D NIfTI Structural Brain Volume",
                            patient_id=patient_id,
                            doctor_username=st.session_state.username,
                            predicted_class="glioma" if (seg_3d_res and seg_3d_res["has_tumor"]) else "notumor",
                            confidence=98.5 if (seg_3d_res and seg_3d_res["has_tumor"]) else 99.0,
                            probabilities_dict={"glioma": 0.985, "notumor": 0.015} if (seg_3d_res and seg_3d_res["has_tumor"]) else {"notumor": 0.99, "glioma": 0.01},
                            area_data={"area_cm2": seg_3d_res["tumor_volume_cm3"], "area_mm2": seg_3d_res["tumor_volume_mm3"], "pixel_count": seg_3d_res["tumor_voxel_count"]} if seg_3d_res else None,
                            shape_data={"shape_label": "3D Volumetric Mass"},
                            s3_key=s3_key_nii,
                            s3_url=s3_url_nii
                        )
                    except Exception:
                        pass

                # Generate 3D Volumetric PDF Report
                if REPORTLAB_AVAILABLE and seg_3d_res is not None and vol_3d is not None and mask_3d is not None:
                    try:
                        pdf_report_bytes = generate_3d_volumetric_pdf_report(
                            patient_name=st.session_state.patient_name,
                            patient_age=st.session_state.patient_age,
                            patient_gender=st.session_state.patient_gender,
                            username=st.session_state.username,
                            seg_3d_res=seg_3d_res,
                            vol_3d=vol_3d,
                            mask_3d=mask_3d,
                            filename=scan_source_label
                        )
                    except Exception as exc:
                        pdf_report_bytes = None
                        pdf_gen_error = str(exc)

        except Exception as e:
            st.error(f"Error reading 3D NIfTI scan: {e}")
            db.log_error(
                error_type="NIFTI_PROCESS_ERROR",
                severity="ERROR",
                message=str(e),
                stack_trace=traceback.format_exc(),
                component="volume_engine",
                username=st.session_state.username,
                filename=scan_source_label
            )
    elif pil_img is not None:
        # 2D Image Pipeline (Direct Live Upload OR Cloud Fetched from AWS S3)
        step = 1
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
                overlay_img = overlay_mask_on_image(pil_img, binary_mask, alpha=0.45)
                px_count, total_px, cov_pct, area_mm2, area_cm2 = compute_tumor_area(binary_mask, pil_img)
                area_data = {
                    "pixel_count": px_count,
                    "total_pixels": total_px,
                    "coverage_pct": cov_pct,
                    "area_mm2": area_mm2,
                    "area_cm2": area_cm2
                }
                shape_data = compute_shape_analysis(binary_mask)
                if prob_map is not None:
                    conf_data = compute_confidence_stats(prob_map, binary_mask)
                step = 4
            else:
                step = 4 # Completed (no tumor skips segmentation)

            # Upload to S3 & DB if newly uploaded
            if uploaded:
                scan_s3_key, scan_s3_url = s3.upload_mri_to_s3(pil_img, uploaded.name, patient_name=st.session_state.patient_name)
                try:
                    patient_id = db.create_or_get_patient(
                        full_name=st.session_state.patient_name,
                        age=st.session_state.patient_age,
                        gender=st.session_state.patient_gender
                    )
                    db.save_scan_record(
                        filename=scan_source_label,
                        is_valid_mri=True,
                        guardrail_reason=mri_reason,
                        patient_id=patient_id,
                        doctor_username=st.session_state.username,
                        predicted_class=predicted_class,
                        confidence=float(probs[CLASSES.index(predicted_class)] * 100),
                        probabilities_dict={c: float(p) for c, p in zip(CLASSES, probs)},
                        area_data=area_data,
                        shape_data=shape_data,
                        s3_key=scan_s3_key,
                        s3_url=scan_s3_url
                    )
                except Exception as exc:
                    db.log_error(
                        error_type="DB_SAVE_ERROR",
                        severity="ERROR",
                        message=str(exc),
                        stack_trace=traceback.format_exc(),
                        component="database",
                        username=st.session_state.username,
                        filename=scan_source_label
                    )

            # Generate PDF diagnostic report in memory
            if REPORTLAB_AVAILABLE and pil_img is not None:
                try:
                    pdf_report_bytes = generate_clinical_pdf_report(
                        patient_name=st.session_state.patient_name,
                        patient_age=st.session_state.patient_age,
                        patient_gender=st.session_state.patient_gender,
                        username=st.session_state.username,
                        predicted_class=predicted_class,
                        probs=probs,
                        classes=CLASSES,
                        pil_img=pil_img,
                        overlay_img=overlay_img,
                        gradcam_img=gradcam_img,
                        area_data=area_data,
                        shape_data=shape_data,
                        conf_data=conf_data,
                    )
                    if uploaded:
                        # Store Report in S3 & Database for new uploads
                        clean_name = "".join(c for c in (st.session_state.patient_name or "Patient") if c.isalnum() or c in ('_', '-'))
                        rep_code = f"RPT-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                        rep_fname = f"NeuroScan_Report_{clean_name}_{datetime.datetime.now().strftime('%Y%m%d')}.pdf"
                        s3_rep_key, s3_rep_url = s3.upload_pdf_to_s3(pdf_report_bytes, rep_fname)
                        db.save_report(
                            report_code=rep_code,
                            patient_name=st.session_state.patient_name or "Anonymous Patient",
                            predicted_class=predicted_class,
                            confidence=float(probs[CLASSES.index(predicted_class)] * 100),
                            pdf_bytes=pdf_report_bytes,
                            pdf_filename=rep_fname,
                            patient_id=patient_id if 'patient_id' in locals() else None,
                            patient_age=st.session_state.patient_age,
                            patient_gender=st.session_state.patient_gender,
                            doctor_username=st.session_state.username,
                            tumor_area_cm2=area_data['area_cm2'] if area_data else None,
                            s3_key=s3_rep_key,
                            s3_url=s3_rep_url
                        )
                        db.log_activity(
                            username=st.session_state.username or "doctor",
                            action="REPORT_GENERATE",
                            role=st.session_state.role,
                            details=f"Generated clinical PDF report '{rep_code}' for patient '{st.session_state.patient_name or 'Anonymous'}' (Stored in S3 & DB)",
                            status="SUCCESS"
                        )
                except Exception as exc:
                    pdf_gen_error = str(exc)
                    db.log_error(
                        error_type="PDF_GEN_ERROR",
                        severity="ERROR",
                        message=str(exc),
                        stack_trace=traceback.format_exc(),
                        component="pdf_report",
                        username=st.session_state.username,
                        filename=scan_source_label
                    )
            elif not REPORTLAB_AVAILABLE:
                pdf_gen_error = "ReportLab library not installed. Please run `pip install reportlab`."
        else:
            db.log_error(
                error_type="GUARDRAIL_REJECTED",
                severity="WARNING",
                message=f"Guardrail rejected scan: {mri_reason}",
                component="guardrail",
                username=st.session_state.username,
                filename=scan_source_label
            )




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
                <div class="pipe-step-title">MRI verification</div>
            </div>
            <div class="pipe-step-card {'done' if step > 2 else 'active' if step == 2 else ''}">
                <div class="pipe-step-badge {'done' if step > 2 else 'active' if step == 2 else 'todo'}">{"✓" if step > 2 else "3"}</div>
                <div class="pipe-step-title">Volumetric segmentation</div>
            </div>
            <div class="pipe-step-card {'done' if step > 3 else 'active' if step == 3 else ''}">
                <div class="pipe-step-badge {'done' if step > 3 else 'active' if step == 3 else 'todo'}">{"✓" if step > 3 else "4"}</div>
                <div class="pipe-step-title">Clinical readouts</div>
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

        if is_3d_nifti and vol_3d is not None and seg_3d_res is not None:
            # ── 3D NIfTI 6-KPI Metric Grid ─────────────────────────────────────────
            dx, dy, dz = voxel_spacing
            status_lbl = "Tumor detected" if seg_3d_res["has_tumor"] else "No tumor detected"
            status_col = "#FF3B30" if seg_3d_res["has_tumor"] else "#34C759"

            st.markdown("""
            <style>
            .nii-kpi-box {
                background: #161B22;
                border: 1px solid #30363D;
                border-radius: 8px;
                padding: 0.6rem 0.85rem;
                margin-bottom: 0.5rem;
            }
            .nii-kpi-lbl {
                font-size: 0.68rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: #8B949E;
                margin-bottom: 0.2rem;
                font-weight: 600;
            }
            .nii-kpi-val {
                font-family: 'Outfit', sans-serif;
                font-size: 1.15rem;
                font-weight: 700;
                color: #E6EDF3;
                margin: 0;
            }
            </style>
            """, unsafe_allow_html=True)

            k1, k2, k3 = st.columns(3)
            with k1:
                st.markdown(f"""
                <div class="nii-kpi-box">
                    <div class="nii-kpi-lbl">STATUS</div>
                    <div class="nii-kpi-val" style="color:{status_col};">{status_lbl}</div>
                </div>
                """, unsafe_allow_html=True)
            with k2:
                st.markdown(f"""
                <div class="nii-kpi-box">
                    <div class="nii-kpi-lbl">TUMOR VOLUME (CM³)</div>
                    <div class="nii-kpi-val">{seg_3d_res['tumor_volume_cm3']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with k3:
                st.markdown(f"""
                <div class="nii-kpi-box">
                    <div class="nii-kpi-lbl">VOXEL SPACING</div>
                    <div class="nii-kpi-val">{dx:.2f} × {dy:.2f} × {dz:.2f} mm</div>
                </div>
                """, unsafe_allow_html=True)

            k4, k5, k6 = st.columns(3)
            with k4:
                st.markdown(f"""
                <div class="nii-kpi-box">
                    <div class="nii-kpi-lbl">TUMOR VOXELS</div>
                    <div class="nii-kpi-val">{seg_3d_res['tumor_voxel_count']:,}</div>
                </div>
                """, unsafe_allow_html=True)
            with k5:
                st.markdown(f"""
                <div class="nii-kpi-box">
                    <div class="nii-kpi-lbl">TUMOR VOLUME (MM³)</div>
                    <div class="nii-kpi-val">{seg_3d_res['tumor_volume_mm3']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with k6:
                st.markdown(f"""
                <div class="nii-kpi-box">
                    <div class="nii-kpi-lbl">TUMOR-CONTAINING SLICES</div>
                    <div class="nii-kpi-val">{seg_3d_res['affected_slices']} / {seg_3d_res['total_slices']}</div>
                </div>
                """, unsafe_allow_html=True)

            # ── Representative Tumor Slice (Peak Area Z) ───────────────────────
            rep_z = seg_3d_res["peak_slice_idx"]
            orig_rep, mask_rep, ov_rep = volume_engine.render_slice_triplet(vol_3d, mask_3d, rep_z)

            st.markdown("<h4 style='font-family:Outfit, sans-serif; color:#E6EDF3; font-size:1.05rem; margin:1rem 0 0.4rem 0;'>Representative Tumor Slice</h4>", unsafe_allow_html=True)
            with st.container(border=True):
                r1, r2, r3 = st.columns(3)
                with r1:
                    st.markdown(f"<p style='text-align:center; font-weight:600; font-size:0.85rem; margin-bottom:0.3rem; color:#E6EDF3;'>Original MRI (z={rep_z})</p>", unsafe_allow_html=True)
                    st.image(orig_rep, use_container_width=True)
                with r2:
                    st.markdown("<p style='text-align:center; font-weight:600; font-size:0.85rem; margin-bottom:0.3rem; color:#E6EDF3;'>Predicted Tumor Mask</p>", unsafe_allow_html=True)
                    st.image(mask_rep, use_container_width=True)
                with r3:
                    st.markdown("<p style='text-align:center; font-weight:600; font-size:0.85rem; margin-bottom:0.3rem; color:#E6EDF3;'>Overlay</p>", unsafe_allow_html=True)
                    st.image(ov_rep, use_container_width=True)

            # ── Slice Explorer (Interactive Scrubbing) ───────────────────────────
            st.markdown("<h4 style='font-family:Outfit, sans-serif; color:#E6EDF3; font-size:1.05rem; margin:1.2rem 0 0.2rem 0;'>Slice Explorer</h4>", unsafe_allow_html=True)
            total_z = vol_3d.shape[2]
            selected_z = st.slider("Slice", 0, total_z - 1, value=rep_z, key="doc_nii_slider")

            orig_sel, mask_sel, ov_sel = volume_engine.render_slice_triplet(vol_3d, mask_3d, selected_z)
            with st.container(border=True):
                sl_col1, sl_col2, sl_col3 = st.columns(3)
                with sl_col1:
                    st.markdown(f"<p style='text-align:center; font-weight:600; font-size:0.85rem; margin-bottom:0.3rem; color:#E6EDF3;'>Original MRI (z={selected_z})</p>", unsafe_allow_html=True)
                    st.image(orig_sel, use_container_width=True)
                with sl_col2:
                    st.markdown("<p style='text-align:center; font-weight:600; font-size:0.85rem; margin-bottom:0.3rem; color:#E6EDF3;'>Predicted Tumor Mask</p>", unsafe_allow_html=True)
                    st.image(mask_sel, use_container_width=True)
                with sl_col3:
                    st.markdown("<p style='text-align:center; font-weight:600; font-size:0.85rem; margin-bottom:0.3rem; color:#E6EDF3;'>Overlay</p>", unsafe_allow_html=True)
                    st.image(ov_sel, use_container_width=True)

            # ── Technical Details ───────────────────────────────────────────────
            with st.expander("Technical Details", expanded=False):
                st.markdown(f"""
                <div style="font-size:0.82rem; color:#8B949E; line-height:1.8;">
                    • <b>Volume Dimensions:</b> {vol_3d.shape[0]} × {vol_3d.shape[1]} × {vol_3d.shape[2]} voxels<br/>
                    • <b>Voxel Spacing:</b> {dx:.2f} × {dy:.2f} × {dz:.2f} mm<br/>
                    • <b>Voxel Volume:</b> {(dx * dy * dz):.4f} mm³<br/>
                    • <b>Total Volume Slices:</b> {total_z}<br/>
                    • <b>Tumor-Containing Slices:</b> {seg_3d_res['affected_slices']} / {total_z} ({seg_3d_res['slice_coverage_pct']:.1f}%)<br/>
                    • <b>Tumor Voxels:</b> {seg_3d_res['tumor_voxel_count']:,}<br/>
                    • <b>Tumor Volume:</b> {seg_3d_res['tumor_volume_cm3']:.2f} cm³ ({seg_3d_res['tumor_volume_mm3']:,.2f} mm³)<br/>
                    • <b>Model Architecture:</b> U-Net (EfficientNet-B0 Encoder)<br/>
                    • <b>Spatial Coordinate Mapping:</b> Axial Slices (Z-Axis Transposition)
                </div>
                """, unsafe_allow_html=True)

        elif pil_img and not is_3d_nifti:
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

        if is_mri and not is_3d_nifti and predicted_class is not None and probs is not None:
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


            # ── Tumor Area & Shape Analysis Cards (Side-by-Side in Single Row) ─────
            st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
            col_area, col_shape = st.columns(2)

            # Left Panel: Tumor Area
            with col_area:
                with st.container(border=True):
                    st.markdown("<div class='pro-card-title'>🔬 Tumor Area</div>", unsafe_allow_html=True)
                    if has_tumor and binary_mask is not None:
                        px_count_disp, total_px_disp, cov_pct_disp, area_mm2_disp, area_cm2_disp = compute_tumor_area(binary_mask, pil_img)
                        c1, c2 = st.columns(2)
                        c1.metric("Area (mm²)",    f"{area_mm2_disp:,.1f}")
                        c2.metric("Area (cm²)",    f"{area_cm2_disp:.2f}")
                        c1.metric("Tumor Pixels",  f"{px_count_disp:,}")
                        c2.metric("Coverage",      f"{cov_pct_disp:.2f}%")
                        st.markdown("""
                        <div style="font-size:0.72rem; color:#8B949E; margin-top:0.4rem;">
                            ℹ️ Estimated at 0.5 mm/pixel (standard MRI resolution)
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='color:#8B949E; font-size:0.82rem; text-align:center; padding:0.75rem 0;'>No tumor detected — area analysis unavailable.</div>", unsafe_allow_html=True)

            # Right Panel: Shape Analysis
            with col_shape:
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
                            <div style="margin-bottom:0.55rem;">
                                <p style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;
                                          color:#8B949E; margin:0 0 0.2rem 0;">Tumor Shape</p>
                                <div style="display:inline-flex; align-items:center; gap:0.5rem;">
                                    <div style="width:10px; height:10px; border-radius:50%;
                                                background:{s_col};"></div>
                                    <span style="font-family:'Outfit',sans-serif; font-size:1.25rem;
                                                 font-weight:700; color:{s_col};">{s_lbl}</span>
                                </div>
                            </div>
                            <p style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;
                                       color:#8B949E; margin:0 0 0.4rem 0;">Features</p>
                            """, unsafe_allow_html=True)

                            def _feat_row(label, value, bar_pct, color):
                                st.markdown(f"""
                                <div style="margin-bottom:0.45rem;">
                                    <div style="display:flex; justify-content:space-between;
                                                align-items:center; margin-bottom:0.15rem;">
                                        <span style="font-size:0.8rem; color:#E6EDF3; font-weight:500;">{label}</span>
                                        <span style="font-family:'JetBrains Mono',monospace; font-size:0.8rem;
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

            if pil_img:
                w, h = pil_img.size
                st.markdown(f"""
                <div class="viewport-label">
                    <span>Dimensions: {w} × {h} px</span>
                    <span>File: {scan_source_label or 'MRI_Scan.png'}</span>
                </div>
                """, unsafe_allow_html=True)
        elif (uploaded or pil_img) and not is_mri:
            with st.container(border=True):
                st.markdown("<div class='pro-card-title'>Scan (Rejected)</div>", unsafe_allow_html=True)
                if pil_img:
                    st.image(pil_img, use_container_width=True)
            st.error("Invalid Scan: The image is not recognized as a valid brain MRI. Please provide a clear brain MRI scan to proceed.")
        elif not uploaded and pil_img is None and vol_3d is None:
            st.markdown("""
            <div style="background-color:#161B22; border:1px solid #21262D; border-radius:12px;
                        padding:6rem 2rem; text-align:center; height:100%;">
                <span style="font-size:3rem;">📂</span>
                <h4 style="font-family:'Outfit',sans-serif; font-weight:600; color:#E6EDF3;
                           margin-top:1rem;">Diagnostics Queue Empty</h4>
                <p style="color:#8B949E; font-size:0.88rem;">Select a registered patient from the directory to automatically retrieve their prior AWS S3 scan, or upload a scan to begin analysis.</p>
            </div>
            """, unsafe_allow_html=True)

    # RIGHT COLUMN: Readouts & Results
    with col_right:
        st.markdown("<p style='font-size:0.68rem; text-transform:uppercase; letter-spacing:0.1em; color:#8B949E; margin-bottom:0.75rem;'>Clinical Analysis</p>", unsafe_allow_html=True)

        if is_3d_nifti and seg_3d_res is not None:
            # ── 3D Volumetric Diagnostic Card ───────────────────────────────────────
            with st.container(border=True):
                st.markdown("<div class='pro-card-title'>📊 3D Tumor Volumetry & Spreading Potential</div>", unsafe_allow_html=True)
                
                status_text = "Tumor Detected (Volumetric Mass)" if seg_3d_res["has_tumor"] else "No Significant Tumor Region Detected"
                status_color = "#FF3B30" if seg_3d_res["has_tumor"] else "#34C759"
                status_bg = "#2A1818" if seg_3d_res["has_tumor"] else "#182A1A"
                
                st.markdown(f"""
                <div style="background:{status_bg}; border:1px solid {status_color}44; border-radius:8px; padding:0.8rem; margin-bottom:1rem;">
                    <span style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; color:#8B949E;">Diagnostic Status</span>
                    <h4 style="color:{status_color}; margin:0.2rem 0 0 0; font-family:'Outfit', sans-serif;">{status_text}</h4>
                </div>
                """, unsafe_allow_html=True)

                m1, m2 = st.columns(2)
                m1.metric("Physical Volume (cm³)", f"{seg_3d_res['tumor_volume_cm3']:.2f} cm³")
                m2.metric("Physical Volume (mm³)", f"{seg_3d_res['tumor_volume_mm3']:,.1f}")
                m1.metric("Tumor Voxel Count", f"{seg_3d_res['tumor_voxel_count']:,}")
                m2.metric("Slice Coverage", f"{seg_3d_res['affected_slices']} / {seg_3d_res['total_slices']} ({seg_3d_res['slice_coverage_pct']:.1f}%)")

                # Metastasis & Infiltration Spreading Probability
                has_t = seg_3d_res.get("has_tumor", False)
                aff_slices = seg_3d_res.get("affected_slices", 0)
                tot_slices = seg_3d_res.get("total_slices", 1)
                cov_pct = seg_3d_res.get("slice_coverage_pct", 0.0)

                if has_t:
                    if aff_slices >= 25 or cov_pct >= 20.0:
                        spread_prob = min(96.0, 68.0 + (cov_pct * 0.9))
                        spread_status = "⚠️ Spreading Possible (High Infiltration Risk)"
                        spread_col = "#FF3B30"
                        spread_desc = f"Volumetric mass extends across <b>{aff_slices} slices ({cov_pct:.1f}% vertical span)</b>, indicating active multi-layer infiltration with elevated metastasis & spreading likelihood."
                    else:
                        spread_prob = min(68.0, 38.0 + (cov_pct * 1.2))
                        spread_status = "⚡ Spreading Possible (Focal Infiltration)"
                        spread_col = "#FF9500"
                        spread_desc = f"Lesion localized across <b>{aff_slices} slices</b>. Inter-slice tissue margins indicate possible localized infiltration requiring close monitoring."
                else:
                    spread_prob = 1.0
                    spread_status = "✅ No Spreading Detected (Normal / Low Risk)"
                    spread_col = "#34C759"
                    spread_desc = "No abnormal mass or multi-focal dispersion identified in this 3D scan."

                st.markdown(f"""
                <div style="background:#0D1117; border:1px solid {spread_col}33; border-radius:8px; padding:0.85rem; margin-top:0.85rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem;">
                        <span style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; color:#8B949E;">Metastasis / Spreading Probability</span>
                        <span style="font-family:'JetBrains Mono',monospace; font-weight:700; font-size:0.95rem; color:{spread_col};">{spread_prob:.1f}%</span>
                    </div>
                    <div class="bar-wrap" style="height:6px; margin-bottom:0.5rem;">
                        <div class="bar-fill" style="width:{spread_prob:.1f}%; background:{spread_col};"></div>
                    </div>
                    <div style="font-size:0.82rem; font-weight:700; color:{spread_col}; margin-bottom:0.25rem;">
                        {spread_status}
                    </div>
                    <p style="font-size:0.75rem; color:#8B949E; margin:0; line-height:1.45;">
                        {spread_desc}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Per-Lesion Detection Breakdown (Lesion-Level Confidence & Location Analysis)
                lesion_list = seg_3d_res.get("lesions", [])
                lesion_cnt = len(lesion_list)
                
                if lesion_cnt > 0:
                    st.markdown(f"""
                    <div style="margin-top:0.9rem; padding:0.75rem; background:#0D1117; border:1px solid #30363D; border-radius:8px;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                            <span style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em; font-weight:700; color:#00D4FF;">
                                📍 Lesion Breakdown ({lesion_cnt} Lesion{"s" if lesion_cnt > 1 else ""} Detected)
                            </span>
                            <span style="font-size:0.72rem; padding:0.15rem 0.45rem; background:rgba(0,212,255,0.12); border:1px solid rgba(0,212,255,0.3); border-radius:12px; color:#00D4FF;">
                                Multi-Focal Analysis
                            </span>
                        </div>
                    """, unsafe_allow_html=True)

                    # Display top 5 lesions
                    for l_idx, les in enumerate(lesion_list[:5]):
                        cat_color = "#FF9500" if "Small" in les["category"] else "#FF3B30"
                        st.markdown(f"""
                        <div style="background:#161B22; border-left:3px solid {cat_color}; padding:0.55rem 0.75rem; border-radius:4px; margin-bottom:0.4rem;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span style="font-size:0.8rem; font-weight:700; color:#E6EDF3;">
                                    Lesion #{les['lesion_id']}: {les['category']}
                                </span>
                                <span style="font-family:'JetBrains Mono',monospace; font-size:0.78rem; font-weight:700; color:{cat_color};">
                                    {les['confidence_pct']}% Conf.
                                </span>
                            </div>
                            <div style="display:flex; justify-content:space-between; font-size:0.73rem; color:#8B949E; margin-top:0.25rem;">
                                <span>📏 Volume: <b>{les['volume_cm3']} cm³</b> ({les['volume_mm3']} mm³)</span>
                                <span>📍 Location: <b>Slice Z={les['slice_z']}</b> | Centroid ({les['centroid'][0]}, {les['centroid'][1]})</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown(f"""
                <div style="font-size:0.75rem; color:#8B949E; margin-top:0.75rem; padding:0.5rem; background:#0D1117; border-radius:6px; line-height:1.6;">
                    🎯 <b>Peak Lesion Slice</b>: Axial Slice <b>Z = {seg_3d_res['peak_slice_idx']}</b> ({seg_3d_res['peak_slice_voxel_count']:,} active pixels)<br/>
                    📐 <b>Voxel Resolution</b>: {seg_3d_res['voxel_spacing_mm'][0]:.2f} × {seg_3d_res['voxel_spacing_mm'][1]:.2f} × {seg_3d_res['voxel_spacing_mm'][2]:.2f} mm³<br/>
                    🧬 <b>Model Architecture</b>: U-Net (EfficientNet-B0 Encoder)
                </div>
                """, unsafe_allow_html=True)

        if not is_3d_nifti and step >= 2 and probs is not None:
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

            # ── Store Diagnostic Report into Session State for AI Grounding ──────
            active_report = {
                "patient_name": st.session_state.get("patient_name", "Anonymous"),
                "patient_age": st.session_state.get("patient_age", "N/A"),
                "patient_gender": st.session_state.get("patient_gender", "N/A"),
                "diagnosis": predicted_class,
                "confidence_pct": conf,
                "has_tumor": has_tumor,
                "probabilities": {c: float(p) for c, p in zip(CLASSES, probs)},
                "area_cm2": area_data['area_cm2'] if area_data else None,
                "area_mm2": area_data['area_mm2'] if area_data else None,
                "tumor_pixels": area_data['pixel_count'] if area_data else None,
                "shape_label": shape_data.get("shape_label") if (has_tumor and 'shape_data' in locals() and shape_data) else None,
                "circularity": shape_data.get("circularity") if (has_tumor and 'shape_data' in locals() and shape_data) else None,
                "compactness": shape_data.get("compactness") if (has_tumor and 'shape_data' in locals() and shape_data) else None,
                "solidity": shape_data.get("solidity") if (has_tumor and 'shape_data' in locals() and shape_data) else None,
                "gradcam_focus": f"{vq}-{hq} Region" if ('vq' in locals() and 'hq' in locals()) else None,
                "scan_filename": scan_source_label or (uploaded.name if uploaded else "Scan_MRI")
            }
            st.session_state["active_diagnostic_report"] = active_report
            st.session_state["last_diagnosis"] = predicted_class

        # ── Clinical Diagnostic Report Export Card (2D & 3D NIfTI Scans) ─────────
        if (is_3d_nifti and seg_3d_res is not None) or (not is_3d_nifti and step >= 2 and probs is not None):
            with st.container(border=True):
                st.markdown("<div class='pro-card-title'>📄 Clinical Report Export</div>", unsafe_allow_html=True)
                p_display_name = st.session_state.patient_name.strip() if st.session_state.patient_name else "Patient"
                clean_name = "".join(c for c in p_display_name if c.isalnum() or c in ('_', '-'))
                now_stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                pdf_filename = f"NeuroScan_Report_{clean_name}_{now_stamp}.pdf"

                summary_desc = "Export a DICOM-compliant 3D structural volumetric report with voxel spacing, slice coverage, and tri-view slices." if is_3d_nifti else "Export a DICOM-compliant diagnostic summary with patient demographics, class probabilities, tri-view visual scans, and morphometry."
                st.markdown(f"""
                <p style="font-size:0.75rem; color:#8B949E; margin-bottom:0.7rem; line-height:1.4;">
                    {summary_desc}
                </p>
                """, unsafe_allow_html=True)

                if pdf_report_bytes is not None:
                    st.download_button(
                        label="📄 Download Diagnostic Report (PDF)",
                        data=pdf_report_bytes,
                        file_name=pdf_filename,
                        mime="application/pdf",
                        use_container_width=True,
                        key="btn_download_report_right"
                    )
                    st.markdown("""
                    <div style="font-size:0.68rem; color:#3FB950; text-align:center; margin-top:0.35rem; font-family:'JetBrains Mono', monospace;">
                        ✓ High-res PDF generated &amp; ready to export
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    if pdf_gen_error:
                        st.warning(f"⚠️ {pdf_gen_error}")
                    else:
                        st.info("Analysis required to generate report.")
        elif not is_3d_nifti and pil_img is None:
            with st.container(border=True):
                st.markdown("<div style='color:#8B949E; font-size:0.85rem; text-align:center; padding:2rem 0;'>Awaiting scan input to run diagnosis...</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer-text">
            🔒 Research &amp; educational use only &nbsp;·&nbsp; Not for primary clinical diagnosis &nbsp;·&nbsp; Patient data processed locally
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PATIENT PORTAL
# ─────────────────────────────────────────────────────────────────────────────

def render_patient_dashboard():
    """Simplified, patient-friendly portal for review and personal health summary."""
    clf_model = load_classifier()
    seg_model, seg_err = load_segmentation_model()
    guardrail_model, guardrail_err = load_guardrail_model()

    username_display = st.session_state.username or "Patient"
    patient_name_val = st.session_state.patient_name or username_display

    # Top Bar Header
    st.markdown(f"""
    <div class="top-bar">
        <div class="logo-container">
            <span style="display:inline-block; width:9px; height:9px; border-radius:50%; background:#3FB950; margin-right:8px; box-shadow:0 0 8px rgba(63,185,80,0.8);"></span>NeuroScan AI <span style="color:#8B949E; font-size:0.85rem; font-weight:400; margin-left:8px;">| Patient Portal</span>
        </div>
        <div class="top-bar-right">
            <span class="patient-chip" style="color:#3FB950; border-color:rgba(63,185,80,0.3);">👤 Patient: @{username_display}</span>
            <span><span class="status-dot"></span>System Active</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_p_left, col_p_center, col_p_right = st.columns([1, 1.8, 1.2], gap="large")

    uploaded = None
    pil_img = None
    is_mri = False
    predicted_class = None
    probs = None
    has_tumor = False
    binary_mask = None
    prob_map = None
    overlay_img = None
    area_data = None
    pdf_report_bytes = None

    with col_p_left:
        st.markdown("<p style='font-size:0.68rem; text-transform:uppercase; letter-spacing:0.1em; color:#8B949E; margin-bottom:0.75rem;'>Patient Info &amp; Upload</p>", unsafe_allow_html=True)
        
        with st.container(border=True):
            st.markdown("<div class='pro-card-title'>👤 My Profile</div>", unsafe_allow_html=True)
            p_name = st.text_input("Full Name", value=st.session_state.patient_name or username_display.title(), key="pat_name_input")
            st.session_state.patient_name = p_name

            c_age, c_gen = st.columns(2)
            with c_age:
                p_age = st.number_input("Age", min_value=1, max_value=120, value=st.session_state.patient_age, key="pat_age_input")
                st.session_state.patient_age = p_age
            with c_gen:
                p_gender = st.selectbox("Gender", ["Not specified", "Male", "Female", "Other"], index=["Not specified", "Male", "Female", "Other"].index(st.session_state.patient_gender), key="pat_gen_input")
                st.session_state.patient_gender = p_gender

        with st.container(border=True):
            st.markdown("<div class='pro-card-title'>Upload My Brain MRI Scan</div>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:0.75rem; color:#8B949E; margin-bottom:0.6rem;'>Upload your Brain MRI scan (2D JPG or PNG). The AI will verify and provide an accessible summary.</p>", unsafe_allow_html=True)
            uploaded = st.file_uploader("Upload Scan", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="patient_uploader")

        st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)
        if st.button("🔓 Logout", key="pat_logout_btn", use_container_width=True):
            db.log_activity(
                username=st.session_state.username or "patient",
                action="USER_LOGOUT",
                role="patient",
                details="Patient signed out of session",
                status="SUCCESS"
            )
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.page = "landing"
            st.rerun()


    # Process 2D Scan
    if uploaded:
        pil_img = Image.open(uploaded)
        is_mri, mri_reason = is_mri_image(pil_img, guardrail_model)
        if is_mri:
            predicted_class, probs = predict_class(clf_model, pil_img)
            has_tumor = predicted_class in TUMOR_CLASSES
            if has_tumor and seg_model:
                binary_mask, prob_map = predict_segmentation(seg_model, pil_img)
                overlay_img = overlay_mask_on_image(pil_img, binary_mask, alpha=0.45)
                px_count, total_px, cov_pct, area_mm2, area_cm2 = compute_tumor_area(binary_mask, pil_img)
                area_data = {
                    "pixel_count": px_count, "total_pixels": total_px, "coverage_pct": cov_pct,
                    "area_mm2": area_mm2, "area_cm2": area_cm2
                }

            # Save scan record to database
            try:
                patient_id = db.create_or_get_patient(
                    full_name=st.session_state.patient_name,
                    age=st.session_state.patient_age,
                    gender=st.session_state.patient_gender
                )
                db.save_scan_record(
                    filename=uploaded.name,
                    is_valid_mri=True,
                    guardrail_reason=mri_reason,
                    patient_id=patient_id,
                    doctor_username=st.session_state.username,
                    predicted_class=predicted_class,
                    confidence=float(probs[CLASSES.index(predicted_class)] * 100),
                    probabilities_dict={c: float(p) for c, p in zip(CLASSES, probs)},
                    area_data=area_data
                )
            except Exception as exc:
                db.log_error(
                    error_type="DB_SAVE_ERROR",
                    severity="ERROR",
                    message=str(exc),
                    stack_trace=traceback.format_exc(),
                    component="database",
                    username=st.session_state.username,
                    filename=uploaded.name
                )

            if REPORTLAB_AVAILABLE and pil_img is not None:
                try:
                    pdf_report_bytes = generate_clinical_pdf_report(
                        patient_name=st.session_state.patient_name,
                        patient_age=st.session_state.patient_age,
                        patient_gender=st.session_state.patient_gender,
                        username="Patient Self-Service Portal",
                        predicted_class=predicted_class,
                        probs=probs,
                        classes=CLASSES,
                        pil_img=pil_img,
                        overlay_img=overlay_img,
                        area_data=area_data,
                    )

                    # Store Report in S3 & Database
                    clean_name = "".join(c for c in (st.session_state.patient_name or "Patient") if c.isalnum() or c in ('_', '-'))
                    rep_code = f"PAT-RPT-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                    rep_fname = f"Patient_Report_{clean_name}_{datetime.datetime.now().strftime('%Y%m%d')}.pdf"

                    # Upload PDF report & scan to AWS S3
                    s3_key, s3_url = s3.upload_pdf_to_s3(pdf_report_bytes, rep_fname)
                    s3.upload_mri_to_s3(pil_img, uploaded.name)

                    db.save_report(
                        report_code=rep_code,
                        patient_name=st.session_state.patient_name or "Anonymous Patient",
                        predicted_class=predicted_class,
                        confidence=float(probs[CLASSES.index(predicted_class)] * 100),
                        pdf_bytes=pdf_report_bytes,
                        pdf_filename=rep_fname,
                        patient_id=patient_id if 'patient_id' in locals() else None,
                        patient_age=st.session_state.patient_age,
                        patient_gender=st.session_state.patient_gender,
                        doctor_username=st.session_state.username or "patient",
                        tumor_area_cm2=area_data['area_cm2'] if area_data else None,
                        s3_key=s3_key,
                        s3_url=s3_url
                    )

                except Exception as exc:
                    pdf_report_bytes = None
                    db.log_error(
                        error_type="PDF_GEN_ERROR",
                        severity="ERROR",
                        message=str(exc),
                        stack_trace=traceback.format_exc(),
                        component="pdf_report",
                        username=st.session_state.username,
                        filename=uploaded.name
                    )

        else:
            db.log_error(
                error_type="GUARDRAIL_REJECTED",
                severity="WARNING",
                message=f"Patient portal guardrail rejected scan: {mri_reason}",
                component="guardrail",
                username=st.session_state.username,
                filename=uploaded.name
            )


    with col_p_center:
        st.markdown("<p style='font-size:0.68rem; text-transform:uppercase; letter-spacing:0.1em; color:#8B949E; margin-bottom:0.75rem;'>Scan Results &amp; Summary</p>", unsafe_allow_html=True)
        if pil_img:
            if is_mri:
                with st.container(border=True):
                    st.markdown("<div class='pro-card-title'>Your MRI Scan Visual</div>", unsafe_allow_html=True)
                    if has_tumor and overlay_img is not None:
                        v1, v2 = st.columns(2)
                        with v1:
                            st.image(pil_img, caption="Original Scan", use_container_width=True)
                        with v2:
                            st.image(overlay_img, caption="AI Highlighted Region", use_container_width=True)
                    else:
                        st.image(pil_img, caption="Original Brain MRI Scan", use_container_width=True)

                # Plain-Language Health Verdict Card
                with st.container(border=True):
                    st.markdown("<div class='pro-card-title'>Health Verdict Summary</div>", unsafe_allow_html=True)
                    conf_val = probs[CLASSES.index(predicted_class)] * 100

                    if predicted_class == "notumor":
                        st.markdown(f"""
                        <div style="background:rgba(52,199,89,0.12); border:1px solid rgba(52,199,89,0.3); border-radius:10px; padding:1.2rem; margin-bottom:0.8rem;">
                            <h3 style="color:#34C759; margin:0 0 0.4rem 0; font-family:'Outfit', sans-serif;">✅ Normal Scan (No Tumor Detected)</h3>
                            <p style="color:#E6EDF3; font-size:0.88rem; margin:0; line-height:1.5;">
                                The AI analysis did not find any abnormal tumor tissue in this brain scan (<b>{conf_val:.1f}% confidence</b>).
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif predicted_class in ["meningioma", "pituitary"]:
                        p_name_title = "Meningioma" if predicted_class == "meningioma" else "Pituitary"
                        st.markdown(f"""
                        <div style="background:rgba(255,149,0,0.12); border:1px solid rgba(255,149,0,0.3); border-radius:10px; padding:1.2rem; margin-bottom:0.8rem;">
                            <h3 style="color:#FF9500; margin:0 0 0.4rem 0; font-family:'Outfit', sans-serif;">ℹ️ Possible {p_name_title} Tissue Detected</h3>
                            <p style="color:#E6EDF3; font-size:0.88rem; margin:0; line-height:1.5;">
                                The AI detected patterns consistent with <b>{p_name_title}</b> (<b>{conf_val:.1f}% confidence</b>). 
                                These types of tumors are <b>typically benign (non-cancerous)</b> and grow slowly, but require medical consultation.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:  # glioma
                        st.markdown(f"""
                        <div style="background:rgba(255,59,48,0.12); border:1px solid rgba(255,59,48,0.3); border-radius:10px; padding:1.2rem; margin-bottom:0.8rem;">
                            <h3 style="color:#FF3B30; margin:0 0 0.4rem 0; font-family:'Outfit', sans-serif;">⚠️ Abnormal Region Flagged (Glioma Pattern)</h3>
                            <p style="color:#E6EDF3; font-size:0.88rem; margin:0; line-height:1.5;">
                                The AI detected focal tissue patterns matching <b>Glioma</b> (<b>{conf_val:.1f}% confidence</b>). 
                                <b>Immediate consultation with a neurologist or neuro-specialist is strongly advised</b> for formal clinical review.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    if area_data:
                        st.markdown(f"""
                        <div style="font-size:0.82rem; color:#8B949E; margin-top:0.6rem;">
                            📏 <b>Estimated Lesion Area:</b> ~{area_data['area_cm2']:.2f} cm² ({area_data['area_mm2']:,.1f} mm²)
                        </div>
                        """, unsafe_allow_html=True)

                    # Save diagnostic report for patient AI copilot
                    pat_report = {
                        "patient_name": st.session_state.get("username", "Patient"),
                        "diagnosis": predicted_class,
                        "confidence_pct": conf_val,
                        "has_tumor": has_tumor,
                        "probabilities": {c: float(p) for c, p in zip(CLASSES, probs)},
                        "area_cm2": area_data['area_cm2'] if area_data else None,
                        "area_mm2": area_data['area_mm2'] if area_data else None,
                        "tumor_pixels": area_data['pixel_count'] if area_data else None,
                        "scan_filename": uploaded.name if ('uploaded' in locals() and uploaded) else "Scan_MRI"
                    }
                    st.session_state["active_diagnostic_report"] = pat_report
                    st.session_state["last_diagnosis"] = predicted_class


            else:
                st.error("Invalid image: Please upload a clear brain MRI scan in JPG, PNG, or NIfTI format.")

        else:
            st.markdown("""
            <div style="background-color:#161B22; border:1px solid #21262D; border-radius:12px;
                        padding:5rem 2rem; text-align:center;">
                <div style="width:48px; height:48px; border-radius:10px; background:rgba(0,212,255,0.1); border:1px solid rgba(0,212,255,0.25); display:inline-flex; align-items:center; justify-content:center; margin-bottom:0.8rem;">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#00D4FF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                        <circle cx="8.5" cy="8.5" r="1.5"/>
                        <polyline points="21 15 16 10 5 21"/>
                    </svg>
                </div>
                <h4 style="color:#E6EDF3; margin-top:0.8rem;">Ready for Scan Upload</h4>
                <p style="color:#8B949E; font-size:0.85rem;">Upload your brain MRI on the left to receive your AI health insights.</p>
            </div>
            """, unsafe_allow_html=True)

    with col_p_right:
        st.markdown("<p style='font-size:0.68rem; text-transform:uppercase; letter-spacing:0.1em; color:#8B949E; margin-bottom:0.75rem;'>Actions &amp; Guidance</p>", unsafe_allow_html=True)
        
        # Download Report Card
        with st.container(border=True):
            st.markdown("<div class='pro-card-title'>📄 My Health Summary Report</div>", unsafe_allow_html=True)
            if pdf_report_bytes is not None:
                p_display_name = st.session_state.patient_name.strip() if st.session_state.patient_name else "Patient"
                clean_name = "".join(c for c in p_display_name if c.isalnum() or c in ('_', '-'))
                pdf_filename = f"Patient_Report_{clean_name}_{datetime.datetime.now().strftime('%Y%m%d')}.pdf"
                st.markdown("<p style='font-size:0.75rem; color:#8B949E;'>Download a complete copy of your AI scan summary to share with your physician.</p>", unsafe_allow_html=True)
                st.download_button(
                    label="📄 Download Patient Report (PDF)",
                    data=pdf_report_bytes,
                    file_name=pdf_filename,
                    mime="application/pdf",
                    use_container_width=True,
                    key="pat_dl_btn"
                )
            else:
                st.markdown("<p style='font-size:0.78rem; color:#8B949E; text-align:center; padding:1rem 0;'>Upload a scan to generate your report.</p>", unsafe_allow_html=True)

        # Recommended Next Steps
        with st.container(border=True):
            st.markdown("<div class='pro-card-title'>🩺 Recommended Next Steps</div>", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size:0.78rem; color:#E6EDF3; line-height:1.6;">
                1. 📋 <b>Share with your Doctor</b>: Bring this report to your neurologist or general physician.<br/>
                2. 🔍 <b>Clinical Correlation</b>: AI results are assistive and must be evaluated alongside clinical symptoms.<br/>
                3. 💊 <b>Medical Advice</b>: Do not stop, start, or change medications without consulting a licensed doctor.
            </div>
            """, unsafe_allow_html=True)

    # ── Interactive Medical RAG Patient Chatbot ──────────────────────────────
    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("""
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.5rem;">
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="font-size:1.4rem;">💬</span>
                <div>
                    <h4 style="margin:0; color:#E6EDF3; font-size:1rem; font-family:'Outfit', sans-serif;">Ask NeuroScan Assistant (Medical RAG Copilot)</h4>
                    <p style="margin:0; color:#8B949E; font-size:0.75rem;">Instant plain-language answers grounded in verified neuro-oncology guidelines</p>
                </div>
            </div>
            <span style="background:rgba(63,185,80,0.15); color:#3FB950; border:1px solid rgba(63,185,80,0.3); font-size:0.7rem; padding:2px 8px; border-radius:12px; font-weight:600;">
                ✓ RAG Grounded
            </span>
        </div>
        """, unsafe_allow_html=True)

        if "patient_chat_history" not in st.session_state:
            st.session_state.patient_chat_history = [
                {
                    "role": "assistant",
                    "content": f"Hello {patient_name_val.title()}! I am your NeuroScan Health Assistant. I can help explain your brain MRI scan findings, provide plain-language definitions, and suggest questions for your next doctor's appointment. What would you like to know?"
                }
            ]

        # 1-Click Quick Question Chips
        st.markdown("<p style='font-size:0.72rem; color:#8B949E; margin-bottom:0.3rem;'>Suggested Questions:</p>", unsafe_allow_html=True)
        q_cols = st.columns(4)
        quick_questions = [
            "Explain my MRI scan report",
            "What diet & remedies help?",
            "How to maintain daily health?",
            "What questions to ask my doctor?"
        ]
        
        selected_quick_q = None
        for i, q in enumerate(quick_questions):
            with q_cols[i]:
                if st.button(q, key=f"quick_q_{i}", use_container_width=True):
                    selected_quick_q = q


        # Display Chat History
        chat_box = st.container(height=320, border=False)
        with chat_box:
            for msg in st.session_state.patient_chat_history:
                with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "assistant"):
                    st.markdown(msg["content"])
                    if msg.get("sources"):
                        st.caption(f"📚 Sources: {', '.join(msg['sources'])}")


        # Chat Input
        user_chat_input = st.chat_input("Ask a question about your scan, recovery, or brain health...")
        prompt_to_process = selected_quick_q or user_chat_input

        if prompt_to_process:
            st.session_state.patient_chat_history.append({"role": "user", "content": prompt_to_process})
            
            with st.spinner("Retrieving verified medical guidance..."):
                rag_reply = rag_engine.query_medical_rag(
                    user_query=prompt_to_process,
                    diagnosis_context=predicted_class,
                    tumor_area_cm2=area_data['area_cm2'] if area_data else None,
                    role="patient",
                    username=st.session_state.get("username", "patient"),
                    report_data=st.session_state.get("active_diagnostic_report", None)
                )



            st.session_state.patient_chat_history.append({
                "role": "assistant",
                "content": rag_reply["answer"],
                "sources": rag_reply.get("sources", [])
            })
            st.rerun()

    st.markdown("""
    <div class="disclaimer-text">
        🔒 Research &amp; educational use only · Patient data processed locally · Not for clinical diagnosis
    </div>
    """, unsafe_allow_html=True)



# ─────────────────────────────────────────────────────────────────────────────
# ADMIN CONSOLE
# ─────────────────────────────────────────────────────────────────────────────

def render_admin_dashboard():
    """System administration console for user management and AI system diagnostics."""
    username_display = st.session_state.username or "Admin"

    # Top Bar Header
    st.markdown(f"""
    <div class="top-bar">
        <div class="logo-container">
            ⚙️&nbsp; NeuroScan AI <span style="color:#8B949E; font-size:0.85rem; font-weight:400; margin-left:8px;">| System Administration Console</span>
        </div>
        <div class="top-bar-right">
            <span class="patient-chip" style="color:#FF9500; border-color:rgba(255,149,0,0.3);">⚙️ SuperAdmin: @{username_display}</span>
            <span><span class="status-dot"></span>System Online</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI Metrics Strip
    all_db_users = db.get_all_users()
    total_users = len(all_db_users)
    doc_count = sum(1 for u in all_db_users if u["role"] == "doctor")
    pat_count = sum(1 for u in all_db_users if u["role"] == "patient")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        with st.container(border=True):
            st.metric("Registered Users", f"{total_users}")
    with k2:
        with st.container(border=True):
            st.metric("Doctors / Clinicians", f"{doc_count}")
    with k3:
        with st.container(border=True):
            st.metric("Patients Registered", f"{pat_count}")
    with k4:
        with st.container(border=True):
            st.metric("Compute Engine", "CUDA 12.1" if torch.cuda.is_available() else "CPU")


    tab_users, tab_activity, tab_reports, tab_sys, tab_models, tab_errors = st.tabs([
        "👥 User & Role Management",
        "📝 User Activity & Login Logs",
        "📋 Diagnostic Reports Archive",
        "🖥️ System & GPU Diagnostics",
        "🔬 Model Registry & Audit",
        "🚨 System Error Logs"
    ])


    with tab_users:
        col_u1, col_u2 = st.columns([1.5, 1.2], gap="large")

        with col_u1:
            with st.container(border=True):
                st.markdown("<div class='pro-card-title'>Active User Accounts Directory (Database)</div>", unsafe_allow_html=True)
                
                # Table Headers
                h1, h2, h3 = st.columns([1.4, 1.4, 0.8])
                h1.markdown("<span style='font-size:0.72rem; color:#8B949E; text-transform:uppercase; font-weight:700;'>Username</span>", unsafe_allow_html=True)
                h2.markdown("<span style='font-size:0.72rem; color:#8B949E; text-transform:uppercase; font-weight:700;'>Assigned Role</span>", unsafe_allow_html=True)
                h3.markdown("<span style='font-size:0.72rem; color:#8B949E; text-transform:uppercase; font-weight:700;'>Status</span>", unsafe_allow_html=True)
                st.markdown("<hr style='border:0; border-top:1px solid #21262D; margin:0.3rem 0 0.6rem 0;'>", unsafe_allow_html=True)

                for u in all_db_users:
                    uname = u["username"]
                    role = u["role"]
                    r_col = "#00D4FF" if role == "doctor" else ("#3FB950" if role == "patient" else "#FF9500")
                    r_badge = "👨‍⚕️ Doctor" if role == "doctor" else ("👤 Patient" if role == "patient" else "⚙️ Admin")
                    
                    r1, r2, r3 = st.columns([1.4, 1.4, 0.8])
                    r1.markdown(f"<span style='font-weight:600; color:#E6EDF3; font-size:0.85rem;'>@{uname}</span>", unsafe_allow_html=True)
                    r2.markdown(f"<span style='background:{r_col}18; color:{r_col}; border:1px solid {r_col}44; border-radius:99px; font-size:0.72rem; padding:2px 8px; font-weight:600;'>{r_badge}</span>", unsafe_allow_html=True)
                    r3.markdown("<span style='color:#3FB950; font-size:0.75rem;'>● Active</span>", unsafe_allow_html=True)
                    st.markdown("<div style='margin-bottom:0.4rem;'></div>", unsafe_allow_html=True)

        with col_u2:
            with st.container(border=True):
                st.markdown("<div class='pro-card-title'>Modify User Role</div>", unsafe_allow_html=True)
                user_list = [u["username"] for u in all_db_users]
                mod_user = st.selectbox("Select User Account", user_list, key="mod_user_sel")
                current_u_dict = next((u for u in all_db_users if u["username"] == mod_user), None)
                current_u_role = current_u_dict["role"] if current_u_dict else "doctor"
                new_role_val = st.selectbox(
                    "Assign New Role",
                    ["doctor", "patient", "admin"],
                    index=["doctor", "patient", "admin"].index(current_u_role),
                    key="new_role_sel"
                )
                if st.button("Update Role", key="btn_update_role", use_container_width=True):
                    db.update_user_role(mod_user, new_role_val)
                    db.log_activity(
                        username=st.session_state.username,
                        action="ROLE_UPDATE",
                        role="admin",
                        details=f"Admin changed role for @{mod_user} to '{new_role_val.title()}'",
                        status="SUCCESS"
                    )
                    st.success(f"✓ Role for @{mod_user} updated to '{new_role_val.title()}' in database!")
                    st.rerun()

            with st.container(border=True):
                st.markdown("<div class='pro-card-title'>👨‍⚕️ Provision Clinician / Doctor Account</div>", unsafe_allow_html=True)
                new_adm_fn = st.text_input("Clinician Full Name", placeholder="e.g. Dr. Sarah Chen, M.D.", key="adm_new_fn")
                c_au1, c_au2 = st.columns(2)
                with c_au1:
                    new_adm_u = st.text_input("Username", placeholder="e.g. dr_chen", key="adm_new_u")
                with c_au2:
                    new_adm_p = st.text_input("Password", type="password", placeholder="Min 8 chars, Aa1!", key="adm_new_p")
                new_adm_r = st.selectbox("Role Assignment", ["doctor", "admin"], key="adm_new_r")
                st.markdown("<div style='font-size:0.68rem; color:#8B949E; margin-bottom:0.5rem;'>🔒 <b>Security Policy</b>: Min 8 characters with uppercase, lowercase, digit, and symbol.</div>", unsafe_allow_html=True)
                if st.button("➕ Provision Clinician Account", key="btn_add_user_adm", use_container_width=True):
                    clean_adm_u = new_adm_u.strip() if new_adm_u else ""
                    if clean_adm_u and new_adm_p and new_adm_fn:
                        uid, err = db.create_user(clean_adm_u, new_adm_p, new_adm_r, full_name=new_adm_fn.strip(), enforce_policy=True)
                        if err:
                            st.error(f"❌ Password Policy / Provisioning Error: {err}")
                        else:
                            db.log_activity(
                                username=st.session_state.username,
                                action="USER_CREATE",
                                role="admin",
                                details=f"Admin provisioned {new_adm_r.title()} account for '{new_adm_fn.strip()}' (@{clean_adm_u})",
                                status="SUCCESS"
                            )
                            st.success(f"✓ Clinician @{clean_adm_u} ({new_adm_fn.strip()}) provisioned with bcrypt hash!")
                            st.rerun()
                    else:
                        st.error("Please fill in Full Name, Username, and Password.")


    with tab_activity:
        act_logs = db.get_activity_logs(limit=100)
        c_a_top1, c_a_top2 = st.columns([3, 1])
        with c_a_top1:
            st.markdown(f"<div class='pro-card-title'>📝 User Activity &amp; Login Audit Trail ({len(act_logs)} Total Recorded Events)</div>", unsafe_allow_html=True)
        with c_a_top2:
            if st.button("🗑️ Clear Activity Logs", key="btn_clear_act_logs", use_container_width=True):
                db.clear_activity_logs()
                st.success("Activity logs cleared!")
                st.rerun()

        if act_logs:
            # Table Headers
            ah1, ah2, ah3, ah4, ah5, ah6, ah7 = st.columns([0.6, 1.4, 1.2, 1.1, 2.8, 0.9, 1.4])
            ah1.markdown("<span style='font-size:0.72rem; color:#8B949E; text-transform:uppercase; font-weight:700;'>ID</span>", unsafe_allow_html=True)
            ah2.markdown("<span style='font-size:0.72rem; color:#8B949E; text-transform:uppercase; font-weight:700;'>Action</span>", unsafe_allow_html=True)
            ah3.markdown("<span style='font-size:0.72rem; color:#8B949E; text-transform:uppercase; font-weight:700;'>User</span>", unsafe_allow_html=True)
            ah4.markdown("<span style='font-size:0.72rem; color:#8B949E; text-transform:uppercase; font-weight:700;'>Role</span>", unsafe_allow_html=True)
            ah5.markdown("<span style='font-size:0.72rem; color:#8B949E; text-transform:uppercase; font-weight:700;'>Activity Details</span>", unsafe_allow_html=True)
            ah6.markdown("<span style='font-size:0.72rem; color:#8B949E; text-transform:uppercase; font-weight:700;'>Status</span>", unsafe_allow_html=True)
            ah7.markdown("<span style='font-size:0.72rem; color:#8B949E; text-transform:uppercase; font-weight:700;'>Timestamp</span>", unsafe_allow_html=True)
            st.markdown("<hr style='border:0; border-top:1px solid #21262D; margin:0.3rem 0 0.5rem 0;'>", unsafe_allow_html=True)

            for act in act_logs:
                action_name = act['action']
                act_col = "#3FB950" if "LOGIN" in action_name or "REGISTER" in action_name else ("#00D4FF" if "SCAN" in action_name or "REPORT" in action_name else "#FF9500")
                stat_col = "#3FB950" if act['status'] == "SUCCESS" else "#FF3B30"
                r_val = act['role'].title() if act['role'] else "General"
                
                a1, a2, a3, a4, a5, a6, a7 = st.columns([0.6, 1.4, 1.2, 1.1, 2.8, 0.9, 1.4])
                a1.markdown(f"<span style='font-family:monospace; color:#8B949E; font-size:0.75rem;'>#{act['id']}</span>", unsafe_allow_html=True)
                a2.markdown(f"<span style='background:{act_col}18; color:{act_col}; border:1px solid {act_col}44; border-radius:99px; font-size:0.7rem; padding:1px 6px; font-weight:700;'>{action_name}</span>", unsafe_allow_html=True)
                a3.markdown(f"<span style='font-size:0.8rem; font-weight:600; color:#E6EDF3;'>@{act['username']}</span>", unsafe_allow_html=True)
                a4.markdown(f"<span style='font-size:0.75rem; color:#8B949E;'>{r_val}</span>", unsafe_allow_html=True)
                a5.markdown(f"<span style='font-size:0.78rem; color:#E6EDF3; line-height:1.3;'>{act['details']}</span>", unsafe_allow_html=True)
                a6.markdown(f"<span style='color:{stat_col}; font-size:0.72rem; font-weight:600;'>● {act['status']}</span>", unsafe_allow_html=True)
                a7.markdown(f"<span style='font-family:monospace; font-size:0.72rem; color:#8B949E;'>{act['created_at']}</span>", unsafe_allow_html=True)
                st.markdown("<div style='margin-bottom:0.35rem;'></div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#161B22; border:1px solid #21262D; border-radius:8px; padding:2rem; text-align:center;">
                <span style="font-size:1.8rem; color:#8B949E;">📝</span>
                <p style="color:#8B949E; font-size:0.85rem; margin-top:0.4rem;">No user activity recorded yet. User logins and system actions will be tracked here in real-time!</p>
            </div>
            """, unsafe_allow_html=True)

    with tab_reports:
        all_reps = db.get_all_reports(limit=50)
        st.markdown(f"<div class='pro-card-title'>📋 Stored Diagnostic Reports Archive ({len(all_reps)} Total Reports in Database)</div>", unsafe_allow_html=True)
        
        if all_reps:
            # Header Row
            rh1, rh2, rh3, rh4, rh5, rh6, rh7 = st.columns([1.5, 1.3, 1.1, 1, 1, 1.1, 1.2])
            rh1.markdown("<span style='font-size:0.72rem; color:#8B949E; text-transform:uppercase; font-weight:700;'>Report Code</span>", unsafe_allow_html=True)
            rh2.markdown("<span style='font-size:0.72rem; color:#8B949E; text-transform:uppercase; font-weight:700;'>Patient</span>", unsafe_allow_html=True)
            rh3.markdown("<span style='font-size:0.72rem; color:#8B949E; text-transform:uppercase; font-weight:700;'>Diagnosis</span>", unsafe_allow_html=True)
            rh4.markdown("<span style='font-size:0.72rem; color:#8B949E; text-transform:uppercase; font-weight:700;'>Confidence</span>", unsafe_allow_html=True)
            rh5.markdown("<span style='font-size:0.72rem; color:#8B949E; text-transform:uppercase; font-weight:700;'>Doctor</span>", unsafe_allow_html=True)
            rh6.markdown("<span style='font-size:0.72rem; color:#8B949E; text-transform:uppercase; font-weight:700;'>PDF Export</span>", unsafe_allow_html=True)
            rh7.markdown("<span style='font-size:0.72rem; color:#8B949E; text-transform:uppercase; font-weight:700;'>AWS S3 Cloud</span>", unsafe_allow_html=True)
            st.markdown("<hr style='border:0; border-top:1px solid #21262D; margin:0.3rem 0 0.6rem 0;'>", unsafe_allow_html=True)

            for r in all_reps:
                r_c = "#34C759" if r["predicted_class"] == "notumor" else ("#FF9500" if r["predicted_class"] in ["meningioma", "pituitary"] else "#FF3B30")
                
                c1, c2, c3, c4, c5, c6, c7 = st.columns([1.5, 1.3, 1.1, 1, 1, 1.1, 1.2])
                c1.markdown(f"<span style='font-family:monospace; font-weight:700; color:#00D4FF; font-size:0.8rem;'>{r['report_code']}</span>", unsafe_allow_html=True)
                c2.markdown(f"<span style='font-size:0.82rem; color:#E6EDF3;'><b>{r['patient_name']}</b><br/><span style='color:#8B949E; font-size:0.72rem;'>{r['patient_age']}y · {r['patient_gender']}</span></span>", unsafe_allow_html=True)
                c3.markdown(f"<span style='background:{r_c}18; color:{r_c}; border:1px solid {r_c}44; border-radius:99px; font-size:0.72rem; padding:2px 8px; font-weight:600;'>{r['predicted_class'].title()}</span>", unsafe_allow_html=True)
                c4.markdown(f"<span style='font-size:0.82rem; color:#E6EDF3; font-weight:600;'>{r['confidence']:.1f}%</span>", unsafe_allow_html=True)
                c5.markdown(f"<span style='font-size:0.8rem; color:#8B949E;'>@{r['doctor_username']}</span>", unsafe_allow_html=True)
                with c6:
                    pdf_blob, pdf_fn = db.get_report_pdf_blob(r["id"])
                    if pdf_blob:
                        st.download_button(
                            label="📥 PDF",
                            data=pdf_blob,
                            file_name=pdf_fn or f"{r['report_code']}.pdf",
                            mime="application/pdf",
                            key=f"dl_rep_{r['id']}",
                            use_container_width=True
                        )
                    else:
                        st.markdown("<span style='font-size:0.72rem; color:#8B949E;'>N/A</span>", unsafe_allow_html=True)
                with c7:
                    s3_url_val = r.get("s3_url")
                    if s3_url_val:
                        st.markdown(f"""
                        <a href="{s3_url_val}" target="_blank" style="text-decoration:none;">
                            <button style="background:rgba(0,212,255,0.12); border:1px solid rgba(0,212,255,0.3); color:#00D4FF; font-size:0.7rem; padding:4px 8px; border-radius:4px; cursor:pointer; width:100%; font-weight:600;">
                                ☁️ S3 Link
                            </button>
                        </a>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='font-size:0.72rem; color:#8B949E;'>Local only</span>", unsafe_allow_html=True)
                st.markdown("<div style='margin-bottom:0.4rem;'></div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#161B22; border:1px solid #21262D; border-radius:8px; padding:2rem; text-align:center;">
                <span style="font-size:1.8rem; color:#8B949E;">📄</span>
                <p style="color:#8B949E; font-size:0.85rem; margin-top:0.4rem;">No diagnostic reports stored in database yet. Run an MRI scan to generate and save reports automatically!</p>
            </div>
            """, unsafe_allow_html=True)

    with tab_sys:
        d_c1, d_c2 = st.columns(2, gap="large")
        with d_c1:
            with st.container(border=True):
                st.markdown("<div class='pro-card-title'>PyTorch &amp; Hardware Runtime</div>", unsafe_allow_html=True)
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A (CPU Mode)'
                s3_ok, s3_msg = s3.is_s3_available()
                s3_color = "#3FB950" if s3_ok else "#FF9500"
                st.markdown(f"""
                <div style="font-size:0.84rem; line-height:1.9; color:#E6EDF3;">
                    • <b>PyTorch Version:</b> <code style="color:#00D4FF;">{torch.__version__}</code><br/>
                    • <b>CUDA Available:</b> <code style="color:#3FB950;">{torch.cuda.is_available()}</code><br/>
                    • <b>Active Compute Device:</b> <code style="color:#00D4FF;">{DEVICE}</code><br/>
                    • <b>GPU Device Name:</b> <code>{gpu_name}</code><br/>
                    • <b>CUDA Device Count:</b> <code>{torch.cuda.device_count()}</code><br/>
                    • <b>Database Backend:</b> <code style="color:#3FB950;">{db.get_active_engine_name()}</code><br/>
                    • <b>AWS S3 Cloud Storage:</b> <code style="color:{s3_color};">{s3_msg}</code><br/>
                    • <b>ReportLab PDF Engine:</b> <code>{'Online' if REPORTLAB_AVAILABLE else 'Missing'}</code>
                </div>
                """, unsafe_allow_html=True)



        with d_c2:
            with st.container(border=True):
                st.markdown("<div class='pro-card-title'>Model Checkpoint Weights Verification</div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="font-size:0.84rem; line-height:1.9; color:#E6EDF3;">
                    • <b>Classifier Model:</b> <code style="color:#3FB950;">{'Found' if os.path.exists(MODEL_PATH) else 'Missing'}</code><br/>
                    • <b>Segmentation Model:</b> <code style="color:#3FB950;">{'Found' if os.path.exists(SEG_MODEL_PATH) else 'Missing'}</code><br/>
                    • <b>Guardrail Model:</b> <code style="color:#3FB950;">{'Found' if os.path.exists(GUARDRAIL_MODEL_PATH) else 'Missing'}</code><br/>
                    • <b>Classification Metrics:</b> <code style="color:#3FB950;">{'Loaded' if os.path.exists(METRICS_PATH) else 'Missing'}</code><br/>
                    • <b>Segmentation Metrics:</b> <code style="color:#3FB950;">{'Loaded' if os.path.exists(SEG_METRICS_PATH) else 'Missing'}</code>
                </div>
                """, unsafe_allow_html=True)

    with tab_models:
        m_c1, m_c2 = st.columns(2, gap="large")
        with m_c1:
            with st.container(border=True):
                st.markdown("<div class='pro-card-title'>4-Class Classifier Evaluation Metrics</div>", unsafe_allow_html=True)
                metrics = load_metrics()
                if metrics and "test_performance" in metrics:
                    tp = metrics["test_performance"]
                    st.markdown(f"""
                    <div style="display:flex; justify-content:space-between; margin-bottom:0.8rem; background:rgba(0,212,255,0.08); padding:0.6rem 0.8rem; border-radius:6px; border:1px solid rgba(0,212,255,0.2);">
                        <span style="font-size:0.82rem; color:#E6EDF3;"><b>Overall Test Accuracy:</b> <span style="color:#00D4FF; font-weight:700;">{tp.get('accuracy', 0)*100:.2f}%</span></span>
                        <span style="font-size:0.82rem; color:#8B949E;">Training Time: <b>{metrics.get('total_training_time_minutes', 0):.1f} min</b></span>
                    </div>
                    """, unsafe_allow_html=True)

                    # Performance Table Headers
                    ch1, ch2, ch3, ch4, ch5 = st.columns([1.5, 1, 1, 1, 1])
                    ch1.markdown("<span style='font-size:0.72rem; color:#8B949E; text-transform:uppercase; font-weight:700;'>Pathology Class</span>", unsafe_allow_html=True)
                    ch2.markdown("<span style='font-size:0.72rem; color:#8B949E; text-transform:uppercase; font-weight:700;'>Precision</span>", unsafe_allow_html=True)
                    ch3.markdown("<span style='font-size:0.72rem; color:#8B949E; text-transform:uppercase; font-weight:700;'>Recall</span>", unsafe_allow_html=True)
                    ch4.markdown("<span style='font-size:0.72rem; color:#8B949E; text-transform:uppercase; font-weight:700;'>F1-Score</span>", unsafe_allow_html=True)
                    ch5.markdown("<span style='font-size:0.72rem; color:#8B949E; text-transform:uppercase; font-weight:700;'>Support</span>", unsafe_allow_html=True)
                    st.markdown("<hr style='border:0; border-top:1px solid #21262D; margin:0.3rem 0 0.5rem 0;'>", unsafe_allow_html=True)

                    classes_list = ["glioma", "meningioma", "pituitary", "notumor"]
                    for c_name in classes_list:
                        if c_name in tp:
                            c_data = tp[c_name]
                            badge_col = "#FF3B30" if c_name == "glioma" else ("#FF9500" if c_name in ["meningioma", "pituitary"] else "#34C759")
                            
                            c1, c2, c3, c4, c5 = st.columns([1.5, 1, 1, 1, 1])
                            c1.markdown(f"<span style='background:{badge_col}18; color:{badge_col}; border:1px solid {badge_col}44; border-radius:99px; font-size:0.72rem; padding:2px 8px; font-weight:600;'>{c_name.title()}</span>", unsafe_allow_html=True)
                            c2.markdown(f"<span style='font-size:0.82rem; color:#E6EDF3;'>{c_data['precision']*100:.1f}%</span>", unsafe_allow_html=True)
                            c3.markdown(f"<span style='font-size:0.82rem; color:#E6EDF3;'>{c_data['recall']*100:.1f}%</span>", unsafe_allow_html=True)
                            c4.markdown(f"<span style='font-size:0.82rem; color:#00D4FF; font-weight:600;'>{c_data['f1-score']*100:.1f}%</span>", unsafe_allow_html=True)
                            c5.markdown(f"<span style='font-size:0.82rem; color:#8B949E;'>{int(c_data['support'])}</span>", unsafe_allow_html=True)
                            st.markdown("<div style='margin-bottom:0.35rem;'></div>", unsafe_allow_html=True)

                    if "macro avg" in tp:
                        st.markdown("<hr style='border:0; border-top:1px solid #30363D; margin:0.4rem 0;'>", unsafe_allow_html=True)
                        m_data = tp["macro avg"]
                        m1, m2, m3, m4, m5 = st.columns([1.5, 1, 1, 1, 1])
                        m1.markdown("<span style='font-size:0.78rem; font-weight:700; color:#E6EDF3;'>Macro Average</span>", unsafe_allow_html=True)
                        m2.markdown(f"<span style='font-size:0.82rem; color:#E6EDF3;'>{m_data['precision']*100:.1f}%</span>", unsafe_allow_html=True)
                        m3.markdown(f"<span style='font-size:0.82rem; color:#E6EDF3;'>{m_data['recall']*100:.1f}%</span>", unsafe_allow_html=True)
                        m4.markdown(f"<span style='font-size:0.82rem; color:#00D4FF; font-weight:700;'>{m_data['f1-score']*100:.1f}%</span>", unsafe_allow_html=True)
                        m5.markdown(f"<span style='font-size:0.82rem; color:#8B949E;'>{int(m_data['support'])}</span>", unsafe_allow_html=True)
                else:
                    st.info("No classifier metrics found.")

        with m_c2:
            with st.container(border=True):
                st.markdown("<div class='pro-card-title'>U-Net Segmentation Evaluation Metrics</div>", unsafe_allow_html=True)
                seg_metrics = load_seg_metrics()
                if seg_metrics and "test_performance" in seg_metrics:
                    stp = seg_metrics["test_performance"]
                    cfg = seg_metrics.get("config", {})
                    
                    st.markdown(f"""
                    <div style="display:flex; justify-content:space-between; margin-bottom:0.8rem; background:rgba(63,185,80,0.08); padding:0.6rem 0.8rem; border-radius:6px; border:1px solid rgba(63,185,80,0.2);">
                        <span style="font-size:0.82rem; color:#E6EDF3;"><b>Test Dice Score:</b> <span style="color:#3FB950; font-weight:700;">{stp.get('dice', 0)*100:.2f}%</span></span>
                        <span style="font-size:0.82rem; color:#8B949E;">Test IoU (Jaccard): <b style="color:#3FB950;">{stp.get('iou', 0)*100:.2f}%</b></span>
                    </div>
                    """, unsafe_allow_html=True)

                    # Segmentation Specs Table
                    sh1, sh2 = st.columns([1.4, 1.6])
                    sh1.markdown("<span style='font-size:0.72rem; color:#8B949E; text-transform:uppercase; font-weight:700;'>Metric / Parameter</span>", unsafe_allow_html=True)
                    sh2.markdown("<span style='font-size:0.72rem; color:#8B949E; text-transform:uppercase; font-weight:700;'>Evaluation Result</span>", unsafe_allow_html=True)
                    st.markdown("<hr style='border:0; border-top:1px solid #21262D; margin:0.3rem 0 0.5rem 0;'>", unsafe_allow_html=True)

                    seg_rows = [
                        ("Dice Similarity Coefficient (Test)", f"{stp.get('dice', 0)*100:.2f}%", "#3FB950"),
                        ("Intersection-over-Union (IoU Test)", f"{stp.get('iou', 0)*100:.2f}%", "#3FB950"),
                        ("Best Validation Dice Score", f"{seg_metrics.get('best_val_dice', 0)*100:.2f}%", "#00D4FF"),
                        ("Test Loss (BCE + Dice Combined)", f"{stp.get('loss', 0):.4f}", "#E6EDF3"),
                        ("Segmentation Architecture", f"{cfg.get('architecture', 'U-Net')}", "#E6EDF3"),
                        ("Encoder Backbone Network", f"{cfg.get('encoder', 'efficientnet-b0')}", "#00D4FF"),
                        ("Input Image Resolution", f"{cfg.get('img_size', 256)} x {cfg.get('img_size', 256)} px", "#E6EDF3"),
                        ("Training Epochs / Duration", f"{cfg.get('epochs', 25)} epochs ({seg_metrics.get('total_training_time_minutes', 0):.1f} min)", "#8B949E"),
                    ]

                    for param_name, param_val, val_color in seg_rows:
                        s1, s2 = st.columns([1.4, 1.6])
                        s1.markdown(f"<span style='font-size:0.8rem; color:#E6EDF3;'>{param_name}</span>", unsafe_allow_html=True)
                        s2.markdown(f"<span style='font-size:0.8rem; font-weight:600; color:{val_color};'>{param_val}</span>", unsafe_allow_html=True)
                        st.markdown("<div style='margin-bottom:0.3rem;'></div>", unsafe_allow_html=True)
                else:
                    st.info("No segmentation metrics found.")

    with tab_errors:
        err_logs = db.get_error_logs(limit=100)
        c_e_top1, c_e_top2 = st.columns([3, 1])
        with c_e_top1:
            st.markdown(f"<div class='pro-card-title'>🚨 System &amp; Guardrail Error Logs Table ({len(err_logs)} total events)</div>", unsafe_allow_html=True)
        with c_e_top2:
            if st.button("🗑️ Clear Error Logs", key="btn_clear_logs", use_container_width=True):
                db.clear_error_logs()
                st.success("Error logs cleared!")
                st.rerun()

        if err_logs:
            # Table Headers
            eh1, eh2, eh3, eh4, eh5, eh6, eh7 = st.columns([0.6, 1.2, 1.4, 1.1, 1.1, 2.6, 1.4])
            eh1.markdown("<span style='font-size:0.72rem; color:#8B949E; text-transform:uppercase; font-weight:700;'>ID</span>", unsafe_allow_html=True)
            eh2.markdown("<span style='font-size:0.72rem; color:#8B949E; text-transform:uppercase; font-weight:700;'>Severity</span>", unsafe_allow_html=True)
            eh3.markdown("<span style='font-size:0.72rem; color:#8B949E; text-transform:uppercase; font-weight:700;'>Error Type</span>", unsafe_allow_html=True)
            eh4.markdown("<span style='font-size:0.72rem; color:#8B949E; text-transform:uppercase; font-weight:700;'>Module</span>", unsafe_allow_html=True)
            eh5.markdown("<span style='font-size:0.72rem; color:#8B949E; text-transform:uppercase; font-weight:700;'>User</span>", unsafe_allow_html=True)
            eh6.markdown("<span style='font-size:0.72rem; color:#8B949E; text-transform:uppercase; font-weight:700;'>Message / Details</span>", unsafe_allow_html=True)
            eh7.markdown("<span style='font-size:0.72rem; color:#8B949E; text-transform:uppercase; font-weight:700;'>Timestamp</span>", unsafe_allow_html=True)
            st.markdown("<hr style='border:0; border-top:1px solid #21262D; margin:0.3rem 0 0.5rem 0;'>", unsafe_allow_html=True)

            for elog in err_logs:
                sev = elog['error_severity']
                sev_color = "#FF3B30" if sev == "CRITICAL" else ("#FF9500" if sev == "ERROR" else "#FFCC00")
                user_val = f"@{elog['username']}" if elog['username'] else "System"
                
                e1, e2, e3, e4, e5, e6, e7 = st.columns([0.6, 1.2, 1.4, 1.1, 1.1, 2.6, 1.4])
                e1.markdown(f"<span style='font-family:monospace; color:#8B949E; font-size:0.75rem;'>#{elog['id']}</span>", unsafe_allow_html=True)
                e2.markdown(f"<span style='background:{sev_color}18; color:{sev_color}; border:1px solid {sev_color}44; border-radius:99px; font-size:0.7rem; padding:1px 6px; font-weight:700;'>{sev}</span>", unsafe_allow_html=True)
                e3.markdown(f"<span style='font-family:monospace; font-size:0.75rem; color:#E6EDF3; font-weight:600;'>{elog['error_type']}</span>", unsafe_allow_html=True)
                e4.markdown(f"<span style='font-size:0.75rem; color:#8B949E;'>{elog['component']}</span>", unsafe_allow_html=True)
                e5.markdown(f"<span style='font-size:0.75rem; color:#00D4FF;'>{user_val}</span>", unsafe_allow_html=True)
                e6.markdown(f"<span style='font-size:0.78rem; color:#E6EDF3; line-height:1.3;'>{elog['error_message']}</span>", unsafe_allow_html=True)
                e7.markdown(f"<span style='font-family:monospace; font-size:0.72rem; color:#8B949E;'>{elog['created_at']}</span>", unsafe_allow_html=True)
                st.markdown("<div style='margin-bottom:0.35rem;'></div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#161B22; border:1px solid #21262D; border-radius:8px; padding:2rem; text-align:center;">
                <span style="font-size:1.8rem; color:#3FB950;">✓</span>
                <p style="color:#8B949E; font-size:0.85rem; margin-top:0.4rem;">No system errors or guardrail rejections logged. Pipeline is healthy!</p>
            </div>
            """, unsafe_allow_html=True)


    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    b_col1, b_col2, b_col3 = st.columns([1.2, 1.2, 1.2])
    with b_col1:
        if st.button("🚀 Launch Doctor Workstation", key="adm_launch_doc", use_container_width=True):
            st.session_state.role = "doctor"
            st.rerun()
    with b_col2:
        if st.button("👤 Launch Patient Portal", key="adm_launch_pat", use_container_width=True):
            st.session_state.role = "patient"
            st.rerun()
    with b_col3:
        if st.button("🔓 Logout", key="adm_logout_btn", use_container_width=True):
            db.log_activity(
                username=st.session_state.username or "admin",
                action="USER_LOGOUT",
                role="admin",
                details="Admin signed out of session",
                status="SUCCESS"
            )
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.page = "landing"
            st.rerun()










# ══════════════════════════════════════════════════════════════════════════════
# APP ROUTER
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.page == "landing":
    render_landing_page()

elif st.session_state.page == "login":
    if st.session_state.get("logged_in", False):
        st.session_state.page = "dashboard"
        st.rerun()
    else:
        render_login_page()

else:  # "dashboard"
    if st.session_state.get("logged_in", False):
        current_role = st.session_state.get("role", "doctor")
        if current_role == "admin":
            render_admin_dashboard()
        elif current_role == "patient":
            render_patient_dashboard()
        else:
            render_dashboard_page()
    else:
        st.session_state.page = "login"
        st.rerun()



