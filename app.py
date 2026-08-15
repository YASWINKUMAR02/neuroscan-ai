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

# ── CSS (ProHealth Premium UI Theme) ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* Main Overrides */
html, body, [class*="css"] {
    font-family: 'Inter', 'Outfit', sans-serif;
    color: #1A253C;
}

.stApp {
    background-color: #F4F6F9 !important;
}

/* Eliminate Streamlit Header Spacing & Layout Padding */
[data-testid="stHeader"] {
    display: none !important;
}
.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 0.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* Hide Default Streamlit Widgets */
header {visibility: hidden;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Custom Top Bar */
.top-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.4rem 0;
    margin-bottom: 0.75rem;
    border-bottom: 1px solid #E5E9F0;
}
.logo-container {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-family: 'Outfit', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #0066FF;
}

/* Styling Streamlit Containers as Cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF !important;
    border: 1px solid #EAF0F6 !important;
    border-radius: 16px !important;
    padding: 0.85rem !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.01) !important;
    margin-bottom: 0.75rem !important;
}

/* Card Titles */
.pro-card-title {
    font-family: 'Outfit', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: #1A253C;
    margin-bottom: 0.5rem;
}

/* File Uploader styling */
div[data-testid="stFileUploader"] {
    background-color: #F8FAFC !important;
    border: 1px dashed #DDE3EA !important;
    border-radius: 10px !important;
    padding: 0.4rem !important;
}

/* Pipeline steps */
.pipe-step-card {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.5rem 0.65rem;
    border-radius: 10px;
    margin-bottom: 0.4rem;
    background-color: #F8FAFC;
    border: 1px solid #EAF0F6;
    transition: all 0.2s ease;
}
.pipe-step-card.active {
    background-color: #F0F6FF;
    border-color: #0066FF;
}
.pipe-step-card.done {
    background-color: #F6FFF9;
    border-color: #10B981;
}
.pipe-step-badge {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.7rem;
}
.pipe-step-badge.active { background-color: #0066FF; color: #FFFFFF; }
.pipe-step-badge.done { background-color: #10B981; color: #FFFFFF; }
.pipe-step-badge.todo { background-color: #E2E8F0; color: #7C8BA1; }

.pipe-step-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #1A253C;
}

/* Circular Metric Indicator */
.circle-progress-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.6rem;
}
.circle-progress {
    position: relative;
    width: 56px;
    height: 56px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
}
.circle-val {
    position: relative;
    z-index: 10;
    font-size: 0.95rem;
    font-weight: 700;
    color: #1A253C;
    font-family: 'JetBrains Mono', monospace;
}

/* Custom horizontal progress bars */
.bar-wrap {
    width: 100%;
    background-color: #F1F5F9;
    height: 6px;
    border-radius: 99px;
    overflow: hidden;
    margin-top: 0.25rem;
}
.bar-fill {
    height: 100%;
    border-radius: 99px;
}

/* Viewport Image Reticle Container */
.viewport-frame {
    background-color: #FFFFFF;
    border-radius: 12px;
    overflow: hidden;
}

.viewport-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #7C8BA1;
    margin-top: 0.5rem;
    display: flex;
    justify-content: space-between;
}

/* Metric overrides */
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: #1A253C !important;
}
[data-testid="stMetricLabel"] {
    text-transform: uppercase !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.05em !important;
    color: #7C8BA1 !important;
}

.eyebrow {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #7C8BA1;
    margin-bottom: 0.4rem;
}

.disclaimer-text {
    font-size: 0.72rem;
    color: #7C8BA1;
    margin-top: 1.5rem;
    line-height: 1.4;
    text-align: center;
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

# ── Model Loaders ─────────────────────────────────────────────────────────────
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
def is_mri_image(pil_img: Image.Image):
    img_rgb  = np.array(pil_img.convert("RGB"))
    img_gray = np.array(pil_img.convert("L"))
    img_hsv  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mean_sat = img_hsv[:, :, 1].mean()
    if mean_sat > 60:
        return False, f"High saturation ({mean_sat:.1f}/255). MRI is greyscale."
    mean_brightness = img_gray.mean()
    if mean_brightness > 200:
        return False, f"Image too bright ({mean_brightness:.1f}/255)."
    dark_ratio = (img_gray < 40).sum() / img_gray.size
    if dark_ratio < 0.15:
        return False, f"Low dark background ratio ({dark_ratio*100:.1f}%)."
    bright_ratio = (img_gray > 60).sum() / img_gray.size
    if bright_ratio < 0.05:
        return False, f"Almost entirely black ({bright_ratio*100:.1f}%)."
    return True, "Valid scan format"

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

# ── Load Models ───────────────────────────────────────────────────────────────
with st.spinner("Initializing models..."):
    clf_model = load_classifier()
    seg_model, seg_err = load_segmentation_model()
    metrics = load_metrics()
    seg_metrics = load_seg_metrics()

# ── Top Bar Header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
    <div class="logo-container">
        <span>🌐</span> ProHealth TumorOI OS
    </div>
</div>
""", unsafe_allow_html=True)

# ── Layout Grid ───────────────────────────────────────────────────────────────
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

# LEFT COLUMN: Pipeline & File Upload
with col_left:
    st.markdown("<h2 style='font-family: Outfit, sans-serif; font-weight: 700; color: #1A253C; margin-bottom: 0.2rem; font-size: 1.6rem;'>Overview</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #7C8BA1; font-size: 0.9rem; margin-bottom: 1.5rem;'>Patient Health Scan</p>", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown("<div class='pro-card-title'>Upload Diagnostics</div>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload MRI", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# Automatic Pipeline Execution
if uploaded:
    step = 1
    pil_img = Image.open(uploaded)
    is_mri, mri_reason = is_mri_image(pil_img)
    if is_mri:
        step = 2
        predicted_class, probs = predict_class(clf_model, pil_img)
        has_tumor = predicted_class in TUMOR_CLASSES
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

    if uploaded and not is_mri and mri_reason != "":
        st.error(f"Image Rejected: {mri_reason}")

# CENTER COLUMN: Viewport Grid (Original, Overlay, Heatmap shown together)
with col_center:
    st.markdown("<h2 style='font-family: Outfit, sans-serif; font-weight: 700; color: #1A253C; margin-bottom: 0.2rem; font-size: 1.6rem;'>AI Viewport</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #7C8BA1; font-size: 0.9rem; margin-bottom: 1.5rem;'>Pathology Visualizations</p>", unsafe_allow_html=True)

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

                # Second Row: Heatmap (using 2 columns to match the top row's size)
                st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
                view_c3, view_c4 = st.columns(2, gap="medium")
                with view_c3:
                    with st.container(border=True):
                        st.markdown("<div class='pro-card-title'>Probability Heatmap (Explanation)</div>", unsafe_allow_html=True)
                        prob_norm = (prob_map * 255).astype(np.uint8)
                        heatmap = cv2.applyColorMap(prob_norm, cv2.COLORMAP_JET)
                        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                        st.image(heatmap_rgb, use_container_width=True)
                with view_c4:
                    # Spacing placeholder to keep heatmap sized at exactly 50% width
                    pass
            else:
                # Normal scan with no tumor: just show original
                with st.container(border=True):
                    st.markdown("<div class='pro-card-title'>Original MRI Scan</div>", unsafe_allow_html=True)
                    st.image(pil_img, use_container_width=True)
                st.info("No pathology detected. Segmentation skipped.")

            w, h = pil_img.size
            st.markdown(f"""
            <div class="viewport-label">
                <span>Dimensions: {w} × {h} px</span>
                <span>File: {uploaded.name}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning(f"MRI guardrail failed: {mri_reason}")
    else:
        st.markdown("""
        <div style="background-color: #FFFFFF; border: 1px solid #EAF0F6; border-radius: 24px; padding: 6rem 2rem; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.02); height: 100%;">
            <span style="font-size: 4rem;">📂</span>
            <h4 style="font-family: 'Outfit', sans-serif; font-weight: 600; color: #1A253C; margin-top: 1rem;">Diagnostics Queue Empty</h4>
            <p style="color: #7C8BA1; font-size: 0.9rem;">Upload a brain MRI scan from the left panel to trigger the model pipeline.</p>
        </div>
        """, unsafe_allow_html=True)

# RIGHT COLUMN: Readouts & Results
with col_right:
    st.markdown("<h2 style='font-family: Outfit, sans-serif; font-weight: 700; color: #1A253C; margin-bottom: 0.2rem; font-size: 1.6rem;'>Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #7C8BA1; font-size: 0.9rem; margin-bottom: 1.5rem;'>Diagnosis & Statistics</p>", unsafe_allow_html=True)

    if step >= 2 and probs is not None:
        # Findings Card
        with st.container(border=True):
            st.markdown("<div class='pro-card-title'>Diagnosis</div>", unsafe_allow_html=True)
            
            class_info_map = {
                "glioma": ("Glioma Findings", "#0066FF", "#FF3B30"),
                "meningioma": ("Meningioma Findings", "#0066FF", "#FF9500"),
                "pituitary": ("Pituitary Findings", "#0066FF", "#5AC8FA"),
                "notumor": ("No Tumor Detected", "#0066FF", "#34C759")
            }
            lbl, theme_color, stroke_color = class_info_map[predicted_class]
            conf = probs[CLASSES.index(predicted_class)] * 100

            # Render circular progress indicator at top right of findings
            st.markdown(f"""
            <div class="circle-progress-container">
                <div>
                    <h3 style="font-family: 'Outfit', sans-serif; font-weight: 700; color: #1A253C; margin: 0; font-size: 1.3rem;">{lbl}</h3>
                    <p style="color: #7C8BA1; font-size: 0.8rem; margin: 0; margin-top: 0.2rem;">Classification Confidence</p>
                </div>
                <div class="circle-progress" style="background: conic-gradient({stroke_color} calc({conf} * 1%), #F1F5F9 0);">
                    <div class="circle-val">{conf:.0f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probabilities block
            st.markdown("<p style='font-size: 0.85rem; font-weight: 600; color: #1A253C; margin-top: 1rem; margin-bottom: 0.5rem;'>Other Class Distribution</p>", unsafe_allow_html=True)
            for c, p in sorted(zip(CLASSES, probs), key=lambda x: -x[1]):
                if c == predicted_class: continue
                p_pct = p * 100
                c_lbl = class_info_map[c][0].replace(" Findings", "").replace(" Detected", "")
                bar_color = class_info_map[c][2]
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; margin-bottom: 0.25rem;">
                    <span style="color: #7C8BA1;">{c_lbl}</span>
                    <span class="mono-stat" style="font-weight: 600; color: #1A253C;">{p_pct:.1f}%</span>
                </div>
                <div class="bar-wrap" style="height: 5px; margin-bottom: 0.75rem;">
                    <div class="bar-fill" style="width: {p_pct}%; background-color: {bar_color};"></div>
                </div>
                """, unsafe_allow_html=True)

        # Metrics Card
        with st.container(border=True):
            st.markdown("<div class='pro-card-title'>Segmentation Metrics</div>", unsafe_allow_html=True)
            
            if step == 4 and seg_metrics and seg_model and has_tumor and binary_mask is not None:
                dice = seg_metrics["test_performance"]["dice"]
                iou = seg_metrics["test_performance"]["iou"]
                cov = 100.0 * binary_mask.sum() / binary_mask.size
                px = int(binary_mask.sum())
                
                c1, c2 = st.columns(2)
                c1.metric("Dice Similarity", f"{dice:.3f}")
                c2.metric("IoU Score", f"{iou:.3f}")
                c1.metric("Tumor Coverage", f"{cov:.1f}%")
                c2.metric("Active Pixels", f"{px:,}")
            else:
                c1, c2 = st.columns(2)
                c1.metric("Dice Similarity", "--")
                c2.metric("IoU Score", "--")
                c1.metric("Tumor Coverage", "--")
                c2.metric("Active Pixels", "--")
    else:
        with st.container(border=True):
            st.markdown("<div style='color: #7C8BA1; font-size: 0.85rem; text-align: center; padding: 2rem 0;'>Awaiting scan input to run diagnosis...</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-text">
        🔒 Research use only. Not for primary diagnosis. Patient data encrypted.
    </div>
    """, unsafe_allow_html=True)
