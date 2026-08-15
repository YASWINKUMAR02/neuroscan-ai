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
gradcam_img = None
gradcam_raw = None

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

                # Second Row: Segmentation heatmap + Grad-CAM
                st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
                view_c3, view_c4 = st.columns(2, gap="medium")
                with view_c3:
                    with st.container(border=True):
                        st.markdown("<div class='pro-card-title'>Segmentation Heatmap</div>", unsafe_allow_html=True)
                        prob_norm = (prob_map * 255).astype(np.uint8)
                        heatmap = cv2.applyColorMap(prob_norm, cv2.COLORMAP_JET)
                        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                        st.image(heatmap_rgb, use_container_width=True)
                with view_c4:
                    with st.container(border=True):
                        st.markdown("<div class='pro-card-title'>Grad-CAM <span style='font-size:0.7rem;color:#7C8BA1;font-weight:400;'>(XAI · Classifier Attention)</span></div>", unsafe_allow_html=True)
                        if gradcam_img:
                            st.image(gradcam_img, use_container_width=True)
                        else:
                            st.caption("Grad-CAM unavailable.")
            else:
                # Normal scan with no tumor: show original + Grad-CAM
                view_n1, view_n2 = st.columns(2, gap="medium")
                with view_n1:
                    with st.container(border=True):
                        st.markdown("<div class='pro-card-title'>Original MRI Scan</div>", unsafe_allow_html=True)
                        st.image(pil_img, use_container_width=True)
                with view_n2:
                    with st.container(border=True):
                        st.markdown("<div class='pro-card-title'>Grad-CAM <span style='font-size:0.7rem;color:#7C8BA1;font-weight:400;'>(XAI · Classifier Attention)</span></div>", unsafe_allow_html=True)
                        if gradcam_img:
                            st.image(gradcam_img, use_container_width=True)
                        else:
                            st.caption("Grad-CAM unavailable.")
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
        # ── XAI Explainability Card ───────────────────────────────────────────
        with st.container(border=True):
            st.markdown("<div class='pro-card-title'>🔍 XAI · Grad-CAM</div>", unsafe_allow_html=True)
            if gradcam_img:
                st.image(gradcam_img, use_container_width=True)
                # Top activated region as text interpretation
                if gradcam_raw is not None:
                    cam_gray = np.array(gradcam_raw.convert("L")).astype(np.float32) / 255.0
                    h, w = cam_gray.shape
                    cy, cx = np.unravel_index(cam_gray.argmax(), cam_gray.shape)
                    # Quadrant label
                    vq = "Upper" if cy < h // 2 else "Lower"
                    hq = "Left"  if cx < w // 2 else "Right"
                    focus_pct = float(cam_gray[cam_gray >= 0.7].size / cam_gray.size * 100)
                    st.markdown(f"""
                    <div style="margin-top:0.5rem;">
                        <p style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;
                                  color:#7C8BA1; margin:0 0 0.3rem 0;">Classifier Focus Region</p>
                        <p style="font-family:'Outfit',sans-serif; font-size:0.95rem; font-weight:700;
                                  color:#1A253C; margin:0;">{vq}-{hq} Region</p>
                        <p style="font-size:0.78rem; color:#7C8BA1; margin:0.25rem 0 0.6rem 0;">
                            Peak activation at ({cx}px, {cy}px)
                        </p>
                        <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:0.2rem;">
                            <span style="color:#7C8BA1;">High-attention area (≥70%)</span>
                            <span style="font-weight:700; color:#0066FF;">{focus_pct:.1f}%</span>
                        </div>
                        <div class="bar-wrap">
                            <div class="bar-fill" style="width:{min(focus_pct,100):.1f}%; background:#0066FF;"></div>
                        </div>
                        <p style="font-size:0.70rem; color:#7C8BA1; margin-top:0.5rem; line-height:1.4;">
                            Red/yellow regions indicate areas the classifier weighted most heavily
                            when predicting <strong>{predicted_class.capitalize()}</strong>.
                        </p>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown("<div style='color:#7C8BA1;font-size:0.82rem;text-align:center;padding:0.75rem 0;'>Grad-CAM unavailable for this scan.</div>", unsafe_allow_html=True)

        # Findings Card
        with st.container(border=True):
            st.markdown("<div class='pro-card-title'>Diagnosis</div>", unsafe_allow_html=True)
            
            class_info_map = {
                "glioma": ("Glioma Findings", "#0066FF", "#FF3B30", "Malignant", "#FFEBEA"),
                "meningioma": ("Meningioma Findings", "#0066FF", "#FF9500", "Typically Benign", "#FFF3E0"),
                "pituitary": ("Pituitary Findings", "#0066FF", "#5AC8FA", "Typically Benign", "#E5F6FD"),
                "notumor": ("No Tumor Detected", "#0066FF", "#34C759", "Healthy", "#E8F5E9")
            }
            lbl, theme_color, stroke_color, status_lbl, status_bg = class_info_map[predicted_class]
            conf = probs[CLASSES.index(predicted_class)] * 100

            # Render circular progress indicator at top right of findings
            st.markdown(f"""
            <div class="circle-progress-container">
                <div>
                    <h3 style="font-family: 'Outfit', sans-serif; font-weight: 700; color: #1A253C; margin: 0; font-size: 1.3rem;">{lbl}</h3>
                    <div style="display: inline-block; background-color: {status_bg}; color: {stroke_color}; font-size: 0.72rem; font-weight: 700; padding: 2px 10px; border-radius: 99px; margin-top: 0.4rem; text-transform: uppercase; letter-spacing: 0.05em;">{status_lbl}</div>
                    <p style="color: #7C8BA1; font-size: 0.8rem; margin: 0; margin-top: 0.4rem;">Classification Confidence</p>
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
                status_lbl_other = class_info_map[c][3]
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; margin-bottom: 0.25rem;">
                    <span style="color: #7C8BA1;">{c_lbl} ({status_lbl_other})</span>
                    <span class="mono-stat" style="font-weight: 600; color: #1A253C;">{p_pct:.1f}%</span>
                </div>
                <div class="bar-wrap" style="height: 5px; margin-bottom: 0.75rem;">
                    <div class="bar-fill" style="width: {p_pct}%; background-color: {bar_color};"></div>
                </div>
                """, unsafe_allow_html=True)

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

        # ── Tumor Area Calculation Card ────────────────────────────────────────
        with st.container(border=True):
            st.markdown("<div class='pro-card-title'>🔬 Tumor Area</div>", unsafe_allow_html=True)
            if has_tumor and binary_mask is not None:
                px_count, total_px, cov_pct, area_mm2, area_cm2 = compute_tumor_area(binary_mask, pil_img)
                c1, c2 = st.columns(2)
                c1.metric("Area (mm²)",    f"{area_mm2:,.1f}")
                c2.metric("Area (cm²)",    f"{area_cm2:.2f}")
                c1.metric("Tumor Pixels",  f"{px_count:,}")
                c2.metric("Coverage",      f"{cov_pct:.2f}%")
                st.markdown(f"""
                <div style="font-size:0.72rem; color:#7C8BA1; margin-top:0.4rem;">
                    ℹ️ Estimated at 0.5 mm/pixel (standard MRI resolution)
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("<div style='color:#7C8BA1; font-size:0.82rem; text-align:center; padding:0.75rem 0;'>No tumor detected — area analysis unavailable.</div>", unsafe_allow_html=True)

        # ── Tumor Shape Analysis Card ──────────────────────────────────────────
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

                    # ── Prominent shape type header ──────────────────────────
                    st.markdown(f"""
                    <div style="margin-bottom:1rem;">
                        <p style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;
                                  color:#7C8BA1; margin:0 0 0.3rem 0;">Tumor Shape</p>
                        <div style="display:inline-flex; align-items:center; gap:0.5rem;">
                            <div style="width:10px; height:10px; border-radius:50%;
                                        background:{s_col};"></div>
                            <span style="font-family:'Outfit',sans-serif; font-size:1.4rem;
                                         font-weight:700; color:{s_col};">{s_lbl}</span>
                        </div>
                    </div>
                    <p style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;
                               color:#7C8BA1; margin:0 0 0.6rem 0;">Features</p>
                    """, unsafe_allow_html=True)

                    # ── Feature rows with bar ────────────────────────────────
                    def _feat_row(label, value, bar_pct, color):
                        st.markdown(f"""
                        <div style="margin-bottom:0.65rem;">
                            <div style="display:flex; justify-content:space-between;
                                        align-items:center; margin-bottom:0.2rem;">
                                <span style="font-size:0.82rem; color:#1A253C; font-weight:500;">{label}</span>
                                <span style="font-family:'JetBrains Mono',monospace; font-size:0.82rem;
                                             font-weight:700; color:#1A253C;">{value:.3f}</span>
                            </div>
                            <div class="bar-wrap">
                                <div class="bar-fill" style="width:{min(bar_pct,100):.1f}%;
                                     background:{color};"></div>
                            </div>
                        </div>""", unsafe_allow_html=True)

                    circ_col = "#34C759" if circ >= 0.60 else "#FF3B30"
                    comp_col = "#34C759" if comp >= 0.60 else "#FF3B30"
                    sol_col  = "#34C759" if sol  >= 0.90 else ("#FF9500" if sol  >= 0.75 else "#FF3B30")

                    _feat_row("Circularity",  circ, circ * 100, circ_col)
                    _feat_row("Compactness",  comp, comp * 100, comp_col)
                    _feat_row("Solidity",     sol,  sol  * 100, sol_col)
                else:
                    st.markdown("<div style='color:#7C8BA1;font-size:0.82rem;'>No contour found in mask.</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='color:#7C8BA1; font-size:0.82rem; text-align:center; padding:0.75rem 0;'>No tumor detected — shape analysis unavailable.</div>", unsafe_allow_html=True)

        # ── Segmentation Confidence Map Card ──────────────────────────────────
        with st.container(border=True):
            st.markdown("<div class='pro-card-title'>📊 Confidence Map</div>", unsafe_allow_html=True)
            if has_tumor and binary_mask is not None and prob_map is not None:
                conf = compute_confidence_stats(prob_map, binary_mask)
                if conf:
                    mean_pct = conf["mean"] * 100
                    max_pct  = conf["max"]  * 100
                    std_pct  = conf["std"]  * 100
                    hc       = conf["high_conf_pct"]

                    # Overall confidence level label
                    if mean_pct >= 80:   conf_lbl, conf_col = "High",     "#34C759"
                    elif mean_pct >= 55: conf_lbl, conf_col = "Moderate", "#FF9500"
                    else:                conf_lbl, conf_col = "Low",       "#FF3B30"

                    st.markdown(f"""
                    <div style="margin-bottom:1rem;">
                        <p style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;
                                  color:#7C8BA1; margin:0 0 0.3rem 0;">Confidence Level</p>
                        <div style="display:inline-flex; align-items:center; gap:0.5rem;">
                            <div style="width:10px; height:10px; border-radius:50%;
                                        background:{conf_col};"></div>
                            <span style="font-family:'Outfit',sans-serif; font-size:1.4rem;
                                         font-weight:700; color:{conf_col};">{conf_lbl}</span>
                        </div>
                    </div>
                    <p style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;
                               color:#7C8BA1; margin:0 0 0.6rem 0;">Features</p>
                    """, unsafe_allow_html=True)

                    def _conf_row(label, value_str, bar_pct, color):
                        st.markdown(f"""
                        <div style="margin-bottom:0.65rem;">
                            <div style="display:flex; justify-content:space-between;
                                        align-items:center; margin-bottom:0.2rem;">
                                <span style="font-size:0.82rem; color:#1A253C; font-weight:500;">{label}</span>
                                <span style="font-family:'JetBrains Mono',monospace; font-size:0.82rem;
                                             font-weight:700; color:#1A253C;">{value_str}</span>
                            </div>
                            <div class="bar-wrap">
                                <div class="bar-fill" style="width:{min(bar_pct,100):.1f}%;
                                     background:{color};"></div>
                            </div>
                        </div>""", unsafe_allow_html=True)

                    mean_col = "#34C759" if mean_pct >= 80 else ("#FF9500" if mean_pct >= 55 else "#FF3B30")
                    peak_col = "#34C759" if max_pct  >= 90 else ("#FF9500" if max_pct  >= 70 else "#FF3B30")
                    hc_col   = "#34C759" if hc       >= 70 else ("#FF9500" if hc       >= 40 else "#FF3B30")

                    _conf_row("Mean Confidence",       f"{mean_pct:.1f}%", mean_pct, mean_col)
                    _conf_row("Peak Confidence",       f"{max_pct:.1f}%",  max_pct,  peak_col)
                    _conf_row("High-conf Region (≥75%)", f"{hc:.1f}%",    hc,       hc_col)
            else:
                st.markdown("<div style='color:#7C8BA1; font-size:0.82rem; text-align:center; padding:0.75rem 0;'>No tumor detected — confidence map unavailable.</div>", unsafe_allow_html=True)

    else:
        with st.container(border=True):
            st.markdown("<div style='color: #7C8BA1; font-size: 0.85rem; text-align: center; padding: 2rem 0;'>Awaiting scan input to run diagnosis...</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-text">
        🔒 Research use only. Not for primary diagnosis. Patient data encrypted.
    </div>
    """, unsafe_allow_html=True)
