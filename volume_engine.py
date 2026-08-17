"""
volume_engine.py — 3D Volumetric NIfTI MRI Segmentation Engine for NeuroScan AI
Processes structural 3D NIfTI brain MRI scans (.nii, .nii.gz), performs volume-wide
slice-by-slice segmentation with U-Net (EfficientNet-B0), and computes physical voxel
volume (cm³ & mm³), voxel counts, and slice coverage matching clinical benchmarks.
"""

import os
import io
import tempfile
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False


def load_nifti_from_bytes(file_bytes: bytes, filename: str = "scan.nii") -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float, float]], Dict[str, Any]]:
    """
    Loads a 3D NIfTI file from memory bytes into a normalized 3D numpy array (W, H, Z)
    and extracts header voxel spacing (dx, dy, dz) in mm.
    """
    if not NIBABEL_AVAILABLE:
        raise ImportError("nibabel library is required for 3D NIfTI MRI files. Install with: pip install nibabel")

    suffix = ".nii.gz" if filename.endswith(".nii.gz") else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        nii_obj = nib.load(tmp_path)
        data = nii_obj.get_fdata(dtype=np.float32)
        header = nii_obj.header
        zooms = header.get_zooms()
        dx, dy, dz = float(zooms[0]), float(zooms[1]), float(zooms[2]) if len(zooms) >= 3 else (1.0, 1.0, 1.0)

        # Handle 4D volumes by taking first volume/timepoint if necessary
        if data.ndim == 4:
            data = data[..., 0]

        # 1st-99th percentile intensity clipping & uint8 normalization [0..255]
        p1, p99 = np.percentile(data, (1, 99))
        if p99 > p1:
            norm_vol = np.clip(data, p1, p99)
            norm_vol = (norm_vol - p1) / (p99 - p1)
        else:
            max_val = np.max(data)
            norm_vol = data / max_val if max_val > 0 else np.zeros_like(data)

        norm_vol = (norm_vol * 255.0).astype(np.uint8)

        metadata = {
            "shape": list(data.shape),
            "voxel_spacing_mm": (dx, dy, dz),
            "data_dtype": str(data.dtype),
            "total_slices": int(data.shape[2]) if data.ndim >= 3 else 1,
            "filename": filename
        }

        return norm_vol, (dx, dy, dz), metadata

    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def segment_3d_volume(
    volume: np.ndarray,
    voxel_spacing: Tuple[float, float, float],
    model: nn.Module,
    device: torch.device,
    batch_size: int = 16,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Runs slice-by-slice segmentation across all axial slices (Z-axis).
    Computes true physical volume in cm³ and mm³, positive voxel count, and slice coverage.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume (W, H, Z), got shape {volume.shape}")

    orig_w, orig_h, z_slices = volume.shape
    dx, dy, dz = voxel_spacing
    voxel_volume_mm3 = dx * dy * dz

    predicted_mask_3d = np.zeros((orig_w, orig_h, z_slices), dtype=np.uint8)
    slice_areas_px = []

    # ImageNet normalization statistics
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    model.eval()

    with torch.no_grad():
        for start_idx in range(0, z_slices, batch_size):
            end_idx = min(start_idx + batch_size, z_slices)

            batch_tensors = []
            for z in range(start_idx, end_idx):
                # Transpose for proper anatomical axial orientation
                slice_2d = volume[:, :, z].T
                img_resized = cv2.resize(slice_2d, (256, 256), interpolation=cv2.INTER_LINEAR)
                img_3c = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

                # Normalize with ImageNet mean & std
                img_norm = (img_3c.astype(np.float32) / 255.0 - mean) / std
                # Transpose to (3, 256, 256)
                tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()
                batch_tensors.append(tensor)

            batch_tensor = torch.stack(batch_tensors).to(device)
            logits = model(batch_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()  # (B, 1, 256, 256)

            for i, z in enumerate(range(start_idx, end_idx)):
                prob_map = probs[i, 0]
                bin_pred_256 = (prob_map > threshold).astype(np.uint8)
                # Resize back to (orig_w, orig_h) and transpose back
                pred_orig = cv2.resize(bin_pred_256, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).T
                predicted_mask_3d[:, :, z] = pred_orig
                slice_areas_px.append(int(np.sum(pred_orig)))

    # Metrics calculation
    tumor_voxels = int(np.sum(predicted_mask_3d))
    vol_mm3 = tumor_voxels * voxel_volume_mm3
    vol_cm3 = vol_mm3 / 1000.0

    affected_slice_indices = [z for z, count in enumerate(slice_areas_px) if count > 0]
    affected_slices_count = len(affected_slice_indices)
    slice_coverage_pct = (affected_slices_count / z_slices * 100.0) if z_slices > 0 else 0.0

    rep_z = int(np.argmax(slice_areas_px)) if affected_slices_count > 0 else z_slices // 2
    has_tumor = tumor_voxels > 0

    return {
        "has_tumor": has_tumor,
        "mask_3d": predicted_mask_3d,
        "tumor_voxel_count": tumor_voxels,
        "tumor_volume_mm3": vol_mm3,
        "tumor_volume_cm3": vol_cm3,
        "voxel_spacing_mm": (dx, dy, dz),
        "total_slices": z_slices,
        "affected_slices": affected_slices_count,
        "affected_slice_indices": affected_slice_indices,
        "slice_coverage_pct": slice_coverage_pct,
        "peak_slice_idx": rep_z,
        "peak_slice_voxel_count": int(slice_areas_px[rep_z]) if slice_areas_px else 0,
        "slice_areas_px": slice_areas_px
    }


def render_slice_triplet(
    volume: np.ndarray,
    mask_3d: np.ndarray,
    z: int
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """
    Renders the exact 3-panel representation matching standard medical visualization:
    1. Original MRI slice (grayscale)
    2. Predicted Tumor Mask (Reds colormap)
    3. Overlay (MRI slice with Autumn/Red overlay)
    """
    z_max = volume.shape[2] - 1
    z_safe = max(0, min(z, z_max))

    mri_slice = volume[:, :, z_safe].T
    mask_slice = mask_3d[:, :, z_safe].T if mask_3d is not None else np.zeros_like(mri_slice, dtype=np.uint8)

    # 1. Original MRI (Grayscale)
    img_orig = Image.fromarray(mri_slice.astype(np.uint8)).convert("RGB")

    # 2. Predicted Tumor Mask with 'Reds' colormap (cream background, dark crimson lesion)
    reds_cmap = plt.get_cmap("Reds")
    mask_rgba = (reds_cmap(mask_slice.astype(np.float32)) * 255.0).astype(np.uint8)
    img_mask = Image.fromarray(mask_rgba[:, :, :3])

    # 3. Overlay: Grayscale MRI + Autumn colored highlight for segmented tumor
    autumn_cmap = plt.get_cmap("autumn")
    overlay_rgba = (autumn_cmap(mask_slice.astype(np.float32)) * 255.0).astype(np.uint8)
    base_rgb = cv2.cvtColor(mri_slice.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
    overlay_res = base_rgb.copy()
    has_mask = mask_slice > 0
    if np.any(has_mask):
        overlay_res[has_mask] = (0.5 * base_rgb[has_mask] + 0.5 * overlay_rgba[has_mask, :3]).astype(np.uint8)
    img_overlay = Image.fromarray(overlay_res)

    return img_orig, img_mask, img_overlay
