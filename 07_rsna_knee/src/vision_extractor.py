"""
Lightweight Fast DICOM Vision & Metadata Extractor for RSNA Knee Studies.
Extracts clinical image intensity statistics (fluid fraction, percentiles, contrast)
and DICOM tags (sex, slice thickness, resolution) from central slices per series.
Runs in milliseconds per study to satisfy Kaggle Efficiency requirements.
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
try:
    import pydicom
except ImportError:
    pydicom = None


def extract_slice_features(dcm_path: str) -> dict:
    """
    Extracts intensity and acquisition features from a single DICOM slice.
    """
    if pydicom is None:
        return {}

    try:
        dcm = pydicom.dcmread(dcm_path, stop_before_pixels=False)
    except Exception:
        return {}

    feats = {}

    # 1. Metadata tags
    sex_str = str(getattr(dcm, "PatientSex", "U")).strip().upper()
    feats["is_female"] = 1.0 if sex_str == "F" else (0.0 if sex_str == "M" else 0.5)

    try:
        feats["slice_thickness"] = float(getattr(dcm, "SliceThickness", 3.0))
    except (ValueError, TypeError):
        feats["slice_thickness"] = 3.0

    try:
        feats["repetition_time"] = float(getattr(dcm, "RepetitionTime", 3000.0))
    except (ValueError, TypeError):
        feats["repetition_time"] = 3000.0

    try:
        feats["echo_time"] = float(getattr(dcm, "EchoTime", 40.0))
    except (ValueError, TypeError):
        feats["echo_time"] = 40.0

    # 2. Image intensity statistics
    try:
        img = dcm.pixel_array.astype(np.float32)
        slope = float(getattr(dcm, "RescaleSlope", 1.0))
        intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
        img = img * slope + intercept

        # Robust normalization
        p1, p99 = np.percentile(img, [1, 99])
        if p99 > p1:
            norm_img = np.clip((img - p1) / (p99 - p1), 0.0, 1.0)
        else:
            norm_img = np.zeros_like(img)

        feats["img_mean"] = float(np.mean(norm_img))
        feats["img_std"] = float(np.std(norm_img))
        feats["img_p90"] = float(np.percentile(norm_img, 90))
        feats["img_p95"] = float(np.percentile(norm_img, 95))
        feats["fluid_fraction"] = float(np.mean(norm_img > 0.82))
    except Exception:
        feats["img_mean"] = 0.30
        feats["img_std"] = 0.25
        feats["img_p90"] = 0.70
        feats["img_p95"] = 0.80
        feats["fluid_fraction"] = 0.04

    return feats


def extract_study_dicom_features(study_uid: str, study_series_df: pd.DataFrame, series_base_dir: Path) -> dict:
    """
    Extracts aggregated image and protocol features for a given study across all its series.
    """
    study_dir = series_base_dir / study_uid
    if not study_dir.exists():
        # Maybe flat directory or search under series_base_dir
        matches = list(series_base_dir.glob(f"*{study_uid}*"))
        if matches:
            study_dir = matches[0]
        else:
            study_dir = None

    study_feats = {
        "StudyInstanceUID": study_uid,
        "is_female": 0.5,
        "mean_thickness": 3.0,
        "total_slices": 0,
        "sagittal_fluid_fraction": 0.04,
        "coronal_fluid_fraction": 0.04,
        "axial_fluid_fraction": 0.04,
        "sagittal_p95": 0.80,
        "coronal_p95": 0.80,
        "axial_p95": 0.80,
    }

    if study_dir is None or not study_dir.exists():
        return study_feats

    plane_fluid_fractions = {"Sagittal": [], "Coronal": [], "Axial": []}
    plane_p95s = {"Sagittal": [], "Coronal": [], "Axial": []}
    sex_list = []
    thickness_list = []
    total_slices = 0

    for _, row in study_series_df.iterrows():
        series_uid = row["SeriesInstanceUID"]
        plane = str(row.get("Anatomical_Plane", "Sagittal"))
        is_fluid_fs = (row.get("Fluid_Sensitive", 0) == 1) and (row.get("Fat_Suppression", 0) == 1)

        series_dir = study_dir / series_uid
        if not series_dir.exists():
            matches = list(study_dir.glob(f"*{series_uid}*"))
            if matches:
                series_dir = matches[0]
            else:
                continue

        dcm_files = sorted(list(series_dir.glob("*.dcm")))
        if not dcm_files:
            continue

        n_slices = len(dcm_files)
        total_slices += n_slices

        # Read middle slice
        mid_idx = n_slices // 2
        mid_dcm_path = dcm_files[mid_idx]

        slice_feats = extract_slice_features(str(mid_dcm_path))
        if not slice_feats:
            continue

        sex_list.append(slice_feats.get("is_female", 0.5))
        thickness_list.append(slice_feats.get("slice_thickness", 3.0))

        if plane in plane_fluid_fractions and is_fluid_fs:
            plane_fluid_fractions[plane].append(slice_feats.get("fluid_fraction", 0.04))
            plane_p95s[plane].append(slice_feats.get("img_p95", 0.80))

    study_feats["total_slices"] = total_slices
    if sex_list:
        study_feats["is_female"] = float(np.mean(sex_list))
    if thickness_list:
        study_feats["mean_thickness"] = float(np.mean(thickness_list))

    for plane in ["Sagittal", "Coronal", "Axial"]:
        key_frac = f"{plane.lower()}_fluid_fraction"
        key_p95 = f"{plane.lower()}_p95"
        if plane_fluid_fractions[plane]:
            study_feats[key_frac] = float(np.mean(plane_fluid_fractions[plane]))
        if plane_p95s[plane]:
            study_feats[key_p95] = float(np.mean(plane_p95s[plane]))

    return study_feats
