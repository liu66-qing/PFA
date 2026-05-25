"""Download and preprocess BTCV dataset for PFA experiments.

BTCV (Beyond The Cranial Vault) / Synapse Multi-Organ:
- 30 abdominal CT scans with 13 organ annotations
- Organs: spleen, right kidney, left kidney, gallbladder, esophagus,
  liver, stomach, aorta, IVC, portal vein, pancreas, right adrenal, left adrenal
- Label mapping matches AMOS subset

Source: Synapse platform (requires manual download) or cached copies.
We use the preprocessed version from TransUNet/nnUNet community.
"""
import os
import sys
import json
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from scipy.ndimage import zoom

TARGET_SPACING = (1.5, 1.5, 1.5)
OUTPUT_DIR = Path("/root/autodl-tmp/data/btcv_cached")

# BTCV label mapping (1-indexed):
# 1: spleen, 2: right kidney, 3: left kidney, 4: gallbladder,
# 5: esophagus, 6: liver, 7: stomach, 8: aorta,
# 9: IVC, 10: portal/splenic vein, 11: pancreas,
# 12: right adrenal, 13: left adrenal
BTCV_ORGANS = {
    1: "spleen", 2: "right_kidney", 3: "left_kidney", 4: "gallbladder",
    5: "esophagus", 6: "liver", 7: "stomach", 8: "aorta",
    9: "IVC", 10: "portal_vein", 11: "pancreas",
    12: "right_adrenal", 13: "left_adrenal"
}


def resample_volume(image_sitk, target_spacing, is_label=False):
    """Resample to target spacing."""
    orig_spacing = image_sitk.GetSpacing()
    orig_size = image_sitk.GetSize()
    target_size = [
        int(round(orig_size[i] * (orig_spacing[i] / target_spacing[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(target_size)
    resampler.SetOutputDirection(image_sitk.GetDirection())
    resampler.SetOutputOrigin(image_sitk.GetOrigin())
    resampler.SetTransform(sitk.Transform())

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(image_sitk)


def normalize_ct(arr):
    """CT windowing and normalization."""
    arr = np.clip(arr, -1024, 1024)
    arr = (arr - arr.mean()) / (arr.std() + 1e-8)
    return arr.astype(np.float32)


def process_btcv(raw_dir, train_ratio=0.7):
    """Process raw BTCV NIfTI files into cached .npy format."""
    raw_dir = Path(raw_dir)

    # Find image and label files
    img_dir = None
    lbl_dir = None

    # Common BTCV directory structures
    for candidate in ["img", "imagesTr", "images", "RawData/Training/img"]:
        if (raw_dir / candidate).exists():
            img_dir = raw_dir / candidate
            break

    for candidate in ["label", "labelsTr", "labels", "RawData/Training/label"]:
        if (raw_dir / candidate).exists():
            lbl_dir = raw_dir / candidate
            break

    if img_dir is None or lbl_dir is None:
        print(f"ERROR: Cannot find image/label dirs in {raw_dir}")
        print(f"Contents: {list(raw_dir.rglob('*'))[:20]}")
        sys.exit(1)

    img_files = sorted(img_dir.glob("*.nii.gz"))
    if not img_files:
        img_files = sorted(img_dir.glob("*.nii"))
    print(f"Found {len(img_files)} images in {img_dir}")
    print(f"Labels in {lbl_dir}")

    # Split into train/val
    n_train = int(len(img_files) * train_ratio)
    train_files = img_files[:n_train]
    val_files = img_files[n_train:]
    print(f"Split: {len(train_files)} train, {len(val_files)} val")

    # Create output dirs
    for subdir in ["imagesTr", "labelsTr", "imagesVa", "labelsVa"]:
        (OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

    # Process
    for split_name, files, img_out, lbl_out in [
        ("Train", train_files, "imagesTr", "labelsTr"),
        ("Val", val_files, "imagesVa", "labelsVa"),
    ]:
        print(f"\nProcessing {split_name} ({len(files)} cases)...")
        for i, img_path in enumerate(files):
            case_name = img_path.stem.replace(".nii", "")

            # Find matching label
            lbl_path = None
            for ext in [".nii.gz", ".nii"]:
                candidate = lbl_dir / f"{case_name}{ext}"
                if candidate.exists():
                    lbl_path = candidate
                    break
                # Try label prefix variations
                for prefix in ["label", "seg"]:
                    candidate = lbl_dir / f"{prefix}{case_name.replace(img, )}{ext}"
                    if candidate.exists():
                        lbl_path = candidate
                        break

            if lbl_path is None:
                # Try matching by index
                lbl_files = sorted(lbl_dir.glob("*.nii*"))
                if i < len(lbl_files):
                    lbl_path = lbl_files[i]

            if lbl_path is None:
                print(f"  WARNING: No label for {case_name}, skipping")
                continue

            # Load and resample
            img_sitk = sitk.ReadImage(str(img_path))
            lbl_sitk = sitk.ReadImage(str(lbl_path))

            img_resampled = resample_volume(img_sitk, TARGET_SPACING, is_label=False)
            lbl_resampled = resample_volume(lbl_sitk, TARGET_SPACING, is_label=True)

            img_arr = sitk.GetArrayFromImage(img_resampled)
            lbl_arr = sitk.GetArrayFromImage(lbl_resampled).astype(np.int16)

            # Normalize
            img_arr = normalize_ct(img_arr)

            # Save
            out_name = f"{case_name}.npy"
            np.save(str(OUTPUT_DIR / img_out / out_name), img_arr)
            np.save(str(OUTPUT_DIR / lbl_out / out_name), lbl_arr)

            n_organs = len(np.unique(lbl_arr)) - 1
            print(f"  {case_name}: shape={img_arr.shape}, organs={n_organs}")

    # Save metadata
    meta = {
        "dataset": "BTCV",
        "n_train": len(train_files),
        "n_val": len(val_files),
        "spacing": list(TARGET_SPACING),
        "organs": BTCV_ORGANS,
    }
    with (OUTPUT_DIR / "metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! Cached data saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, help="Path to raw BTCV data")
    args = parser.parse_args()
    process_btcv(args.raw_dir)
