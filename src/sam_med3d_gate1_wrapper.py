"""TorchIO-free SAM-Med3D Gate 1 inference wrapper.

This wrapper mirrors the practical parts of the official SAM-Med3D validation
pipeline while avoiding TorchIO version-sensitive transforms:

1. read NIfTI as SimpleITK array in Z, Y, X order;
2. resample image and label to target spacing;
3. crop/pad a class-specific 128^3 ROI around the target label;
4. z-normalize nonzero image voxels;
5. run SAM-Med3D with point prompts in ROI tensor coordinates;
6. paste the ROI prediction back onto the resampled grid and compute Dice.

The important protocol detail is that SAM-Med3D's 3D prompt encoder is used in
the same coordinate order as the ROI tensor. Do not reverse ZYX to XYZ here.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


sitk = None


@dataclass
class CropPadInfo:
    src_start: list[int]
    src_end: list[int]
    dst_start: list[int]
    dst_end: list[int]
    center_zyx: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM-Med3D Gate 1 point-noise vulnerability evaluation."
    )
    parser.add_argument("--sam-root", default="/root/autodl-tmp/SAM-Med3D")
    parser.add_argument("--checkpoint", default="/root/autodl-tmp/SAM-Med3D/ckpt/sam_med3d_turbo.pth")
    parser.add_argument(
        "--image",
        default="/root/autodl-tmp/SAM-Med3D/test_data/amos_val_toy_data/imagesVa/amos_0013.nii.gz",
    )
    parser.add_argument(
        "--label",
        default="/root/autodl-tmp/SAM-Med3D/test_data/amos_val_toy_data/labelsVa/amos_0013.nii.gz",
    )
    parser.add_argument("--output-dir", default="results/vulnerability")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-type", default="vit_b_ori")
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--target-spacing", type=float, nargs=3, default=(1.5, 1.5, 1.5))
    parser.add_argument("--noise-levels", type=int, nargs="+", default=[0, 2, 5, 8, 10, 15, 20])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-roi-voxels", type=int, default=50)
    parser.add_argument(
        "--prompt",
        choices=["center", "random"],
        default="center",
        help="Clean positive point source inside each organ ROI.",
    )
    parser.add_argument(
        "--save-preds",
        action="store_true",
        help="Save per-organ clean/noisy ROI predictions as compressed npz files.",
    )
    return parser.parse_args()


def require_simpleitk():
    global sitk
    if sitk is not None:
        return sitk
    try:
        import SimpleITK as sitk_module
    except ImportError as exc:  # pragma: no cover - dependency check for server runs
        raise SystemExit("SimpleITK is required: pip install SimpleITK") from exc
    sitk = sitk_module
    return sitk


def add_sam_root(sam_root: str) -> None:
    root = Path(sam_root)
    if not root.exists():
        raise FileNotFoundError(f"SAM-Med3D root not found: {root}")
    sys.path.insert(0, str(root))


def load_model(sam_root: str, checkpoint: str, model_type: str, device: torch.device) -> torch.nn.Module:
    add_sam_root(sam_root)
    from segment_anything.build_sam3D import sam_model_registry3D

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = sam_model_registry3D[model_type](checkpoint=None)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def read_image(path: str) -> tuple[sitk.Image, np.ndarray]:
    sitk = require_simpleitk()
    image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(image)
    return image, array


def resample_sitk(
    image: sitk.Image,
    target_spacing_xyz: Iterable[float],
    interpolator: int,
    default_value: float = 0.0,
) -> sitk.Image:
    sitk = require_simpleitk()
    target_spacing_xyz = tuple(float(v) for v in target_spacing_xyz)
    orig_spacing = image.GetSpacing()
    orig_size = image.GetSize()
    target_size = [
        int(round(orig_size[i] * (orig_spacing[i] / target_spacing_xyz[i])))
        for i in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputSpacing(target_spacing_xyz)
    resampler.SetSize(target_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetDefaultPixelValue(default_value)
    return resampler.Execute(image)


def crop_or_pad_around_mask(
    image_zyx: np.ndarray,
    label_zyx: np.ndarray,
    crop_size: int,
) -> tuple[np.ndarray, np.ndarray, CropPadInfo]:
    coords = np.argwhere(label_zyx > 0)
    if len(coords) == 0:
        raise ValueError("Cannot crop ROI because label mask is empty.")

    center = np.rint(coords.mean(axis=0)).astype(int)
    half = crop_size // 2
    desired_start = center - half
    desired_end = desired_start + crop_size

    src_start = np.maximum(desired_start, 0)
    src_end = np.minimum(desired_end, np.array(image_zyx.shape))
    dst_start = src_start - desired_start
    dst_end = dst_start + (src_end - src_start)

    roi_image = np.zeros((crop_size, crop_size, crop_size), dtype=np.float32)
    roi_label = np.zeros((crop_size, crop_size, crop_size), dtype=np.uint8)
    src_slices = tuple(slice(int(s), int(e)) for s, e in zip(src_start, src_end))
    dst_slices = tuple(slice(int(s), int(e)) for s, e in zip(dst_start, dst_end))
    roi_image[dst_slices] = image_zyx[src_slices].astype(np.float32)
    roi_label[dst_slices] = label_zyx[src_slices].astype(np.uint8)

    info = CropPadInfo(
        src_start=src_start.astype(int).tolist(),
        src_end=src_end.astype(int).tolist(),
        dst_start=dst_start.astype(int).tolist(),
        dst_end=dst_end.astype(int).tolist(),
        center_zyx=center.astype(int).tolist(),
    )
    return roi_image, roi_label, info


def paste_roi(roi_pred: np.ndarray, output_shape: tuple[int, int, int], info: CropPadInfo) -> np.ndarray:
    full = np.zeros(output_shape, dtype=np.uint8)
    src_slices = tuple(slice(s, e) for s, e in zip(info.dst_start, info.dst_end))
    dst_slices = tuple(slice(s, e) for s, e in zip(info.src_start, info.src_end))
    full[dst_slices] = roi_pred[src_slices].astype(np.uint8)
    return full


def z_normalize_like_official(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=True)
    mask = image > 0
    if not np.any(mask):
        mask = image != 0
    if not np.any(mask):
        return image
    mean = float(image[mask].mean())
    std = float(image[mask].std())
    image[mask] = (image[mask] - mean) / (std + 1e-8)
    return image


def choose_clean_point(mask: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        raise ValueError("Cannot sample a point from an empty mask.")
    if mode == "random":
        return coords[int(rng.integers(0, len(coords)))].astype(int)
    center = coords.mean(axis=0)
    nearest = np.argmin(np.sum((coords - center) ** 2, axis=1))
    return coords[int(nearest)].astype(int)


def perturb_point(
    point: np.ndarray,
    noise: int,
    shape: tuple[int, int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    if noise <= 0:
        return point.copy()
    shift = rng.integers(-noise, noise + 1, size=3)
    return np.clip(point + shift, 0, np.array(shape) - 1).astype(int)


def infer_point(
    model: torch.nn.Module,
    roi_image: np.ndarray,
    point_zyx: np.ndarray,
    device: torch.device,
    low_res_logits: torch.Tensor | None = None,
) -> tuple[np.ndarray, torch.Tensor]:
    crop_size = roi_image.shape[0]
    roi_tensor = torch.from_numpy(roi_image).float().unsqueeze(0).unsqueeze(0).to(device)
    point_tensor = torch.as_tensor(point_zyx, dtype=torch.float32, device=device).view(1, 1, 3)
    label_tensor = torch.ones((1, 1), dtype=torch.long, device=device)
    if low_res_logits is None:
        low_res_logits = torch.zeros(
            (1, 1, crop_size // 4, crop_size // 4, crop_size // 4),
            dtype=torch.float32,
            device=device,
        )

    with torch.no_grad():
        image_embedding = model.image_encoder(roi_tensor)
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=[point_tensor, label_tensor],
            boxes=None,
            masks=low_res_logits,
        )
        low_res_masks, _ = model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        masks_hr = F.interpolate(
            low_res_masks,
            size=roi_image.shape,
            mode="trilinear",
            align_corners=False,
        )
    pred = (torch.sigmoid(masks_hr).cpu().numpy().squeeze() > 0.5).astype(np.uint8)
    return pred, low_res_masks.detach()


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    denom = int(pred_b.sum() + gt_b.sum())
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred_b, gt_b).sum() / denom)


def evaluate(args: argparse.Namespace) -> dict:
    sitk = require_simpleitk()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.sam_root, args.checkpoint, args.model_type, device)

    image_sitk, image_orig = read_image(args.image)
    label_sitk, label_orig = read_image(args.label)
    if image_orig.shape != label_orig.shape:
        raise ValueError(f"Image/label shape mismatch: {image_orig.shape} vs {label_orig.shape}")

    image_resampled = resample_sitk(
        image_sitk,
        args.target_spacing,
        sitk.sitkLinear,
        default_value=float(np.min(image_orig)),
    )
    label_resampled = resample_sitk(label_sitk, args.target_spacing, sitk.sitkNearestNeighbor)
    image_arr = sitk.GetArrayFromImage(image_resampled).astype(np.float32)
    label_arr = sitk.GetArrayFromImage(label_resampled).astype(np.int16)

    organs = [int(v) for v in np.unique(label_arr) if int(v) > 0]
    rows: list[dict] = []
    roi_meta: list[dict] = []
    print(f"Device: {device}")
    print(f"Original shape ZYX: {image_orig.shape}; resampled shape ZYX: {image_arr.shape}")
    print(f"Original spacing XYZ: {image_sitk.GetSpacing()}; target spacing XYZ: {tuple(args.target_spacing)}")
    print(f"Testing labels: {organs}")

    for organ_id in organs:
        organ_label = (label_arr == organ_id).astype(np.uint8)
        if int(organ_label.sum()) < args.min_roi_voxels:
            print(f"  label {organ_id:2d}: skipped, too small after resample ({int(organ_label.sum())} voxels)")
            continue

        roi_image, roi_label, crop_info = crop_or_pad_around_mask(image_arr, organ_label, args.crop_size)
        roi_image = z_normalize_like_official(roi_image)
        clean_point = choose_clean_point(roi_label, args.prompt, rng)
        roi_voxels = int(roi_label.sum())
        roi_meta.append({"organ_id": organ_id, "roi_voxels": roi_voxels, **asdict(crop_info)})

        organ_results = {}
        for noise in args.noise_levels:
            for trial in range(args.trials):
                trial_rng = np.random.default_rng(args.seed + organ_id * 10000 + noise * 100 + trial)
                point = perturb_point(clean_point, noise, roi_label.shape, trial_rng)
                pred_roi, _ = infer_point(model, roi_image, point, device)
                pred_full = paste_roi(pred_roi, label_arr.shape, crop_info)
                dice = dice_score(pred_full, organ_label)
                point_inside = bool(roi_label[tuple(point)] > 0)
                row = {
                    "organ_id": organ_id,
                    "roi_voxels": roi_voxels,
                    "noise": noise,
                    "trial": trial,
                    "dice": dice,
                    "clean_point_zyx": clean_point.tolist(),
                    "point_zyx": point.tolist(),
                    "point_inside_gt": point_inside,
                }
                rows.append(row)
                organ_results.setdefault(noise, []).append(dice)

                if args.save_preds and noise in (0, 10) and trial == 0:
                    np.savez_compressed(
                        output_dir / f"organ{organ_id:02d}_noise{noise:02d}_trial{trial}.npz",
                        pred_roi=pred_roi,
                        roi_label=roi_label,
                        point_zyx=point,
                    )

        clean = float(np.mean(organ_results.get(0, [np.nan])))
        msg = [f"label {organ_id:2d}: vox={roi_voxels:6d}", f"clean={clean:.4f}"]
        for noise in (5, 10, 20):
            if noise in organ_results:
                msg.append(f"shift{noise}={float(np.mean(organ_results[noise])):.4f}")
        print("  " + "  ".join(msg))

    if not rows:
        raise RuntimeError("No organs were evaluated. Check label IDs and min-roi-voxels.")

    csv_path = output_dir / "sam_med3d_gate1_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    clean_mean = np.mean([r["dice"] for r in rows if r["noise"] == 0])
    summary = {
        "image": args.image,
        "label": args.label,
        "checkpoint": args.checkpoint,
        "crop_size": args.crop_size,
        "target_spacing_xyz": list(args.target_spacing),
        "prompt_order": "ROI tensor coordinate order, stored as ZYX; no ZYX->XYZ reversal",
        "noise_summary": {},
        "gate_1": {},
        "roi_meta": roi_meta,
    }
    print("\nVULNERABILITY SUMMARY")
    print("   Shift | Mean Dice | Drop vs Clean")
    print("-" * 40)
    for noise in args.noise_levels:
        vals = [r["dice"] for r in rows if r["noise"] == noise]
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        drop = float((clean_mean - mean) * 100.0)
        summary["noise_summary"][str(noise)] = {
            "mean_dice": mean,
            "std_dice": std,
            "drop_points_vs_clean": drop,
            "n": len(vals),
        }
        print(f"   {noise:5d} | {mean:9.4f} | {drop:+10.1f} pts")

    shift10 = summary["noise_summary"].get("10", {"mean_dice": clean_mean})
    drop10 = float((clean_mean - shift10["mean_dice"]) * 100.0)
    if drop10 >= 8:
        status = "GREEN"
    elif drop10 >= 4:
        status = "YELLOW"
    else:
        status = "RED"
    summary["gate_1"] = {
        "clean_mean_dice": float(clean_mean),
        "shift10_mean_dice": float(shift10["mean_dice"]),
        "drop10_points": drop10,
        "status": status,
    }

    json_path = output_dir / "sam_med3d_gate1_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nGATE 1")
    print(
        f"Clean={clean_mean:.4f}, Shift10={shift10['mean_dice']:.4f}, "
        f"Drop={drop10:.1f}pts => {status}"
    )
    print(f"Saved rows: {csv_path}")
    print(f"Saved summary: {json_path}")
    return summary


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
