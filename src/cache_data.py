"""Pre-cache AMOS volumes as .npy for fast training IO."""

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def resample(image_sitk, target_spacing, interpolator, default_value=0.0):
    orig_spacing = image_sitk.GetSpacing()
    orig_size = image_sitk.GetSize()
    target_size = [
        int(round(orig_size[i] * (orig_spacing[i] / target_spacing[i])))
        for i in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(target_size)
    resampler.SetOutputDirection(image_sitk.GetDirection())
    resampler.SetOutputOrigin(image_sitk.GetOrigin())
    resampler.SetDefaultPixelValue(default_value)
    return resampler.Execute(image_sitk)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/root/autodl-tmp/data/amos22_raw/amos22")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/data/amos22_cached")
    parser.add_argument("--spacing", type=float, nargs=3, default=[1.5, 1.5, 1.5])
    parser.add_argument("--splits", nargs="+", default=["Tr", "Va"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    spacing = tuple(args.spacing)

    for split in args.splits:
        img_dir = data_dir / f"images{split}"
        lbl_dir = data_dir / f"labels{split}"
        out_img = out_dir / f"images{split}"
        out_lbl = out_dir / f"labels{split}"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        cases = sorted(img_dir.glob("*.nii.gz"))
        print(f"Caching {split}: {len(cases)} cases at spacing {spacing}")

        for i, img_path in enumerate(cases):
            name = img_path.stem.replace(".nii", "")
            lbl_path = lbl_dir / img_path.name

            if (out_img / f"{name}.npy").exists():
                continue

            img_sitk = sitk.ReadImage(str(img_path))
            lbl_sitk = sitk.ReadImage(str(lbl_path))

            img_res = resample(img_sitk, spacing, sitk.sitkLinear,
                               float(sitk.GetArrayFromImage(img_sitk).min()))
            lbl_res = resample(lbl_sitk, spacing, sitk.sitkNearestNeighbor)

            img_arr = sitk.GetArrayFromImage(img_res).astype(np.float32)
            lbl_arr = sitk.GetArrayFromImage(lbl_res).astype(np.int16)

            np.save(out_img / f"{name}.npy", img_arr)
            np.save(out_lbl / f"{name}.npy", lbl_arr)

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(cases)}] {name} shape={img_arr.shape}")

        print(f"  Done: {len(cases)} cached to {out_dir}")


if __name__ == "__main__":
    main()
