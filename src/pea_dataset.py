"""PEA Dataset: Multi-organ prompt sampling for equivariant training.

Samples prompt pairs per volume:
- Same-target pairs: clean point + noisy point within target basin
- Cross-target pairs: prompt for organ A + prompt for organ B (adjacent preferred)
- Multi-prompt pairs: point + bounding box for same organ
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import distance_transform_edt


class PEADataset(Dataset):
    """Dataset that yields ROI crops with structured prompt pairs."""

    ORGAN_ADJACENCY = {
        # AMOS organ adjacency (1-indexed): organ_id -> list of adjacent organ_ids
        1: [2, 3, 8, 9],      # spleen -> kidney_L, kidney_R, aorta, IVC
        2: [1, 3, 9, 10],     # kidney_L -> spleen, kidney_R, IVC, portal_vein
        3: [1, 2, 8],         # kidney_R -> spleen, kidney_L, aorta
        6: [7, 8, 9, 10],     # liver -> stomach, aorta, IVC, portal_vein
        7: [6, 1, 4],         # stomach -> liver, spleen, pancreas
        4: [7, 8, 9],         # pancreas -> stomach, aorta, IVC
        8: [6, 4, 9],         # aorta -> liver, pancreas, IVC
        9: [6, 8, 10],        # IVC -> liver, aorta, portal_vein
        10: [6, 9],           # portal_vein -> liver, IVC
        14: [15],             # bladder -> prostate/uterus
        15: [14],             # prostate/uterus -> bladder
    }

    def __init__(
        self,
        data_dir: str,
        split: str = "Tr",
        crop_size: int = 128,
        target_spacing: tuple = (1.5, 1.5, 1.5),
        noise_range: tuple = (3, 10),
        min_organ_voxels: int = 200,
        samples_per_volume: int = 4,
        cross_target_ratio: float = 0.4,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.crop_size = crop_size
        self.target_spacing = target_spacing
        self.noise_range = noise_range
        self.min_organ_voxels = min_organ_voxels
        self.samples_per_volume = samples_per_volume
        self.cross_target_ratio = cross_target_ratio
        self.rng = np.random.default_rng(seed)

        img_dir = self.data_dir / f"images{split}"
        # Support both .nii.gz (raw) and .npy (cached) formats
        self.cases = sorted(img_dir.glob("*.nii.gz"))
        if not self.cases:
            self.cases = sorted(img_dir.glob("*.npy"))
            self.cached = True
        else:
            self.cached = False
        self.lbl_dir = self.data_dir / f"labels{split}"

    def __len__(self):
        return len(self.cases) * self.samples_per_volume

    def _load_and_resample(self, img_path, lbl_path):
        """Load volume. Uses cached .npy if available, else NIfTI + resample."""
        if self.cached:
            image_arr = np.load(str(img_path)).astype(np.float32)
            label_arr = np.load(str(lbl_path)).astype(np.int16)
            return image_arr, label_arr

        import SimpleITK as sitk

        image_sitk = sitk.ReadImage(str(img_path))
        label_sitk = sitk.ReadImage(str(lbl_path))

        orig_spacing = image_sitk.GetSpacing()
        orig_size = image_sitk.GetSize()
        target_size = [
            int(round(orig_size[i] * (orig_spacing[i] / self.target_spacing[i])))
            for i in range(3)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(self.target_spacing)
        resampler.SetSize(target_size)
        resampler.SetOutputDirection(image_sitk.GetDirection())
        resampler.SetOutputOrigin(image_sitk.GetOrigin())

        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(float(sitk.GetArrayFromImage(image_sitk).min()))
        image_resampled = resampler.Execute(image_sitk)

        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        label_resampled = resampler.Execute(label_sitk)

        image_arr = sitk.GetArrayFromImage(image_resampled).astype(np.float32)
        label_arr = sitk.GetArrayFromImage(label_resampled).astype(np.int16)
        return image_arr, label_arr

    def _z_normalize(self, image: np.ndarray) -> np.ndarray:
        image = image.copy()
        mask = image > 0
        if mask.any():
            image[mask] = (image[mask] - image[mask].mean()) / (image[mask].std() + 1e-8)
        return image

    def _crop_around_organ(self, image, label, organ_mask):
        """Crop 128^3 ROI centered on organ."""
        coords = np.argwhere(organ_mask > 0)
        center = np.rint(coords.mean(axis=0)).astype(int)
        half = self.crop_size // 2

        start = center - half
        end = start + self.crop_size

        src_start = np.maximum(start, 0)
        src_end = np.minimum(end, np.array(image.shape))
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)

        roi_image = np.zeros((self.crop_size,) * 3, dtype=np.float32)
        roi_label = np.zeros((self.crop_size,) * 3, dtype=np.int16)
        src_sl = tuple(slice(int(s), int(e)) for s, e in zip(src_start, src_end))
        dst_sl = tuple(slice(int(s), int(e)) for s, e in zip(dst_start, dst_end))
        roi_image[dst_sl] = image[src_sl]
        roi_label[dst_sl] = label[src_sl]

        return roi_image, roi_label, center

    def _sample_point_in_organ(self, mask: np.ndarray, mode: str = "center") -> np.ndarray:
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return None
        if mode == "center":
            center = coords.mean(axis=0)
            idx = np.argmin(np.sum((coords - center) ** 2, axis=1))
            return coords[idx].astype(int)
        else:
            return coords[self.rng.integers(0, len(coords))].astype(int)

    def _perturb_within_basin(self, point, organ_mask, organ_radius):
        """Perturb point within same-target basin (SDF <= delta)."""
        delta = min(3, int(0.1 * organ_radius))
        noise = self.rng.integers(self.noise_range[0], self.noise_range[1] + 1)
        shift = self.rng.integers(-noise, noise + 1, size=3)
        new_point = np.clip(point + shift, 0, np.array(organ_mask.shape) - 1)
        return new_point.astype(int)

    def _get_box_prompt(self, mask: np.ndarray, jitter: int = 5) -> np.ndarray:
        """Get bounding box [z1,y1,x1,z2,y2,x2] with optional jitter."""
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return None
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        # Add jitter
        mins = np.maximum(mins - self.rng.integers(0, jitter + 1, size=3), 0)
        maxs = np.minimum(maxs + self.rng.integers(0, jitter + 1, size=3),
                          np.array(mask.shape) - 1)
        return np.concatenate([mins, maxs]).astype(int)

    def _select_cross_target(self, primary_organ_id, available_organs):
        """Select cross-target organ, preferring adjacent ones."""
        adjacent = self.ORGAN_ADJACENCY.get(primary_organ_id, [])
        adjacent_available = [o for o in adjacent if o in available_organs]

        if adjacent_available and self.rng.random() < 0.7:
            return self.rng.choice(adjacent_available)
        others = [o for o in available_organs if o != primary_organ_id]
        if others:
            return self.rng.choice(others)
        return None

    def __getitem__(self, idx):
        case_idx = idx // self.samples_per_volume
        img_path = self.cases[case_idx]
        lbl_path = self.lbl_dir / img_path.name

        image_arr, label_arr = self._load_and_resample(img_path, lbl_path)
        image_arr = self._z_normalize(image_arr)

        # Find valid organs
        organs = [int(v) for v in np.unique(label_arr) if int(v) > 0]
        valid_organs = []
        for oid in organs:
            mask = (label_arr == oid)
            if mask.sum() >= self.min_organ_voxels:
                valid_organs.append(oid)

        if not valid_organs:
            return self._empty_sample()

        # Select primary organ
        primary_id = self.rng.choice(valid_organs)
        primary_mask = (label_arr == primary_id).astype(np.uint8)

        # Crop ROI around primary organ
        roi_image, roi_label, _ = self._crop_around_organ(image_arr, label_arr, primary_mask)
        roi_primary_mask = (roi_label == primary_id).astype(np.uint8)

        if roi_primary_mask.sum() < 50:
            return self._empty_sample()

        # Compute organ radius for basin definition
        organ_radius = (roi_primary_mask.sum() * 3 / (4 * np.pi)) ** (1/3)

        # Clean point prompt
        clean_point = self._sample_point_in_organ(roi_primary_mask, "center")
        if clean_point is None:
            return self._empty_sample()

        # Noisy point (same-target perturbation)
        noisy_point = self._perturb_within_basin(clean_point, roi_primary_mask, organ_radius)

        # Box prompt for multi-prompt consistency
        box_prompt = self._get_box_prompt(roi_primary_mask)

        # Cross-target pair
        cross_organ_id = self._select_cross_target(primary_id, valid_organs)
        cross_point = None
        cross_mask = None
        if cross_organ_id is not None:
            roi_cross_mask = (roi_label == cross_organ_id).astype(np.uint8)
            if roi_cross_mask.sum() >= 50:
                cross_point = self._sample_point_in_organ(roi_cross_mask, "random")
                cross_mask = roi_cross_mask

        sample = {
            "image": torch.from_numpy(roi_image).unsqueeze(0),  # 1×D×H×W
            "gt_primary": torch.from_numpy(roi_primary_mask).unsqueeze(0).float(),
            "clean_point": torch.from_numpy(clean_point).float(),
            "noisy_point": torch.from_numpy(noisy_point).float(),
            "box_prompt": torch.from_numpy(box_prompt).float() if box_prompt is not None
                          else torch.zeros(6),
            "has_box": box_prompt is not None,
            "primary_organ_id": primary_id,
        }

        if cross_point is not None and cross_mask is not None:
            sample["cross_point"] = torch.from_numpy(cross_point).float()
            sample["gt_cross"] = torch.from_numpy(cross_mask).unsqueeze(0).float()
            sample["has_cross"] = True
            sample["cross_organ_id"] = cross_organ_id
        else:
            sample["cross_point"] = torch.zeros(3)
            sample["gt_cross"] = torch.zeros(1, self.crop_size, self.crop_size, self.crop_size)
            sample["has_cross"] = False
            sample["cross_organ_id"] = -1

        return sample

    def _empty_sample(self):
        """Return a dummy sample when no valid organ found."""
        cs = self.crop_size
        return {
            "image": torch.zeros(1, cs, cs, cs),
            "gt_primary": torch.zeros(1, cs, cs, cs),
            "clean_point": torch.zeros(3),
            "noisy_point": torch.zeros(3),
            "box_prompt": torch.zeros(6),
            "has_box": False,
            "primary_organ_id": -1,
            "cross_point": torch.zeros(3),
            "gt_cross": torch.zeros(1, cs, cs, cs),
            "has_cross": False,
            "cross_organ_id": -1,
        }
