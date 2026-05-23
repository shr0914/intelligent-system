from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


ENHANCEMENT_MODES = ("none", "gamma", "clahe", "gamma-clahe", "auto")


def mean_luminance(image_bgr: np.ndarray) -> float:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    return float(lab[:, :, 0].mean())


def is_low_light(image_bgr: np.ndarray, threshold: float = 90.0) -> bool:
    return mean_luminance(image_bgr) < threshold


def apply_gamma(image_bgr: np.ndarray, gamma: float = 0.65) -> np.ndarray:
    gamma = max(gamma, 0.05)
    lookup = np.array(
        [((value / 255.0) ** gamma) * 255 for value in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image_bgr, lookup)


def apply_clahe(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=max(clip_limit, 0.1),
        tileGridSize=(max(tile_grid_size, 1), max(tile_grid_size, 1)),
    )
    enhanced_lightness = clahe.apply(lightness)
    enhanced_lab = cv2.merge((enhanced_lightness, channel_a, channel_b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def enhance_low_light(
    image_bgr: np.ndarray,
    mode: str = "none",
    gamma: float = 0.65,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: int = 8,
    low_light_threshold: float = 90.0,
) -> tuple[np.ndarray, bool]:
    if mode not in ENHANCEMENT_MODES:
        raise ValueError(f"Unknown enhancement mode: {mode}")

    should_enhance = mode != "none"
    actual_mode = mode
    if mode == "auto":
        should_enhance = is_low_light(image_bgr, low_light_threshold)
        actual_mode = "gamma-clahe" if should_enhance else "none"

    if not should_enhance or actual_mode == "none":
        return image_bgr.copy(), False
    if actual_mode == "gamma":
        return apply_gamma(image_bgr, gamma), True
    if actual_mode == "clahe":
        return apply_clahe(image_bgr, clahe_clip_limit, clahe_tile_grid_size), True
    if actual_mode == "gamma-clahe":
        gamma_image = apply_gamma(image_bgr, gamma)
        return apply_clahe(gamma_image, clahe_clip_limit, clahe_tile_grid_size), True

    return image_bgr.copy(), False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview low-light enhancement on an image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--mode", choices=ENHANCEMENT_MODES, default="gamma-clahe")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--gamma", type=float, default=0.65)
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--clahe-tile-grid-size", type=int, default=8)
    parser.add_argument("--low-light-threshold", type=float, default=90.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")
    enhanced, applied = enhance_low_light(
        image,
        mode=args.mode,
        gamma=args.gamma,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid_size=args.clahe_tile_grid_size,
        low_light_threshold=args.low_light_threshold,
    )
    output = args.output or args.image.with_name(f"{args.image.stem}_{args.mode}{args.image.suffix}")
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), enhanced)
    print(f"Saved {output} (applied={applied}, luminance={mean_luminance(image):.1f})")


if __name__ == "__main__":
    main()
