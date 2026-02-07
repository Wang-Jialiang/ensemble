import os
import sys

import torch
import torchvision

# Add parent directory to sys.path to allow importing as a package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ensemble.training.augmentation import (
    GriddedPerlinAugmentation,
    GridMaskAugmentation,
)


# Mock Config
class MockConfig:
    def __init__(self):
        self.image_size = 32
        self.mask_pool_size = 1
        self.augmentation_use_mean_fill = False
        self.dataset_mean = [0.485, 0.456, 0.406]
        self.dataset_std = [0.229, 0.224, 0.225]

        # Demo parameters
        self.target_ratio = 0.4
        self.augmentation_prob = 1.0

        # GridMask
        self.gridmask_d_ratio_min = 0.4
        self.gridmask_d_ratio_max = 0.6

        # Perlin
        self.perlin_persistence = 0.5
        self.perlin_octaves = 4
        self.perlin_scale_ratio_min = 0.08
        self.perlin_scale_ratio_max = 0.12

        # Gridded Perlin
        self.gridded_perlin_grid_size = 32  # 网格大小
        self.gridded_perlin_cloud_ratio = 0.6  # 每个网格内云块占比


def get_demo_image(cfg: MockConfig):
    # Create a simple pattern: gradients + circle
    H, W = cfg.image_size, cfg.image_size

    # Generate random noise matching ImageNet stats
    # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
    mean = torch.tensor(cfg.dataset_mean).view(1, 3, 1, 1)
    std = torch.tensor(cfg.dataset_std).view(1, 3, 1, 1)

    # Base noise
    img = torch.randn(1, 3, H, W) * std + mean

    # Boost blue channel to make it "variable blue"
    # We add a bit more intensity to blue while keeping some randomness
    img[:, 2, :, :] += 0.2

    # Clip to valid range [0, 1]
    img = torch.clamp(img, 0.0, 1.0)
    return img


def run_demo():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg = MockConfig()

    # 1. Prepare Base Image
    original_img = get_demo_image(cfg).to(device)

    # 2. Setup Methods
    methods = {
        # "Cutout": CutoutAugmentation.from_config(device, cfg),
        # "PixelHaS": PixelHaSAugmentation.from_config(device, cfg),
        "GridMask": GridMaskAugmentation.from_config(device, cfg),
        "gridded_perlin": GriddedPerlinAugmentation.from_config(device, cfg),
    }

    output_dir = os.path.dirname(os.path.abspath(__file__))

    # Save original
    torchvision.utils.save_image(
        original_img, os.path.join(output_dir, "demo_original.png")
    )
    print("Saved original image")

    for name, method in methods.items():
        print(f"Processing {name}...")
        # Precompute masks
        method.precompute_masks(cfg.target_ratio)

        # Apply
        aug_img, _ = method.apply(
            original_img.clone(), None, cfg.target_ratio, cfg.augmentation_prob
        )

        filename = f"demo_{name.lower()}.png"
        filepath = os.path.join(output_dir, filename)
        torchvision.utils.save_image(aug_img, filepath)
        print(f"Saved {filepath}")


if __name__ == "__main__":
    run_demo()
