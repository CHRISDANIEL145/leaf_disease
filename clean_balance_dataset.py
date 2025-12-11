"""
Dataset Cleaning and Balancing Script
======================================
Cleans the New Plant Diseases Dataset by removing:
- Duplicate images
- Corrupted images  
- Black/blank images
- Low-resolution images
- Blurry/noisy images

Then balances all classes to exact target counts using
downsampling and augmentation.

Author: Auto-generated
"""

import os
import shutil
import hashlib
from pathlib import Path
from collections import defaultdict
import random
from typing import Dict, List, Tuple, Set

import cv2
import numpy as np
from PIL import Image
import imagehash
from tqdm import tqdm
import albumentations as A

# ============================================================================
# CONFIGURATION
# ============================================================================

# Source dataset paths
SOURCE_BASE = Path(r"d:\leaf disease\dataset\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)")
TRAIN_DIR = SOURCE_BASE / "train"
VALID_DIR = SOURCE_BASE / "valid"

# Output paths
OUTPUT_BASE = Path(r"d:\leaf disease\clean_balanced_dataset")
OUTPUT_TRAIN = OUTPUT_BASE / "train"
OUTPUT_VALID = OUTPUT_BASE / "valid"
OUTPUT_TEST = OUTPUT_BASE / "test"

# Target counts per class
TARGET_TRAIN = 250
TARGET_VALID = 50
TARGET_TEST = 50

# Quality thresholds
MIN_RESOLUTION = 64          # Minimum width/height in pixels
MIN_VARIANCE = 100           # Minimum pixel variance (detect blank images)
MIN_MEAN_INTENSITY = 10      # Minimum mean intensity (detect black images)
MAX_MEAN_INTENSITY = 245     # Maximum mean intensity (detect white images)
BLUR_THRESHOLD = 50          # Laplacian variance threshold for blur detection
HASH_SIZE = 16               # Perceptual hash size for duplicate detection

# Augmentation pipeline (compatible with albumentations v2.x)
AUGMENTATION = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Transpose(p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.MotionBlur(blur_limit=5, p=0.2),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
])




# ============================================================================
# IMAGE VALIDATION FUNCTIONS
# ============================================================================

def is_corrupted(image_path: Path) -> bool:
    """Check if image is corrupted and cannot be opened."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        # Re-open because verify() can only be called once
        with Image.open(image_path) as img:
            img.load()
        return False
    except Exception:
        return True


def is_low_resolution(image_path: Path, min_size: int = MIN_RESOLUTION) -> bool:
    """Check if image is below minimum resolution."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return width < min_size or height < min_size
    except Exception:
        return True


def is_blank_or_black(image_path: Path) -> bool:
    """Check if image is visually empty (black, white, or uniform color)."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return True
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = np.var(gray)
        mean_intensity = np.mean(gray)
        
        # Check for low variance (uniform color)
        if variance < MIN_VARIANCE:
            return True
        
        # Check for nearly black
        if mean_intensity < MIN_MEAN_INTENSITY:
            return True
        
        # Check for nearly white
        if mean_intensity > MAX_MEAN_INTENSITY:
            return True
        
        return False
    except Exception:
        return True


def is_blurry(image_path: Path, threshold: float = BLUR_THRESHOLD) -> bool:
    """Check if image is too blurry using Laplacian variance."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return True
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return laplacian_var < threshold
    except Exception:
        return True


def compute_image_hash(image_path: Path) -> str:
    """Compute perceptual hash for duplicate detection."""
    try:
        with Image.open(image_path) as img:
            return str(imagehash.phash(img, hash_size=HASH_SIZE))
    except Exception:
        return ""


def validate_image(image_path: Path) -> Tuple[bool, str]:
    """
    Validate a single image and return (is_valid, reason).
    """
    # Check if corrupted
    if is_corrupted(image_path):
        return False, "corrupted"
    
    # Check resolution
    if is_low_resolution(image_path):
        return False, "low_resolution"
    
    # Check if blank/black
    if is_blank_or_black(image_path):
        return False, "blank_or_black"
    
    # Check if blurry
    if is_blurry(image_path):
        return False, "blurry"
    
    return True, "valid"


# ============================================================================
# DATASET PROCESSING FUNCTIONS
# ============================================================================

def get_class_folders(directory: Path) -> List[str]:
    """Get list of class folder names."""
    if not directory.exists():
        return []
    return [d.name for d in directory.iterdir() if d.is_dir()]


def scan_and_validate_images(source_dir: Path, class_name: str) -> Tuple[List[Path], Dict[str, int]]:
    """
    Scan a class folder and validate all images.
    Returns (valid_images, removal_stats).
    """
    class_path = source_dir / class_name
    if not class_path.exists():
        return [], {}
    
    valid_images = []
    removal_stats = defaultdict(int)
    hash_set: Set[str] = set()
    
    image_files = list(class_path.glob("*"))
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}
    image_files = [f for f in image_files if f.suffix.lower() in image_extensions]
    
    for img_path in image_files:
        # Validate image quality
        is_valid, reason = validate_image(img_path)
        
        if not is_valid:
            removal_stats[reason] += 1
            continue
        
        # Check for duplicates using perceptual hash
        img_hash = compute_image_hash(img_path)
        if img_hash in hash_set:
            removal_stats["duplicate"] += 1
            continue
        
        hash_set.add(img_hash)
        valid_images.append(img_path)
    
    return valid_images, dict(removal_stats)


def augment_image(image_path: Path, output_path: Path) -> bool:
    """Apply augmentation to an image and save it."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = AUGMENTATION(image=img_rgb)
        augmented_img = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(str(output_path), augmented_img)
        return True
    except Exception:
        return False


def balance_class(
    valid_images: List[Path],
    output_dir: Path,
    class_name: str,
    target_count: int
) -> Tuple[int, int]:
    """
    Balance a class to the target count.
    Returns (copied_count, augmented_count).
    """
    output_class_dir = output_dir / class_name
    output_class_dir.mkdir(parents=True, exist_ok=True)
    
    current_count = len(valid_images)
    copied = 0
    augmented = 0
    
    if current_count == 0:
        return 0, 0
    
    if current_count >= target_count:
        # Downsample: randomly select target_count images
        selected_images = random.sample(valid_images, target_count)
        for idx, img_path in enumerate(selected_images):
            output_path = output_class_dir / f"{class_name}_{idx:04d}{img_path.suffix}"
            shutil.copy2(img_path, output_path)
            copied += 1
    else:
        # Copy all valid images first
        for idx, img_path in enumerate(valid_images):
            output_path = output_class_dir / f"{class_name}_{idx:04d}{img_path.suffix}"
            shutil.copy2(img_path, output_path)
            copied += 1
        
        # Augment to fill the gap
        needed = target_count - current_count
        aug_idx = current_count
        
        while augmented < needed:
            # Randomly select an image to augment
            source_img = random.choice(valid_images)
            output_path = output_class_dir / f"{class_name}_{aug_idx:04d}_aug.jpg"
            
            if augment_image(source_img, output_path):
                augmented += 1
                aug_idx += 1
    
    return copied, augmented


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_dataset():
    """Main function to clean and balance the dataset."""
    print("=" * 70)
    print("DATASET CLEANING AND BALANCING SCRIPT")
    print("=" * 70)
    
    # Create output directories
    for output_dir in [OUTPUT_TRAIN, OUTPUT_VALID, OUTPUT_TEST]:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all class names from train directory (source of truth)
    class_names = sorted(get_class_folders(TRAIN_DIR))
    print(f"\nFound {len(class_names)} classes")
    
    # Storage for all valid images per class (combined from train + valid)
    all_valid_images: Dict[str, List[Path]] = {}
    all_removal_stats: Dict[str, Dict[str, int]] = {}
    
    # Phase 1: Scan and validate all images
    print("\n" + "-" * 70)
    print("PHASE 1: Scanning and Validating Images")
    print("-" * 70)
    
    for class_name in tqdm(class_names, desc="Validating classes"):
        class_valid_images = []
        class_removal_stats = defaultdict(int)
        
        # Scan train folder
        train_valid, train_stats = scan_and_validate_images(TRAIN_DIR, class_name)
        class_valid_images.extend(train_valid)
        for reason, count in train_stats.items():
            class_removal_stats[reason] += count
        
        # Scan valid folder
        valid_valid, valid_stats = scan_and_validate_images(VALID_DIR, class_name)
        class_valid_images.extend(valid_valid)
        for reason, count in valid_stats.items():
            class_removal_stats[reason] += count
        
        # Remove cross-folder duplicates
        unique_images = []
        seen_hashes: Set[str] = set()
        duplicates_removed = 0
        
        for img in class_valid_images:
            img_hash = compute_image_hash(img)
            if img_hash not in seen_hashes:
                seen_hashes.add(img_hash)
                unique_images.append(img)
            else:
                duplicates_removed += 1
        
        class_removal_stats["cross_duplicate"] += duplicates_removed
        
        all_valid_images[class_name] = unique_images
        all_removal_stats[class_name] = dict(class_removal_stats)
    
    # Phase 2: Split and balance dataset
    print("\n" + "-" * 70)
    print("PHASE 2: Balancing Dataset")
    print("-" * 70)
    
    total_target = TARGET_TRAIN + TARGET_VALID + TARGET_TEST
    balance_report: Dict[str, Dict] = {}
    
    for class_name in tqdm(class_names, desc="Balancing classes"):
        valid_images = all_valid_images[class_name].copy()
        random.shuffle(valid_images)
        
        original_count = len(valid_images)
        
        # Check if we have enough images
        if original_count == 0:
            print(f"\n  WARNING: {class_name} has no valid images!")
            balance_report[class_name] = {
                "original": 0,
                "removed": sum(all_removal_stats.get(class_name, {}).values()),
                "train_copied": 0, "train_augmented": 0,
                "valid_copied": 0, "valid_augmented": 0,
                "test_copied": 0, "test_augmented": 0,
            }
            continue
        
        # Split images for train/valid/test
        # Prioritize using unique images for each split
        train_images = []
        valid_images_split = []
        test_images = []
        
        if original_count >= total_target:
            # We have enough - just split evenly
            train_images = valid_images[:TARGET_TRAIN]
            valid_images_split = valid_images[TARGET_TRAIN:TARGET_TRAIN + TARGET_VALID]
            test_images = valid_images[TARGET_TRAIN + TARGET_VALID:TARGET_TRAIN + TARGET_VALID + TARGET_TEST]
        else:
            # Split proportionally: 71.4% train, 14.3% valid, 14.3% test
            train_ratio = TARGET_TRAIN / total_target
            valid_ratio = TARGET_VALID / total_target
            
            train_count = max(1, int(original_count * train_ratio))
            valid_count = max(1, int(original_count * valid_ratio))
            test_count = max(1, original_count - train_count - valid_count)
            
            train_images = valid_images[:train_count]
            valid_images_split = valid_images[train_count:train_count + valid_count]
            test_images = valid_images[train_count + valid_count:]
        
        # Balance each split
        train_c, train_a = balance_class(train_images, OUTPUT_TRAIN, class_name, TARGET_TRAIN)
        valid_c, valid_a = balance_class(valid_images_split, OUTPUT_VALID, class_name, TARGET_VALID)
        test_c, test_a = balance_class(test_images, OUTPUT_TEST, class_name, TARGET_TEST)
        
        balance_report[class_name] = {
            "original": original_count,
            "removed": sum(all_removal_stats.get(class_name, {}).values()),
            "train_copied": train_c, "train_augmented": train_a,
            "valid_copied": valid_c, "valid_augmented": valid_a,
            "test_copied": test_c, "test_augmented": test_a,
        }
    
    # Phase 3: Generate Report
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    
    print("\n📊 REMOVAL STATISTICS BY REASON:")
    total_removed = defaultdict(int)
    for class_name, stats in all_removal_stats.items():
        for reason, count in stats.items():
            total_removed[reason] += count
    
    for reason, count in sorted(total_removed.items()):
        print(f"   • {reason}: {count}")
    print(f"   TOTAL REMOVED: {sum(total_removed.values())}")
    
    print("\n📊 PER-CLASS SUMMARY:")
    print("-" * 100)
    print(f"{'Class Name':<50} {'Original':>10} {'Removed':>10} {'Train':>10} {'Valid':>10} {'Test':>10}")
    print("-" * 100)
    
    total_original = 0
    total_removed_count = 0
    
    for class_name in sorted(class_names):
        report = balance_report.get(class_name, {})
        orig = report.get("original", 0)
        removed = report.get("removed", 0)
        train_total = report.get("train_copied", 0) + report.get("train_augmented", 0)
        valid_total = report.get("valid_copied", 0) + report.get("valid_augmented", 0)
        test_total = report.get("test_copied", 0) + report.get("test_augmented", 0)
        
        total_original += orig
        total_removed_count += removed
        
        print(f"{class_name:<50} {orig:>10} {removed:>10} {train_total:>10} {valid_total:>10} {test_total:>10}")
    
    print("-" * 100)
    print(f"{'TOTAL':<50} {total_original:>10} {total_removed_count:>10} {TARGET_TRAIN * len(class_names):>10} {TARGET_VALID * len(class_names):>10} {TARGET_TEST * len(class_names):>10}")
    
    print("\n📊 AUGMENTATION SUMMARY:")
    total_copied = 0
    total_augmented = 0
    
    for class_name, report in balance_report.items():
        copied = report.get("train_copied", 0) + report.get("valid_copied", 0) + report.get("test_copied", 0)
        augmented = report.get("train_augmented", 0) + report.get("valid_augmented", 0) + report.get("test_augmented", 0)
        total_copied += copied
        total_augmented += augmented
    
    print(f"   • Images copied directly: {total_copied}")
    print(f"   • Images created via augmentation: {total_augmented}")
    
    print("\n📁 OUTPUT LOCATION:")
    print(f"   {OUTPUT_BASE}")
    print(f"   ├── train/ ({TARGET_TRAIN} images × {len(class_names)} classes = {TARGET_TRAIN * len(class_names)} images)")
    print(f"   ├── valid/ ({TARGET_VALID} images × {len(class_names)} classes = {TARGET_VALID * len(class_names)} images)")
    print(f"   └── test/  ({TARGET_TEST} images × {len(class_names)} classes = {TARGET_TEST * len(class_names)} images)")
    
    print("\n✅ Dataset cleaning and balancing complete!")
    print("=" * 70)


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    process_dataset()
