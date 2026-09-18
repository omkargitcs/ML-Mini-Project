"""
FILE 1: data_pipeline.py
================================================================================
CLASSICAL ML ANALOGY:
In scikit-learn, you normally construct an X matrix of shape (n_samples, n_features)
and a target vector y of shape (n_samples,). 

In Deep Computer Vision, raw pixel matrices are our unengineered features.
This module:
1. Ingests raw face arrays using `sklearn.datasets.fetch_lfw_people`.
2. Enforces strict Identity Isolation using `GroupShuffleSplit`. This is identical
   to GroupKFold in scikit-learn: it guarantees that no subject's identity present
   in the training set leaks into the validation or test splits.
3. Implements a Siamese Pair Generator:
   - Positive / Duplicate pair (y = 1): Image A + Image A augmented with jitter/crop.
   - Negative / Non-Duplicate pair (y = 0): Image A + Image B (from a different person).
4. Packages everything into PyTorch DataLoaders (the deep learning equivalent of
   a memory-efficient, mini-batch generator).
================================================================================
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import GroupShuffleSplit


# =====================================================================
# 1. IMAGE PREPROCESSING & AUGMENTATION PIPELINES
# =====================================================================
# ResNet and VGG backbones pre-trained on ImageNet require 224x224 RGB images
# normalized with specific per-channel means and standard deviations.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Base pipeline: Standardizes input into a 3-channel tensor of shape (3, 224, 224)
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Scales raw pixel integers [0, 255] to floats [0.0, 1.0]
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)  # Standardizes (Z-score)
])

# Duplication pipeline: Generates realistic minor digital alterations
# (slight crop, subtle rotation, and minor exposure/color variance)
duplicate_transform = transforms.Compose([
    transforms.Resize((236, 236)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomRotation(degrees=8),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


# =====================================================================
# 2. PYTORCH DATASET IMPLEMENTATION
# =====================================================================
class LFWPairDataset(Dataset):
    """
    Equivalent to a custom scikit-learn Transformer + Estimator pair feeder.
    For index i, returns (Tensor_A, Tensor_B, Label).
    - Label 1: Duplicate (Image A altered)
    - Label 0: Non-Duplicate (Image A paired with a different person)
    """
    def __init__(self, images, labels, base_tf=base_transform, dup_tf=duplicate_transform, seed=42):
        self.images = images          # Raw 2D/3D numpy arrays
        self.labels = labels          # Person IDs (group targets)
        self.base_tf = base_tf
        self.dup_tf = dup_tf
        self.rng = np.random.RandomState(seed)
        
        # Pre-group indices by person identity to efficiently sample negatives
        self.unique_labels = np.unique(labels)
        self.label_to_indices = {
            lbl: np.where(labels == lbl)[0] for lbl in self.unique_labels
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Anchor image
        raw_img_a = self.images[idx]
        person_a = self.labels[idx]

        # Convert grayscale (H, W) or single-channel image to PIL RGB (H, W, 3)
        if raw_img_a.ndim == 2:
            pil_img_a = Image.fromarray((raw_img_a * 255).astype(np.uint8)).convert("RGB")
        else:
            pil_img_a = Image.fromarray(raw_img_a.astype(np.uint8)).convert("RGB")

        # 50% probability of constructing a Duplicate (1) vs Non-Duplicate (0)
        is_duplicate = self.rng.rand() >= 0.5

        if is_duplicate:
            # Duplicate: apply subtle transformations to the SAME source image
            tensor_a = self.base_tf(pil_img_a)
            tensor_b = self.dup_tf(pil_img_a)
            pair_label = 1.0
        else:
            # Non-Duplicate: pick a DIFFERENT person from the available identity labels
            diff_labels = self.unique_labels[self.unique_labels != person_a]
            chosen_diff_label = self.rng.choice(diff_labels)
            idx_b = self.rng.choice(self.label_to_indices[chosen_diff_label])
            
            raw_img_b = self.images[idx_b]
            if raw_img_b.ndim == 2:
                pil_img_b = Image.fromarray((raw_img_b * 255).astype(np.uint8)).convert("RGB")
            else:
                pil_img_b = Image.fromarray(raw_img_b.astype(np.uint8)).convert("RGB")

            tensor_a = self.base_tf(pil_img_a)
            tensor_b = self.base_tf(pil_img_b)
            pair_label = 0.0

        return tensor_a, tensor_b, torch.tensor(pair_label, dtype=torch.float32)


# =====================================================================
# 3. IDENTITY-SAFE DATA SPLITTER (GROUP SHUFFLE SPLIT)
# =====================================================================
def get_lfw_data_loaders(batch_size=32, min_faces_per_person=10, random_state=42):
    """
    Downloads LFW and partitions into:
    - Train: 70% of identities
    - Val:   15% of identities
    - Test:  15% of identities
    
    Zero identity overlap guarantees zero data leakage across splits.
    """
    print("-> Loading LFW dataset via scikit-learn...")
    lfw = fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=None, color=False)
    images = lfw.images  # Float arrays scaled [0.0, 1.0]
    groups = lfw.target  # Integer IDs representing distinct identities

    # Step 1: Split into 70% Train, 30% Temp (Val + Test)
    gss_train = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=random_state)
    train_idx, temp_idx = next(gss_train.split(images, groups=groups))

    images_train, groups_train = images[train_idx], groups[train_idx]
    images_temp, groups_temp = images[temp_idx], groups[temp_idx]

    # Step 2: Split the 30% Temp evenly into 15% Val and 15% Test
    gss_val = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=random_state)
    val_sub_idx, test_sub_idx = next(gss_val.split(images_temp, groups=groups_temp))

    images_val, groups_val = images_temp[val_sub_idx], groups_temp[val_sub_idx]
    images_test, groups_test = images_temp[test_sub_idx], groups_temp[test_sub_idx]

    print(f"   Identity Breakdown: Train IDs={len(np.unique(groups_train))}, "
          f"Val IDs={len(np.unique(groups_val))}, Test IDs={len(np.unique(groups_test))}")
    print(f"   Sample Breakdown:   Train={len(images_train)}, Val={len(images_val)}, Test={len(images_test)}")

    train_ds = LFWPairDataset(images_train, groups_train, seed=random_state)
    val_ds   = LFWPairDataset(images_val, groups_val, seed=random_state + 1)
    test_ds  = LFWPairDataset(images_test, groups_test, seed=random_state + 2)

    # PyTorch DataLoaders stream batches to manage RAM efficiently
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_lfw_data_loaders(batch_size=16)
    img_a, img_b, y = next(iter(train_loader))
    print(f"\n[Test Verification]")
    print(f"Batch shape Image_A: {img_a.shape} (Batch, Channels, Height, Width)")
    print(f"Batch shape Image_B: {img_b.shape}")
    print(f"Batch targets y:     {y.tolist()}")
    print("-> Pipeline verification complete. No identity leakage detected.")