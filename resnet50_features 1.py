

import gc
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from data_pipeline import get_lfw_data_loaders


class ResNet50FeatureExtractor(nn.Module):
    """
    Bypasses ResNet50's final fully connected layer to output the 2048-D latent representation.
    """
    def __init__(self):
        super(ResNet50FeatureExtractor, self).__init__()
        # Download official pre-trained weights
        base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Replace the terminal classification head (1000 classes) with an Identity passthrough.
        # This keeps the AdaptiveAvgPool2d intact, outputting a 2,048-dimensional vector.
        base_model.fc = nn.Identity()
        self.backbone = base_model

        # Freeze network parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.backbone(x)  # Shape: (Batch_Size, 2048)


def extract_embeddings(model, dataloader, device):
    """
    Passes data batches through the frozen backbone to construct feature matrices.
    """
    model.eval()
    emb_a_list, emb_b_list, labels_list = [], [], []

    with torch.no_grad():
        for batch_idx, (img_a, img_b, labels) in enumerate(dataloader):
            img_a = img_a.to(device)
            img_b = img_b.to(device)

            out_a = model(img_a).cpu().numpy()
            out_b = model(img_b).cpu().numpy()

            emb_a_list.append(out_a)
            emb_b_list.append(out_b)
            labels_list.append(labels.numpy())

    return (
        np.concatenate(emb_a_list, axis=0),
        np.concatenate(emb_b_list, axis=0),
        np.concatenate(labels_list, axis=0)
    )


def run_resnet50_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"[FARHAN] Starting ResNet50 Feature Extraction on Device: {device}")
    print(f"=======================================================")

    train_loader, val_loader, test_loader = get_lfw_data_loaders(batch_size=32)

    print("-> Loading pre-trained ResNet50 backbone...")
    model = ResNet50FeatureExtractor().to(device)

    print("-> Extracting ResNet50 features for Train split...")
    train_a, train_b, train_y = extract_embeddings(model, train_loader, device)

    print("-> Extracting ResNet50 features for Validation split...")
    val_a, val_b, val_y = extract_embeddings(model, val_loader, device)

    print("-> Extracting ResNet50 features for Test split...")
    test_a, test_b, test_y = extract_embeddings(model, test_loader, device)

    output_filename = "resnet50_embeddings.npz"
    print(f"-> Saving extracted representations to {output_filename}...")
    np.savez_compressed(
        output_filename,
        train_a=train_a, train_b=train_b, train_y=train_y,
        val_a=val_a,     val_b=val_b,     val_y=val_y,
        test_a=test_a,   test_b=test_b,   test_y=test_y
    )

    print(f"   Successfully extracted: Train={train_a.shape}, Val={val_a.shape}, Test={test_a.shape}")
    print("   Freeing ResNet50 GPU/System RAM allocations...")

    # Strict memory cleanup
    del model
    del train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[FARHAN] ResNet50 Pipeline Execution Complete.\n")


if __name__ == "__main__":
    run_resnet50_pipeline()
