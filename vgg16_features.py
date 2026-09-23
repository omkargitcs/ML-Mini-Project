
import gc
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
from data_pipeline import get_lfw_data_loaders


class VGG16FeatureExtractor(nn.Module):
    """
    Wraps pre-trained VGG16 to extract its penultimate 4,096-dimensional representation.
    """
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        # Download pre-trained weights automatically
        base_model = vgg16(weights=VGG16_Weights.DEFAULT)
        
        # Keep the 13 feature-extraction convolutional layers
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        
        # Bypassing the 1000-class head: Keep Linear(25088->4096) -> ReLU -> Dropout -> Linear(4096->4096)
        # We slice up to index 4, dropping the final Linear(4096 -> 1000) layer.
        self.embedding_layers = nn.Sequential(*list(base_model.classifier.children())[:5])
        
        # Freeze weights: In classical terms, this operates as a fixed Transformer
        # (no gradient updates computed during extraction)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten 3D spatial tensor to a 1D vector per image
        x = self.embedding_layers(x)
        return x  # Shape: (Batch_Size, 4096)


def extract_embeddings(model, dataloader, device):
    """
    Iterates over paired DataLoaders and projects both images into the 4096-D space.
    """
    model.eval()
    emb_a_list, emb_b_list, labels_list = [], [], []

    with torch.no_grad():  # Disables gradient tracking to minimize RAM/VRAM footprint
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


def run_vgg16_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"[OMKAR] Starting VGG16 Feature Extraction on Device: {device}")
    print(f"=======================================================")

    train_loader, val_loader, test_loader = get_lfw_data_loaders(batch_size=32)

    print("-> Loading pre-trained VGG16 backbone...")
    model = VGG16FeatureExtractor().to(device)

    print("-> Extracting VGG16 features for Train split...")
    train_a, train_b, train_y = extract_embeddings(model, train_loader, device)

    print("-> Extracting VGG16 features for Validation split...")
    val_a, val_b, val_y = extract_embeddings(model, val_loader, device)

    print("-> Extracting VGG16 features for Test split...")
    test_a, test_b, test_y = extract_embeddings(model, test_loader, device)

    output_filename = "vgg16_embeddings.npz"
    print(f"-> Saving extracted representations to {output_filename}...")
    np.savez_compressed(
        output_filename,
        train_a=train_a, train_b=train_b, train_y=train_y,
        val_a=val_a,     val_b=val_b,     val_y=val_y,
        test_a=test_a,   test_b=test_b,   test_y=test_y
    )

    print(f"   Successfully extracted: Train={train_a.shape}, Val={val_a.shape}, Test={test_a.shape}")
    print("   Freeing VGG16 GPU/System RAM allocations...")

    # Strict memory cleanup to avoid OOM errors across subsequent scripts
    del model
    del train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[OMKAR] VGG16 Pipeline Execution Complete.\n")


if __name__ == "__main__":
    run_vgg16_pipeline()
