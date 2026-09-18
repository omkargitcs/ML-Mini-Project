"""
FILE 5: autoencoder_and_eval.py (Riya's Component & Final Evaluation Suite)
================================================================================
CLASSICAL ML ANALOGY:
1. AUTOENCODER (Unsupervised Feature Extractor):
   An Autoencoder is the non-linear deep learning counterpart of Principal Component
   Analysis (PCA). An encoder compresses images into a 128-D bottleneck, and a
   decoder reconstructs them under Mean Squared Error (MSE) loss. The bottleneck
   serves as our low-dimensional feature representation.

2. BENCHMARK SUITE:
   Acts like an automated Model Selection and Assessment workflow:
   - For feature extractors (VGG16, ResNet50, Autoencoder), it computes pairwise
     Cosine Similarities:
     
         $$\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
         
   - Uses the Validation Set to find the optimal decision threshold via ROC search.
   - Evaluates all 4 pipelines on the unseen Test Set and reports:
     Accuracy, Precision, Recall, F1-Score, and Confusion Matrix metrics.
================================================================================
"""

import gc
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve
)
from sklearn.metrics.pairwise import paired_cosine_distances
from data_pipeline import get_lfw_data_loaders


# =====================================================================
# 1. RIYA'S AUTOENCODER ARCHITECTURE
# =====================================================================
class ConvAutoencoder(nn.Module):
    """
    Non-linear dimensionality reduction:
    Input: (3, 224, 224) -> Latent Bottleneck: (128,) -> Reconstruction: (3, 224, 224)
    """
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),   # -> 16 x 112 x 112
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # -> 32 x 56 x 56
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # -> 64 x 28 x 28
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))                 # -> 64 x 4 x 4 = 1024
        )
        self.to_latent = nn.Linear(64 * 4 * 4, 128)

        # Decoder
        self.from_latent = nn.Linear(128, 64 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # -> 32 x 8 x 8
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=4, padding=0),   # -> 16 x 32 x 32
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 7, stride=7, padding=0),    # -> 3 x 224 x 224
            nn.Tanh()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, 1)
        return self.to_latent(h)

    def decode(self, z):
        h = self.from_latent(z).view(-1, 64, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


def train_autoencoder(epochs=6, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"[RIYA] Training Unsupervised Autoencoder on Device: {device}")
    print(f"=======================================================")

    train_loader, val_loader, test_loader = get_lfw_data_loaders(batch_size=batch_size)
    ae = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ae.parameters(), lr=1e-3)

    for epoch in range(epochs):
        ae.train()
        total_loss = 0.0
        for img_a, _, _ in train_loader:
            img_a = img_a.to(device)
            optimizer.zero_grad()
            reconstruction = ae(img_a)
            loss = criterion(reconstruction, img_a)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(img_a)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"   Epoch [{epoch+1:02d}/{epochs:02d}] -> Reconstruction Loss (MSE): {avg_loss:.5f}")

    print("-> Extracting Autoencoder Bottleneck representations...")
    ae.eval()

    def get_latents(dataloader):
        lat_a, lat_b, targets = [], [], []
        with torch.no_grad():
            for img_a, img_b, y in dataloader:
                z_a = ae.encode(img_a.to(device)).cpu().numpy()
                z_b = ae.encode(img_b.to(device)).cpu().numpy()
                lat_a.append(z_a)
                lat_b.append(z_b)
                targets.append(y.numpy())
        return np.concatenate(lat_a), np.concatenate(lat_b), np.concatenate(targets)

    val_a, val_b, val_y = get_latents(val_loader)
    test_a, test_b, test_y = get_latents(test_loader)

    np.savez_compressed(
        "autoencoder_embeddings.npz",
        val_a=val_a, val_b=val_b, val_y=val_y,
        test_a=test_a, test_b=test_b, test_y=test_y
    )
    print("-> Autoencoder embeddings written to autoencoder_embeddings.npz")

    del ae
    del train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =====================================================================
# 2. MASTER EVALUATION BENCHMARK SUITE
# =====================================================================
def optimize_threshold_by_roc(sim_scores, y_true):
    """
    Finds the optimal threshold on the Validation Set using the Youden Index (J = TPR - FPR).
    """
    fpr, tpr, thresholds = roc_curve(y_true, sim_scores)
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]


def evaluate_cosine_pipeline(name, npz_path):
    """
    Evaluates representation-based pipelines (VGG16, ResNet50, Autoencoder) using Cosine Similarity.
    """
    data = np.load(npz_path)
    
    # Compute Cosine Similarity: 1 - Cosine Distance
    val_sim = 1.0 - paired_cosine_distances(data["val_a"], data["val_b"])
    test_sim = 1.0 - paired_cosine_distances(data["test_a"], data["test_b"])
    
    # Tune threshold on validation data
    best_thresh = optimize_threshold_by_roc(val_sim, data["val_y"])
    
    # Predict on test data using optimal threshold
    test_preds = (test_sim >= best_thresh).astype(int)
    y_test = data["test_y"].astype(int)

    return compute_metrics(name, y_test, test_preds, best_thresh)


def evaluate_siamese_classifier(name, npz_path):
    """
    Evaluates the Custom Siamese Network using predicted probabilities.
    """
    data = np.load(npz_path)
    y_prob = data["y_prob"]
    y_test = data["y_true"].astype(int)

    test_preds = (y_prob >= 0.5).astype(int)
    return compute_metrics(name, y_test, test_preds, threshold=0.50)


def compute_metrics(name, y_true, y_pred, threshold):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "Model": name,
        "Threshold": threshold,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn
    }


def run_benchmark():
    print(f"\n=======================================================")
    print(f"MASTER EVALUATION BENCHMARK (UNSEEN TEST SPLIT)")
    print(f"=======================================================")

    results = []

    if os.path.exists("vgg16_embeddings.npz"):
        results.append(evaluate_cosine_pipeline("VGG16 (Penultimate FC)", "vgg16_embeddings.npz"))
    else:
        print("[!] vgg16_embeddings.npz not found. Run vgg16_features.py first.")

    if os.path.exists("resnet50_embeddings.npz"):
        results.append(evaluate_cosine_pipeline("ResNet50 (AvgPool)", "resnet50_embeddings.npz"))
    else:
        print("[!] resnet50_embeddings.npz not found. Run resnet50_features.py first.")

    if os.path.exists("autoencoder_embeddings.npz"):
        results.append(evaluate_cosine_pipeline("Conv-Autoencoder", "autoencoder_embeddings.npz"))
    else:
        print("[!] autoencoder_embeddings.npz not found. Generating now...")
        train_autoencoder(epochs=5)
        results.append(evaluate_cosine_pipeline("Conv-Autoencoder", "autoencoder_embeddings.npz"))

    if os.path.exists("custom_siamese_predictions.npz"):
        results.append(evaluate_siamese_classifier("Custom Siamese CNN", "custom_siamese_predictions.npz"))
    else:
        print("[!] custom_siamese_predictions.npz not found. Run custom_siamese_cnn.py first.")

    # Format results as a Markdown table
    print("\n### Performance Comparison Table\n")
    header = "| Model Architecture | Threshold | Accuracy | Precision | Recall | F1-Score | TP | TN | FP | FN |"
    divider = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    print(header)
    print(divider)

    for r in results:
        print(f"| {r['Model']} | {r['Threshold']:.3f} | {r['Accuracy']:.4f} | "
              f"{r['Precision']:.4f} | {r['Recall']:.4f} | {r['F1']:.4f} | "
              f"{r['TP']} | {r['TN']} | {r['FP']} | {r['FN']} |")


if __name__ == "__main__":
    train_autoencoder(epochs=5)
    run_benchmark()