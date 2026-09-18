"""
FILE 4: custom_siamese_cnn.py (Sushanth's Component)
================================================================================
CLASSICAL ML ANALOGY:
Rather than taking fixed embeddings from an external model, a Siamese CNN learns
the feature transformation pipeline directly from the training data:
1. A small, customized CNN maps an image to a compact 128-dimensional embedding.
2. Both Image_A and Image_B are passed through the EXACT same network (shared weights).
3. We compute their absolute difference vector: |Embedding_A - Embedding_B|.
4. A final dense classification layer maps this difference to a single probability
   using the Sigmoid function—identical to training a scikit-learn `LogisticRegression`
   model on top of engineered feature differences.
================================================================================
"""

import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data_pipeline import get_lfw_data_loaders


class BaseEmbeddingNet(nn.Module):
    """
    Sub-network: Compresses a 3x224x224 raw image into a 128-dimensional vector.
    """
    def __init__(self):
        super(BaseEmbeddingNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 224x224 -> 112x112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 112x112 -> 56x56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 56x56 -> 28x28

            # Global Average Pooling collapses (128, 28, 28) to (128, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, 128)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x  # Shape: (Batch_Size, 128)


class SiameseNetwork(nn.Module):
    """
    Dual-stream architecture using shared weights for pairwise verification.
    """
    def __init__(self, embedding_net):
        super(SiameseNetwork, self).__init__()
        self.embedding_net = embedding_net
        
        # Classification head: Logistic Regression over the embedding disparity vector
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Produces probabilities in range [0, 1]
        )

    def forward_one(self, x):
        return self.embedding_net(x)

    def forward(self, x1, x2):
        emb1 = self.forward_one(x1)
        emb2 = self.forward_one(x2)
        # Compute absolute vector distance: |u - v|
        diff = torch.abs(emb1 - emb2)
        probability = self.classifier(diff)
        return probability.squeeze(-1)


def train_siamese_model(epochs=10, batch_size=32, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"[SUSHANTH] Training Custom Siamese CNN on Device: {device}")
    print(f"=======================================================")

    train_loader, val_loader, test_loader = get_lfw_data_loaders(batch_size=batch_size)

    base_net = BaseEmbeddingNet()
    siamese_net = SiameseNetwork(base_net).to(device)

    # Loss: Binary Cross Entropy (log-loss in classical ML)
    criterion = nn.BCELoss()
    # Optimizer: Adam (Stochastic Gradient Descent with adaptive momentum)
    optimizer = optim.Adam(siamese_net.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")

    # Training loop
    for epoch in range(epochs):
        siamese_net.train()
        running_loss = 0.0

        for img_a, img_b, targets in train_loader:
            img_a, img_b, targets = img_a.to(device), img_b.to(device), targets.to(device)

            optimizer.zero_grad()
            predictions = siamese_net(img_a, img_b)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(targets)

        train_loss = running_loss / len(train_loader.dataset)

        # Validation step
        siamese_net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img_a, img_b, targets in val_loader:
                img_a, img_b, targets = img_a.to(device), img_b.to(device), targets.to(device)
                predictions = siamese_net(img_a, img_b)
                loss = criterion(predictions, targets)
                val_loss += loss.item() * len(targets)

        val_loss /= len(val_loader.dataset)
        print(f"   Epoch [{epoch+1:02d}/{epochs:02d}] -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(siamese_net.state_dict(), "best_siamese_model.pt")

    print("-> Model training complete. Evaluating on Unseen Test Split...")
    siamese_net.load_state_dict(torch.load("best_siamese_model.pt"))
    siamese_net.eval()

    test_preds, test_targets = [], []
    with torch.no_grad():
        for img_a, img_b, targets in test_loader:
            img_a, img_b = img_a.to(device), img_b.to(device)
            preds = siamese_net(img_a, img_b)
            test_preds.append(preds.cpu().numpy())
            test_targets.append(targets.numpy())

    y_prob = np.concatenate(test_preds)
    y_true = np.concatenate(test_targets)

    # Save outputs for evaluation
    np.savez_compressed("custom_siamese_predictions.npz", y_prob=y_prob, y_true=y_true)
    print("-> Predictions successfully written to custom_siamese_predictions.npz")

    # Clean up memory
    del siamese_net, base_net
    del train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[SUSHANTH] Siamese CNN Pipeline Execution Complete.\n")


if __name__ == "__main__":
    train_siamese_model(epochs=8, batch_size=32)