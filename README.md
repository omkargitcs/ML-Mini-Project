# Deep Learning Feature Extraction & Siamese Network Models

A comprehensive computer vision repository implementing custom Siamese Networks, Autoencoders, and Transfer Learning backbones (ResNet50, VGG16) for feature extraction, representation learning, and similarity evaluation.

---

## 📌 Repository Overview

This project provides end-to-end pipelines for image processing, feature vector generation, model training, and performance evaluation:

* **Data Pipeline (`data_pipeline.py`):** Preprocesses, transforms, and loads image datasets for model ingestion[cite: 11].
* **Custom Siamese Network (`custom_siamese_cnn.py`):** Defines and trains a custom convolutional Siamese architecture for pairwise similarity learning[cite: 11].
* **Autoencoder Pipeline (`autoencoder_and_eval.py`):** Implements unsupervised feature representation via Autoencoders along with evaluation metrics[cite: 11].
* **Transfer Learning (`resnet50_features 1.py`):** Extracts deep feature representations using pre-trained deep convolutional networks[cite: 11].

---

## 📁 Repository Structure

```text
.
├── data_pipeline.py            # Dataset loading and preprocessing utilities
├── custom_siamese_cnn.py       # Custom Siamese CNN model training script
├── autoencoder_and_eval.py     # Autoencoder architecture and evaluation script
├── resnet50_features 1.py      # Feature extraction script using ResNet50
├── best_siamese_model.pt       # Saved weights for the best-performing Siamese model
├── autoencoder_embeddings.npz # Saved compressed embeddings from Autoencoder
├── custom_siamese_predictions.npz # Output predictions from Siamese network
├── resnet50_embeddings.npz     # Extracted feature vectors using ResNet50
├── vgg16_embeddings.npz        # Extracted feature vectors using VGG16
└── MLProjectOutput.txt         # Collected execution logs and metrics output
```[cite: 11]

---

## 🛠️ Requirements & Setup

Ensure you have Python 3.8+ installed along with the following core dependencies:

```bash
pip install torch torchvision numpy scipy matplotlib scikit-learn
