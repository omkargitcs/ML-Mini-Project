# Deep Learning Feature Extraction & Siamese Networks

A computer vision pipeline implementing custom Siamese Networks, Autoencoders, and Transfer Learning backbones (ResNet50, VGG16) for feature extraction and similarity evaluation.

---

## 📁 Repository Structure

```text
.
├── data_pipeline.py            # Dataset loading and preprocessing utilities
├── custom_siamese_cnn.py       # Custom Siamese CNN model architecture & training
├── autoencoder_and_eval.py     # Unsupervised autoencoder & evaluation metrics
├── resnet50_features 1.py      # Feature extraction pipeline using ResNet50
├── best_siamese_model.pt       # Saved PyTorch weights for the Siamese model
├── autoencoder_embeddings.npz # Compressed embeddings from Autoencoder
├── custom_siamese_predictions.npz # Saved similarity prediction outputs
├── resnet50_embeddings.npz     # Feature vectors extracted via ResNet50
├── vgg16_embeddings.npz        # Feature vectors extracted via VGG16
└── MLProjectOutput.txt         # Execution logs and quantitative metrics
