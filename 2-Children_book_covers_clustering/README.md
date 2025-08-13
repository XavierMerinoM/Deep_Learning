# Visual Style Clustering of Children's Book Covers

## Summary of Project

This project explores unsupervised clustering of children's and teen book covers based on visual style. Two distinct approaches are implemented and compared:

1. **Autoencoder-based Feature Extraction**: A custom CNN autoencoder learns latent representations of book covers, which are then clustered.
2. **Image + Latent Fusion**: Combines raw image features (after PCA) with latent vectors from the autoencoder to form enriched representations for clustering.

The goal is to determine which method yields more coherent and meaningful clusters of visual styles.

---

## Preprocessing Steps

- **Dataset**: Kaggle - [6000 Children and Teen Book Covers](https://www.kaggle.com/datasets/thomaskonstantin/6000-children-and-teen-book-covers)
- **Image preprocessing using OpenCV:**
  - Resize to 224×224
  - Normalize pixel values
  - Convert BGR to RGB
- **Data split**: 60% training, 20% validation, 20% testing
- Batch loading for memory efficiency
- Visualization of original vs preprocessed images

---

## Model Architecture

### Model 1: Autoencoder

- **Encoder**:
  - 4 convolutional blocks with batch normalization and L2 regularization
  - Attention mechanism before global average pooling
  - Dense layers leading to a 128-dimensional latent space
- **Decoder**:
  - Dense layers with batch normalization
  - Reshape and transpose convolutions with upsampling
  - Final output: reconstructed image

### Model 2: Latent Fusion

- Combines:
  - Flattened raw image features (after PCA)
  - Latent vectors from the autoencoder
- Resulting fused feature vector used for clustering

---

## Training Process

### Framework

- TensorFlow + Keras
- Mixed precision enabled for optimization
- Trained on Google Colab (GPU)

### Loss Function

- Mean Squared Error (MSE)

### Optimizer

- Adam with AMSGrad variant
- Learning rate: 0.0005

### Metrics

- Training and validation loss tracked over 20 epochs
- Visual convergence plots generated

---

## Results

### Model 1: Autoencoder

- **Optimal clusters**: 4
- **Metrics**:
  - Silhouette Score: **0.4131**
  - Davies-Bouldin Index: **0.9473**
  - Calinski-Harabasz Index: **NaN**
- Cluster sizes: [532, 2816, 1365, 287]
- t-SNE visualization used for cluster inspection

### Model 2: Latent Fusion

- **Optimal clusters**: 4
- **Metrics**:
  - Silhouette Score: **0.2422**
  - Davies-Bouldin Index: **1.4101**
  - Calinski-Harabasz Index: **1214.39**
- Cluster sizes: [2809, 285, 541, 1365]
- t-SNE visualization used for cluster inspection

### Comparative Evaluation

- Model 1 outperforms Model 2 in silhouette and Davies-Bouldin scores, indicating better-defined and more compact clusters.
- Model 2 shows higher Calinski-Harabasz index, suggesting better separation in some configurations.