# CIFAR-10 Data Analysis with DINO Models

This repository provides a pipeline for processing CIFAR-10 dataset using DINOv1 and DINOv2 models, extracting embeddings, performing PCA, and analyzing silhouette scores for clustering. The project demonstrates techniques for handling large datasets and analyzing embedding spaces.

## Features

1. **Optimized CPU Usage**: Utilizes all available CPU threads for efficient data loading and processing.
2. **Data Transformations**: Preprocesses CIFAR-10 data for use with DINO models.
3. **Filtering CIFAR-10 Classes**: Selects specific classes ("bird", "cat", "deer", "dog", "frog") for analysis.
4. **Loading DINO Models**: Loads and compares embeddings from DINOv1 and DINOv2 models.
5. **Principal Component Analysis (PCA)**: Reduces dimensionality of embeddings and visualizes variance explained.
6. **Silhouette Analysis**: Calculates silhouette scores for clustering quality.
7. **Statistical Tests**: Performs paired t-tests to compare embeddings from DINOv1 and DINOv2.
8. **2D Visualization**: Projects embeddings onto 2D space using PCA for visualization.

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- tqdm
- scipy

## Installation

Install the required Python packages:

```bash
pip install torch torchvision scikit-learn matplotlib tqdm scipy
```

## Usage

1. **Prepare Dataset**: The script downloads and preprocesses CIFAR-10 dataset automatically.
2. **Run the Script**: Execute the script to perform the complete pipeline:

```bash
python main.py
```

3. **View Results**: Outputs include embeddings, silhouette scores, PCA variance plots, and 2D visualizations.

## Key Functions

### Data Filtering

Filters CIFAR-10 dataset to include only specific classes.

```python
def filter_dataset(dataset, selected_indices):
    indices = [i for i, (_, label) in enumerate(dataset) if label in selected_indices]
    return Subset(dataset, indices)
```

### PCA

Performs PCA on embeddings.

```python
def perform_pca(embeddings, n_components=50):
    pca = PCA(n_components=n_components, random_state=42)
    transformed = pca.fit_transform(embeddings)
    variance_explained = np.cumsum(pca.explained_variance_ratio_)
    return pca, transformed, variance_explained
```

### Silhouette Analysis

Calculates silhouette scores.

```python
def compute_silhouette(embeddings, labels, n_components=None):
    if n_components:
        pca = PCA(n_components=n_components, random_state=42)
        embeddings = pca.fit_transform(embeddings)
    score = silhouette_score(embeddings, labels)
    return score
```

## Results

1. **Embedding Extraction**: Extracted embeddings using DINOv1 and DINOv2 models.
2. **PCA Variance Explained**: Determined number of components for 95% variance.
3. **Silhouette Scores**: Compared clustering quality of embeddings.
4. **2D Visualization**: Plotted PCA-reduced embeddings in 2D space.

## License

This project is licensed under the MIT License.
