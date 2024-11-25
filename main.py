import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bars
import multiprocessing
from scipy.stats import ttest_rel  # For paired t-test


def main():
    # ============================
    # 1. Optimizing CPU Usage
    # ============================

    # Set the number of CPU threads to the number of physical cores
    num_cpus = multiprocessing.cpu_count()
    torch.set_num_threads(num_cpus)
    print(f"Number of CPU threads set to: {num_cpus}")

    # ============================
    # 2. Device Configuration
    # ============================

    # Device configuration: Use CPU as GPU is unavailable
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # ============================
    # 3. Data Transformations
    # ============================

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # ============================
    # 4. Loading and Filtering CIFAR-10
    # ============================

    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # Select specific classes
    selected_classes = ["bird", "cat", "deer", "dog", "frog"]
    class_to_idx = {cls: idx for idx, cls in enumerate(train_dataset.classes)}
    selected_class_indices = [class_to_idx[cls] for cls in selected_classes]
    print(
        f"Selected classes and their indices: {dict(zip(selected_classes, selected_class_indices))}"
    )

    # Function to filter dataset by selected class indices
    def filter_dataset(dataset, selected_indices):
        indices = [
            i for i, (_, label) in enumerate(dataset) if label in selected_indices
        ]
        return Subset(dataset, indices)

    filtered_train = filter_dataset(train_dataset, selected_class_indices)
    filtered_test = filter_dataset(test_dataset, selected_class_indices)
    combined_dataset = ConcatDataset([filtered_train, filtered_test])
    print(f"Total samples after filtering: {len(combined_dataset)}")

    # ============================
    # 5. DataLoader Configuration
    # ============================

    # Optimize DataLoader parameters
    batch_size = 32  # Reduced from 128 to manage memory usage
    num_workers = min(4, num_cpus)  # Set to min(4, available CPU cores)
    data_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,  # Disable pin_memory for CPU
    )
    print(
        f"DataLoader configured with batch_size={batch_size}, num_workers={num_workers}"
    )

    # ============================
    # 6. Loading DINO Models
    # ============================

    # Function to load DINO models
    def load_dino_model(version="v1"):
        """
        Loads DINO models. Ensure that the model names correspond to actual available models.
        """
        try:
            if version == "v1":
                # Example: Loading DINOv1 ViT-S/16 model
                model = torch.hub.load("facebookresearch/dino:main", "dino_vits16")
            elif version == "v2":
                # Example: Loading DINOv2 ViT-S/16 model
                # Note: DINOv2 may not be available via torch.hub; replace with actual loading mechanism
                model = torch.hub.load("facebookresearch/dinov2:main", "dinov2_vits14")
            else:
                raise ValueError("Unsupported DINO version. Choose 'v1' or 'v2'.")

            model.eval()
            model.to(device)
            return model
        except Exception as e:
            print(f"Error loading DINO {version}: {e}")
            raise

    # Load DINOv1 and DINOv2 models
    try:
        dino_v1 = load_dino_model("v1")
    except Exception as e:
        print(
            "Failed to load DINOv1 model. Please ensure the model name is correct and accessible."
        )
        exit(1)

    try:
        dino_v2 = load_dino_model("v2")
    except Exception as e:
        print(
            "Failed to load DINOv2 model. Please ensure the model name is correct and accessible."
        )
        exit(1)

    print("DINOv1 and DINOv2 models loaded successfully.")

    # ============================
    # 7. Embedding Extraction
    # ============================

    # Function to extract embeddings with progress tracking
    def extract_embeddings(model, data_loader):
        embeddings = []
        labels = []
        with torch.no_grad():
            for images, targets in tqdm(
                data_loader, desc=f"Extracting embeddings with {model}"
            ):
                images = images.to(device)
                features = model(images)

                # Handle different model output formats
                if isinstance(features, (tuple, list)):
                    features = features[0]
                if features.dim() > 2:
                    features = torch.flatten(features, start_dim=1)

                embeddings.append(features.cpu().numpy())
                labels.extend(targets.cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.array(labels)
        return embeddings, labels

    # Extract embeddings using DINOv1
    print("Extracting embeddings using DINOv1...")
    embeddings_v1, labels = extract_embeddings(dino_v1, data_loader)
    print(f"DINOv1 embeddings shape: {embeddings_v1.shape}")

    # Extract embeddings using DINOv2
    print("Extracting embeddings using DINOv2...")
    embeddings_v2, _ = extract_embeddings(dino_v2, data_loader)
    print(f"DINOv2 embeddings shape: {embeddings_v2.shape}")

    # ============================
    # 8. Principal Component Analysis (PCA)
    # ============================

    # Function to perform PCA
    def perform_pca(embeddings, n_components=50):
        pca = PCA(n_components=n_components, random_state=42)
        transformed = pca.fit_transform(embeddings)
        variance_explained = np.cumsum(pca.explained_variance_ratio_)
        return pca, transformed, variance_explained

    # Perform PCA on DINOv1 embeddings
    print("Performing PCA on DINOv1 embeddings...")
    pca_v1, transformed_v1, var_explained_v1 = perform_pca(embeddings_v1)
    print(f"PCA completed for DINOv1.")

    # Perform PCA on DINOv2 embeddings
    print("Performing PCA on DINOv2 embeddings...")
    pca_v2, transformed_v2, var_explained_v2 = perform_pca(embeddings_v2)
    print(f"PCA completed for DINOv2.")

    # ============================
    # 9. Visualization of Variance Explained
    # ============================

    # Function to plot cumulative variance explained
    def plot_variance(var_explained, title="Variance Explained by PCA"):
        plt.figure(figsize=(8, 6))
        plt.plot(var_explained, marker="o")
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Variance Explained")
        plt.title(title)
        plt.grid(True)
        plt.axhline(y=0.95, color="r", linestyle="--", label="95% Variance")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Plot cumulative variance for DINOv1
    plot_variance(var_explained_v1, "DINOv1: Cumulative Variance Explained")

    # Plot cumulative variance for DINOv2
    plot_variance(var_explained_v2, "DINOv2: Cumulative Variance Explained")

    # ============================
    # 10. Determining Number of Components for 95% Variance
    # ============================

    # Function to determine number of components for desired variance
    def components_for_variance(var_explained, threshold=0.95):
        return np.argmax(var_explained >= threshold) + 1

    components_v1 = components_for_variance(var_explained_v1)
    components_v2 = components_for_variance(var_explained_v2)
    print(f"DINOv1: {components_v1} components explain at least 95% variance.")
    print(f"DINOv2: {components_v2} components explain at least 95% variance.")

    # ============================
    # 11. Silhouette Score Calculation
    # ============================

    # Function to compute silhouette score
    def compute_silhouette(embeddings, labels, n_components=None):
        if n_components:
            pca = PCA(n_components=n_components, random_state=42)
            embeddings = pca.fit_transform(embeddings)
        score = silhouette_score(embeddings, labels)
        return score

    # Function to compute per-sample silhouette scores
    def compute_silhouette_samples_custom(embeddings, labels, n_components=None):
        if n_components:
            pca = PCA(n_components=n_components, random_state=42)
            embeddings = pca.fit_transform(embeddings)
        sample_scores = silhouette_samples(embeddings, labels)
        return sample_scores

    # Compute mean silhouette scores with 50 PCA components
    print("Computing mean Silhouette Scores with 50 PCA components...")
    score_v1 = compute_silhouette(embeddings_v1, labels, n_components=50)
    score_v2 = compute_silhouette(embeddings_v2, labels, n_components=50)
    print(f"DINOv1 Mean Silhouette Score (50 PCA components): {score_v1:.4f}")
    print(f"DINOv2 Mean Silhouette Score (50 PCA components): {score_v2:.4f}")

    # Compute per-sample silhouette scores with 50 PCA components
    print("Computing per-sample Silhouette Scores with 50 PCA components...")
    sample_scores_v1 = compute_silhouette_samples_custom(
        embeddings_v1, labels, n_components=50
    )
    sample_scores_v2 = compute_silhouette_samples_custom(
        embeddings_v2, labels, n_components=50
    )

    # ============================
    # 12. Paired t-Test on Silhouette Scores
    # ============================

    print("Performing paired t-test on per-sample Silhouette Scores...")

    # Ensure that sample_scores_v1 and sample_scores_v2 have the same length and correspond to the same samples
    assert len(sample_scores_v1) == len(
        sample_scores_v2
    ), "Mismatch in number of samples between models."

    # Perform paired t-test
    t_stat, p_value = ttest_rel(sample_scores_v1, sample_scores_v2)

    print(f"Paired t-test results:")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")

    # Interpret the results
    alpha = 0.05  # Common significance level
    if p_value < alpha:
        print(
            f"Since p-value ({p_value:.4f}) < alpha ({alpha}), we reject the null hypothesis."
        )
        print(
            "There is a statistically significant difference in Silhouette Scores between DINOv1 and DINOv2."
        )
    else:
        print(
            f"Since p-value ({p_value:.4f}) >= alpha ({alpha}), we fail to reject the null hypothesis."
        )
        print(
            "There is no statistically significant difference in Silhouette Scores between DINOv1 and DINOv2."
        )

    # ============================
    # 13. 2D PCA for Visualization
    # ============================

    # Function to plot 2D PCA embeddings
    def plot_pca_2d(embeddings_2d, labels, title="PCA 2D Projection"):
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=labels,
            cmap="tab10",
            s=10,
            alpha=0.7,
        )
        plt.legend(
            *scatter.legend_elements(),
            title="Classes",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )
        plt.title(title)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Perform and plot 2D PCA for DINOv1
    print("Performing 2D PCA on DINOv1 embeddings...")
    pca_2d_v1 = PCA(n_components=2, random_state=42)
    embeddings_v1_2d = pca_2d_v1.fit_transform(embeddings_v1)
    score_v1_2d = silhouette_score(embeddings_v1_2d, labels)
    print(f"DINOv1 Silhouette Score (2D): {score_v1_2d:.4f}")
    plot_pca_2d(embeddings_v1_2d, labels, "DINOv1 Embeddings PCA 2D")

    # Perform and plot 2D PCA for DINOv2
    print("Performing 2D PCA on DINOv2 embeddings...")
    pca_2d_v2 = PCA(n_components=2, random_state=42)
    embeddings_v2_2d = pca_2d_v2.fit_transform(embeddings_v2)
    score_v2_2d = silhouette_score(embeddings_v2_2d, labels)
    print(f"DINOv2 Silhouette Score (2D): {score_v2_2d:.4f}")
    plot_pca_2d(embeddings_v2_2d, labels, "DINOv2 Embeddings PCA 2D")


if __name__ == "__main__":
    main()
