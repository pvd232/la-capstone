import os
import datetime
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend if saving plots
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import ttest_rel
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torchvision
import torchvision.transforms as transforms
import multiprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from pathlib import Path


def is_cifar10_downloaded(root):
    """
    Checks if the CIFAR-10 dataset is already downloaded in the specified root directory.

    Parameters:
    - root (str): The root directory where the dataset should be stored.

    Returns:
    - bool: True if the dataset is found, False otherwise.
    """
    cifar10_folder = Path(root) / "cifar-10-batches-py"
    return cifar10_folder.exists()


def plot_variance(var_explained, title="Variance Explained by PCA", save_path=None):
    """
    Plots the cumulative variance explained by PCA components.

    Parameters:
    - var_explained (np.ndarray): Cumulative variance explained by each PCA component.
    - title (str): Title of the plot.
    - save_path (str or Path): Path to save the plot image.
    """
    plt.figure(figsize=(10, 8))
    plt.plot(var_explained, marker="o", linestyle="-", color="b")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Variance Explained")
    plt.title(title)
    plt.grid(True)
    plt.axhline(y=0.95, color="r", linestyle="--", label="95% Variance")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_pca_2d(
    embeddings_2d, labels, label_to_name, title="PCA 2D Projection", save_path=None
):
    """
    Plots a 2D PCA scatter plot with class names in the legend.

    Parameters:
    - embeddings_2d (np.ndarray): 2D embeddings after PCA transformation.
    - labels (np.ndarray): Numerical labels corresponding to each data point.
    - label_to_name (dict): Mapping from numerical labels to class names.
    - title (str): Title of the plot.
    - save_path (str or Path): Path to save the plot image.
    """
    plt.figure(figsize=(12, 10))

    # Generate a color map with distinct colors for each class
    cmap = plt.get_cmap("tab10")
    colors = {label: cmap(i) for i, label in enumerate(sorted(label_to_name.keys()))}

    # Plot each class separately
    for label, name in label_to_name.items():
        idx = labels == label
        plt.scatter(
            embeddings_2d[idx, 0],
            embeddings_2d[idx, 1],
            label=name,
            color=colors[label],
            s=50,  # Increased size for better visibility
            alpha=0.7,
        )

    # Create a legend with class names
    plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.title(title, fontsize=16)
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Principal Component 2", fontsize=14)
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def components_for_variance(var_explained, threshold=0.95):
    """
    Determines the minimum number of PCA components required to explain at least the specified threshold of variance.

    Parameters:
    - var_explained (np.ndarray): Cumulative variance explained by PCA components.
    - threshold (float): Desired cumulative variance threshold (default is 0.95 for 95%).

    Returns:
    - int: Number of components needed to reach the threshold. Returns the total number of components if the threshold isn't met.
    """
    # Find indices where cumulative variance meets or exceeds the threshold
    indices = np.where(var_explained >= threshold)[0]

    if len(indices) > 0:
        # Return the first index where the condition is met, plus one for count
        return indices[0] + 1
    else:
        # If threshold not met, return the total number of components
        return len(var_explained)


def extract_embeddings(model, data_loader, device):
    """
    Extracts embeddings from the dataset using the specified model.

    Parameters:
    - model (torch.nn.Module): The model to use for embedding extraction.
    - data_loader (DataLoader): DataLoader for the dataset.
    - device (torch.device): Device to perform computations on.

    Returns:
    - embeddings (np.ndarray): Extracted embeddings.
    - labels (np.ndarray): Corresponding labels.
    """
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, targets in tqdm(
            data_loader,
            desc=f"Extracting embeddings with {model.__class__.__name__}",
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


def perform_pca(embeddings, n_components=50):
    """
    Performs PCA on the embeddings.

    Parameters:
    - embeddings (np.ndarray): High-dimensional embeddings.
    - n_components (int): Number of PCA components to retain.

    Returns:
    - pca (PCA object): Fitted PCA object.
    - transformed (np.ndarray): PCA-transformed embeddings.
    - variance_explained (np.ndarray): Cumulative variance explained by the components.
    """
    pca = PCA(n_components=n_components, random_state=42)
    transformed = pca.fit_transform(embeddings)
    variance_explained = np.cumsum(pca.explained_variance_ratio_)
    return pca, transformed, variance_explained


def compute_silhouette(embeddings, labels, n_components=None):
    """
    Computes the mean Silhouette Score for the embeddings.

    Parameters:
    - embeddings (np.ndarray): Embeddings to evaluate.
    - labels (np.ndarray): True labels for the embeddings.
    - n_components (int, optional): Number of PCA components to reduce to before computing.

    Returns:
    - score (float): Mean Silhouette Score.
    """
    if n_components:
        pca = PCA(n_components=n_components, random_state=42)
        embeddings = pca.fit_transform(embeddings)
    score = silhouette_score(embeddings, labels)
    return score


def compute_silhouette_samples_custom(embeddings, labels, n_components=None):
    """
    Computes the Silhouette Scores for each sample.

    Parameters:
    - embeddings (np.ndarray): Embeddings to evaluate.
    - labels (np.ndarray): True labels for the embeddings.
    - n_components (int, optional): Number of PCA components to reduce to before computing.

    Returns:
    - sample_scores (np.ndarray): Silhouette Scores for each sample.
    """
    if n_components:
        pca = PCA(n_components=n_components, random_state=42)
        embeddings = pca.fit_transform(embeddings)
    sample_scores = silhouette_samples(embeddings, labels)
    return sample_scores


def main():
    # ============================
    # 1. Optimizing CPU Usage
    # ============================
    num_cpus = multiprocessing.cpu_count()
    torch.set_num_threads(num_cpus)
    print(f"Number of CPU threads set to: {num_cpus}")

    # ============================
    # 2. Device Configuration
    # ============================
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
    root_dir = "./data"  # Define your root directory

    # Check if CIFAR-10 is already downloaded
    if is_cifar10_downloaded(root_dir):
        print(f"CIFAR-10 dataset already exists in '{root_dir}'. Skipping download.")
        download_flag = False
    else:
        print(f"CIFAR-10 dataset not found in '{root_dir}'. Starting download...")
        download_flag = True

    # Load the datasets with the determined download flag
    try:
        train_dataset = torchvision.datasets.CIFAR10(
            root=root_dir, train=True, download=download_flag, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=root_dir, train=False, download=download_flag, transform=transform
        )
    except Exception as e:
        print(f"Error loading CIFAR-10 dataset: {e}")
        exit(1)

    selected_classes = ["bird", "cat", "deer", "dog", "frog"]
    class_to_idx = {cls: idx for idx, cls in enumerate(train_dataset.classes)}
    selected_class_indices = [class_to_idx[cls] for cls in selected_classes]
    print(
        f"Selected classes and their indices: {dict(zip(selected_classes, selected_class_indices))}"
    )

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
    batch_size = 32  # Adjust as needed
    num_workers = min(4, num_cpus)
    data_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    print(
        f"DataLoader configured with batch_size={batch_size}, num_workers={num_workers}"
    )

    # ============================
    # 6. Loading DINO Models
    # ============================
    def load_dino_model(version="v1"):
        """
        Loads the specified DINO model.

        Parameters:
        - version (str): 'v1' or 'v2'.

        Returns:
        - model (torch.nn.Module): Loaded DINO model.
        """
        try:
            if version == "v1":
                model = torch.hub.load("facebookresearch/dino:main", "dino_vits16")
            elif version == "v2":
                model = torch.hub.load("facebookresearch/dinov2:main", "dinov2_vits14")
            else:
                raise ValueError("Unsupported DINO version. Choose 'v1' or 'v2'.")
            model.eval()
            model.to(device)
            return model
        except Exception as e:
            print(f"Error loading DINO {version}: {e}")
            raise

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
    # 7. Embedding Extraction and Persistence
    # ============================
    embeddings_v1_path = Path("embeddings_v1.npy")
    embeddings_v2_path = Path("embeddings_v2.npy")
    labels_path = Path("labels.npy")

    if (
        embeddings_v1_path.exists()
        and embeddings_v2_path.exists()
        and labels_path.exists()
    ):
        print("Embeddings and labels found. Loading from disk...")
        embeddings_v1 = np.load(embeddings_v1_path)
        embeddings_v2 = np.load(embeddings_v2_path)
        labels = np.load(labels_path)
        print(f"Loaded embeddings_v1 shape: {embeddings_v1.shape}")
        print(f"Loaded embeddings_v2 shape: {embeddings_v2.shape}")
        print(f"Loaded labels shape: {labels.shape}")
    else:
        print("Embeddings not found. Extracting embeddings...")

        # Extract embeddings using DINOv1
        print("Extracting embeddings using DINOv1...")
        try:
            embeddings_v1, labels = extract_embeddings(dino_v1, data_loader, device)
            print(f"DINOv1 embeddings shape: {embeddings_v1.shape}")
        except Exception as e:
            print(f"Error during DINOv1 embedding extraction: {e}")
            exit(1)

        # Extract embeddings using DINOv2
        print("Extracting embeddings using DINOv2...")
        try:
            embeddings_v2, _ = extract_embeddings(dino_v2, data_loader, device)
            print(f"DINOv2 embeddings shape: {embeddings_v2.shape}")
        except Exception as e:
            print(f"Error during DINOv2 embedding extraction: {e}")
            exit(1)

        # Save embeddings and labels to disk
        print("Saving embeddings and labels to disk...")
        try:
            np.save(embeddings_v1_path, embeddings_v1)
            np.save(embeddings_v2_path, embeddings_v2)
            np.save(labels_path, labels)
            print("Embeddings and labels saved successfully.")
        except Exception as e:
            print(f"Error saving embeddings and labels: {e}")
            exit(1)

    # ============================
    # 8. Principal Component Analysis (PCA)
    # ============================
    print("Performing PCA on DINOv1 embeddings...")
    pca_v1, transformed_v1, var_explained_v1 = perform_pca(embeddings_v1)
    print(f"PCA completed for DINOv1.")

    print("Performing PCA on DINOv2 embeddings...")
    pca_v2, transformed_v2, var_explained_v2 = perform_pca(embeddings_v2)
    print(f"PCA completed for DINOv2.")

    # ============================
    # 9. Visualization of Variance Explained
    # ============================
    # Define timestamp and output directory
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
    output_dir = Path.cwd() / "output" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory set to: {output_dir}")

    # Define unique filenames and save paths
    filename_v1_variance = "dino_v1_variance.png"
    filename_v2_variance = "dino_v2_variance.png"
    save_path_v1_variance = output_dir / filename_v1_variance
    save_path_v2_variance = output_dir / filename_v2_variance

    # Plot cumulative variance for DINOv1
    plot_variance(
        var_explained_v1,
        "DINOv1: Cumulative Variance Explained",
        save_path=save_path_v1_variance,
    )

    # Plot cumulative variance for DINOv2
    plot_variance(
        var_explained_v2,
        "DINOv2: Cumulative Variance Explained",
        save_path=save_path_v2_variance,
    )

    # ============================
    # 10. Determining Number of Components for 95% Variance
    # ============================
    threshold = 0.95
    components_v1 = components_for_variance(var_explained_v1, threshold)
    components_v2 = components_for_variance(var_explained_v2, threshold)
    print(
        f"DINOv1: {components_v1} components explain at least {int(threshold*100)}% variance."
    )
    print(
        f"DINOv2: {components_v2} components explain at least {int(threshold*100)}% variance."
    )

    # ============================
    # 11. Silhouette Score Calculation
    # ============================
    print("Computing mean Silhouette Scores with 50 PCA components...")
    try:
        score_v1 = compute_silhouette(embeddings_v1, labels, n_components=50)
        score_v2 = compute_silhouette(embeddings_v2, labels, n_components=50)
        print(f"DINOv1 Mean Silhouette Score (50 PCA components): {score_v1:.4f}")
        print(f"DINOv2 Mean Silhouette Score (50 PCA components): {score_v2:.4f}")
    except Exception as e:
        print(f"Error computing Silhouette Scores: {e}")
        exit(1)

    print("Computing per-sample Silhouette Scores with 50 PCA components...")
    try:
        sample_scores_v1 = compute_silhouette_samples_custom(
            embeddings_v1, labels, n_components=50
        )
        sample_scores_v2 = compute_silhouette_samples_custom(
            embeddings_v2, labels, n_components=50
        )
    except Exception as e:
        print(f"Error computing per-sample Silhouette Scores: {e}")
        exit(1)

    # ============================
    # 12. Paired t-Test on Silhouette Scores
    # ============================
    print("Performing paired t-test on per-sample Silhouette Scores...")

    if len(sample_scores_v1) != len(sample_scores_v2):
        print("Mismatch in number of samples between models.")
        exit(1)

    try:
        t_stat, p_value = ttest_rel(sample_scores_v1, sample_scores_v2)
    except Exception as e:
        print(f"Error performing paired t-test: {e}")
        exit(1)

    print(f"Paired t-test results:")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")

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
    # Define unique filenames and save paths
    filename_v1_pca2d = "dino_v1_pca2d.png"
    filename_v2_pca2d = "dino_v2_pca2d.png"
    save_path_v1_pca2d = output_dir / filename_v1_pca2d
    save_path_v2_pca2d = output_dir / filename_v2_pca2d

    # Create a mapping from label indices to class names
    label_to_name = {
        label: name for label, name in zip(selected_class_indices, selected_classes)
    }

    # Perform and plot 2D PCA for DINOv1
    print("Performing 2D PCA on DINOv1 embeddings...")
    try:
        pca_2d_v1 = PCA(n_components=2, random_state=42)
        embeddings_v1_2d = pca_2d_v1.fit_transform(embeddings_v1)
        score_v1_2d = silhouette_score(embeddings_v1_2d, labels)
        print(f"DINOv1 Silhouette Score (2D): {score_v1_2d:.4f}")
        plot_pca_2d(
            embeddings_v1_2d,
            labels,
            label_to_name,  # Pass the label_to_name mapping here
            "DINOv1 Embeddings PCA 2D",
            save_path=save_path_v1_pca2d,
        )
    except Exception as e:
        print(f"Error during DINOv1 2D PCA plotting: {e}")
        exit(1)

    # Perform and plot 2D PCA for DINOv2
    print("Performing 2D PCA on DINOv2 embeddings...")
    try:
        pca_2d_v2 = PCA(n_components=2, random_state=42)
        embeddings_v2_2d = pca_2d_v2.fit_transform(embeddings_v2)
        score_v2_2d = silhouette_score(embeddings_v2_2d, labels)
        print(f"DINOv2 Silhouette Score (2D): {score_v2_2d:.4f}")
        plot_pca_2d(
            embeddings_v2_2d,
            labels,
            label_to_name,  # Pass the label_to_name mapping here
            "DINOv2 Embeddings PCA 2D",
            save_path=save_path_v2_pca2d,
        )
    except Exception as e:
        print(f"Error during DINOv2 2D PCA plotting: {e}")
        exit(1)


if __name__ == "__main__":
    main()
