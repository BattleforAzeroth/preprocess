import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple
import time


def load_similarity_matrix(file_path: str = './out/similarity_matrix.pt') -> torch.Tensor:
    """
    Load similarity matrix file

    Args:
        file_path: Path to .pt file

    Returns:
        sim_matrix: Similarity matrix [num_sets, num_sets]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"Loading similarity matrix from {file_path}...")
    start_time = time.time()
    sim_matrix = torch.load(file_path)
    load_time = time.time() - start_time

    print(f"Loading complete! Matrix shape: {sim_matrix.shape}")
    print(f"Loading time: {load_time:.2f} seconds")

    return sim_matrix


def batch_analyze_similarity(sim_matrix: torch.Tensor, bin_size: float = 0.05, batch_size: int = 1000):
    """
    Batch analysis of similarity distribution (upper triangular only)

    Args:
        sim_matrix: Similarity matrix
        bin_size: Histogram bin size
        batch_size: Number of rows per batch
    """
    print("\nStarting batch analysis of similarity distribution (upper triangular only)...")

    num_sets = sim_matrix.shape[0]
    bins = np.arange(0, 1.0 + bin_size, bin_size)
    hist = np.zeros(len(bins) - 1, dtype=np.int64)
    total_pairs = 0

    # Calculate total batches
    num_batches = (num_sets + batch_size - 1) // batch_size
    processed_rows = 0

    start_time = time.time()

    for i in range(0, num_sets, batch_size):
        batch_start = i
        batch_end = min(i + batch_size, num_sets)

        # Get current batch
        batch = sim_matrix[batch_start:batch_end]

        # Process upper triangular portion only
        for row in range(batch.shape[0]):
            global_row = batch_start + row
            # Only take elements after the diagonal to avoid duplicates
            current_sim = batch[row, global_row + 1:num_sets].numpy()

            if len(current_sim) > 0:
                batch_hist, _ = np.histogram(current_sim, bins=bins)
                hist += batch_hist
                total_pairs += len(current_sim)

        processed_rows = batch_end
        elapsed_time = time.time() - start_time
        remaining_batches = num_batches - (i // batch_size + 1)
        eta = (elapsed_time / (i // batch_size + 1)) * remaining_batches if (i // batch_size + 1) > 0 else 0

        print(f"\rProgress: {processed_rows}/{num_sets} rows | "
              f"Elapsed: {elapsed_time:.1f}s | "
              f"ETA: {eta:.1f}s", end='', flush=True)

    print("\nAnalysis complete!")

    # Calculate statistics
    if total_pairs > 0:
        # Calculate weighted mean and std
        bin_centers = bins[:-1] + bin_size / 2
        sum_sim = np.sum(hist * bin_centers)
        mean_sim = sum_sim / total_pairs
        sum_sq_diff = np.sum(hist * (bin_centers - mean_sim) ** 2)
        std_sim = np.sqrt(sum_sq_diff / total_pairs)
    else:
        mean_sim = 0
        std_sim = 0

    # Print statistics
    print(f"\nStatistics:")
    print(f"Total unique pairs: {total_pairs:,}")
    print(f"Mean similarity: {mean_sim:.4f}")
    print(f"Standard deviation: {std_sim:.4f}")

    # Calculate probability density
    prob_density = hist / total_pairs if total_pairs > 0 else np.zeros_like(hist)

    # Plot histogram
    plt.figure(figsize=(14, 7))
    bars = plt.bar(bins[:-1], prob_density, width=bin_size * 0.9,
                   align='edge', alpha=0.7, edgecolor='black', linewidth=0.7)

    # Add statistics lines
    plt.axvline(mean_sim, color='red', linestyle='--', linewidth=1.5,
                label=f'Mean = {mean_sim:.3f}')

    # Configure plot
    plt.title(f'MinHash Similarity Distribution (Bin Size={bin_size}, Batch Size={batch_size})',
              fontsize=14, pad=20)
    plt.xlabel('Similarity', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12)
    plt.xticks(np.arange(0, 1.1, 0.1))

    # Add text labels for ALL bars (not just non-zero)
    max_density = prob_density.max() if len(prob_density) > 0 else 0
    label_offset = max_density * 0.02  # Dynamic offset based on max density

    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + label_offset,
                 f'{prob_density[i]:.3f}',
                 ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()

    # Print distribution table
    print("\nSimilarity Distribution Table:")
    print("Range\t\tCount\t\tPercentage")
    for i in range(len(hist)):
        lower = bins[i]
        upper = bins[i + 1]
        print(f"[{lower:.2f}, {upper:.2f})\t{hist[i]:,}\t\t{hist[i] / total_pairs * 100:.2f}%")


def main():
    # Load similarity matrix
    try:
        sim_matrix = load_similarity_matrix()
    except FileNotFoundError as e:
        print(e)
        return

    batch_analyze_similarity(sim_matrix, bin_size=0.05, batch_size=1000)


if __name__ == "__main__":
    main()