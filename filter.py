import torch
import json
from typing import Dict, Set
import os
import time


def load_similarity_matrix(file_path: str = './out/similarity_matrix.pt', device: str = 'cuda') -> torch.Tensor:
    """加载相似度矩阵到指定设备"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    print(f"Loading similarity matrix from {file_path}...")
    start_time = time.time()
    matrix = torch.load(file_path, map_location=device)
    print(f"Loaded in {time.time() - start_time:.2f}s | Shape: {matrix.shape} | Device: {matrix.device}")
    return matrix


def load_tokenize_results(file_path: str = './out/all_tokenize_res.txt') -> Dict:
    """加载tokenize结果文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    print(f"Loading tokenize results from {file_path}...")
    with open(file_path, 'r') as f:
        return json.load(f)


def find_high_similar_pairs_gpu(
        sim_matrix: torch.Tensor,
        threshold: float = 0.8,
        batch_size: int = 10000,
        keep_lower: bool = True
) -> Set[int]:
    """
    GPU加速查找高相似度对并返回需要移除的索引

    参数:
        sim_matrix: 相似度矩阵 (GPU Tensor)
        threshold: 相似度阈值
        batch_size: 每批处理的行数
        keep_lower: 是否保留索引较小的项目

    返回:
        需要移除的索引集合
    """
    print(f"\nFinding pairs with similarity > {threshold} (batch_size={batch_size})...")
    n = sim_matrix.shape[0]
    to_remove = set()

    # 预计算总对数
    total_pairs = n * (n - 1) // 2
    print(f"Total possible pairs: {total_pairs:,}")

    start_time = time.time()
    processed_pairs = 0

    for i in range(0, n, batch_size):
        batch_start = i
        batch_end = min(i + batch_size, n)

        # 获取当前批次的行块
        batch = sim_matrix[batch_start:batch_end]

        # 创建上三角索引 (直接在GPU上创建)
        rows, cols = torch.triu_indices(
            batch.shape[0],  # 当前批次的行数
            n,  # 总列数
            offset=1,  # 从对角线+1开始
            device=sim_matrix.device  # 确保在相同设备上
        )
        rows += batch_start  # 转换为全局索引

        # 提取上三角元素
        triu_values = batch[rows - batch_start, cols]

        # 找出大于阈值的相似对
        mask = triu_values > threshold
        high_sim_rows = rows[mask]
        high_sim_cols = cols[mask]

        # 确定要保留和移除的项目
        if keep_lower:
            # 保留较小的索引（移除较大的）
            to_remove.update(high_sim_cols.cpu().tolist())
        else:
            # 保留较大的索引（移除较小的）
            to_remove.update(high_sim_rows.cpu().tolist())

        # 更新进度
        processed_pairs += len(triu_values)
        elapsed = time.time() - start_time
        pairs_per_sec = processed_pairs / elapsed if elapsed > 0 else 0
        remaining_pairs = total_pairs - processed_pairs
        eta = remaining_pairs / pairs_per_sec if pairs_per_sec > 0 else 0

        print(f"\rProgress: {processed_pairs:,}/{total_pairs:,} pairs "
              f"({processed_pairs / total_pairs * 100:.1f}%) | "
              f"Speed: {pairs_per_sec / 1e6:.2f}M pairs/s | "
              f"ETA: {eta:.1f}s", end='', flush=True)

    print(f"\nFound {len(to_remove):,} items to remove")
    print(f"Total processing time: {time.time() - start_time:.2f}s")
    return to_remove


def filter_and_save_results(
        token_dict: Dict,
        to_remove: Set[int],
        output_path: str = './out/filtered_tokenize_res.txt'
):
    """过滤并保存结果"""
    print("\nFiltering tokenize results...")

    # 获取有序的key列表（假设与矩阵索引一致）
    keys = list(token_dict.keys())

    # 构建过滤后的字典
    filtered_dict = {
        k: v for i, (k, v) in enumerate(token_dict.items())
        if i not in to_remove and i < len(keys)
    }

    # 保存结果
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(filtered_dict, file)
    print(f"Original items: {len(token_dict):,}")
    print(f"Removed items: {len(to_remove):,}")
    print(f"Filtered items: {len(filtered_dict):,}")
    print(f"Saved to {output_path}")


def main():
    # 配置参数
    threshold = 0.8
    batch_size = 1000  # 根据GPU内存调整
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. 加载数据
    sim_matrix = load_similarity_matrix(device=device)
    token_dict = load_tokenize_results()

    # 2. GPU加速查找高相似度对
    to_remove = find_high_similar_pairs_gpu(
        sim_matrix,
        threshold=threshold,
        batch_size=batch_size,
        keep_lower=True  # 保留索引较小的项目
    )

    # 3. 过滤并保存结果
    filter_and_save_results(token_dict, to_remove)


if __name__ == "__main__":
    main()