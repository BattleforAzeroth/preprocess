import json
import time
import random
from itertools import combinations
from cuml.feature_extraction.text import MinHashEncoder
import cupy as cp
import numpy as np


def generate_test_sets(n_sets=100, min_size=50, max_size=400):
    """生成测试集合"""
    max_item = 1000  # 假设有1000种不同的可能项目
    return [set(random.sample(range(max_item), random.randint(min_size, max_size)))
            for _ in range(n_sets)]


def set_to_string(set_data):
    """将集合转换为字符串表示"""
    return ' '.join(map(str, set_data))


def gpu_minhash_similarity(list_of_sets, num_perm=128):
    """
    GPU加速的MinHash相似度计算

    参数:
        list_of_sets: 集合列表，每个元素是一个set
        num_perm: MinHash的排列数，影响精度

    返回:
        (结果列表, 各阶段耗时字典)
        结果列表包含所有两两相似度的列表，每个元素是元组(i, j, similarity)
        耗时字典包含各阶段的执行时间
    """
    timings = {}
    start_total = time.time()

    # 1. 数据预处理
    start_time = time.time()
    set_strings = [set_to_string(s) for s in list_of_sets]
    timings['data_preparation'] = time.time() - start_time

    # 2. GPU MinHash编码
    start_time = time.time()
    encoder = MinHashEncoder(n_components=num_perm, random_state=42)
    gpu_hashes = encoder.fit_transform(set_strings)
    timings['gpu_minhashing'] = time.time() - start_time

    # 3. GPU相似度计算
    start_time = time.time()
    # 计算余弦相似度矩阵
    norm = cp.linalg.norm(gpu_hashes, axis=1)
    norm_matrix = cp.outer(norm, norm)
    dot_product = gpu_hashes @ gpu_hashes.T
    similarity_matrix = cp.divide(dot_product, norm_matrix,
                                  out=cp.zeros_like(dot_product),
                                  where=norm_matrix != 0)
    timings['gpu_similarity'] = time.time() - start_time

    # 4. 结果转换
    start_time = time.time()
    n = len(list_of_sets)
    results = []
    for i, j in combinations(range(n), 2):
        results.append((i, j, float(similarity_matrix[i, j])))
    timings['result_conversion'] = time.time() - start_time

    timings['total'] = time.time() - start_total

    return results, timings


def validate_results(results, list_of_sets, sample_size=5):
    """验证GPU计算结果与精确Jaccard相似度的差异"""
    sample_pairs = random.sample(results, min(sample_size, len(results)))

    print("\n验证结果 (GPU vs 精确Jaccard):")
    print("索引对\tGPU相似度\t精确相似度\t差值")

    for i, j, gpu_sim in sample_pairs:
        set1 = list_of_sets[i]
        set2 = list_of_sets[j]

        # 计算精确Jaccard相似度
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        exact_sim = intersection / union if union != 0 else 0

        print(f"{i}-{j}\t{gpu_sim:.4f}\t\t{exact_sim:.4f}\t\t{abs(gpu_sim - exact_sim):.4f}")


if __name__ == "__main__":
    # 生成测试数据
    n_sets = 100
    print(f"生成 {n_sets} 个随机集合...")
    list_of_sets = generate_test_sets(n_sets=n_sets)

    # GPU加速计算
    print("\n开始GPU加速的MinHash相似度计算...")
    results, timings = gpu_minhash_similarity(list_of_sets)

    # 打印耗时信息
    print(f"\n总计算完成时间: {timings['total']:.4f}秒")
    print("\n各阶段耗时分析:")
    for stage, t in timings.items():
        print(f"{stage:20}: {t:.4f}秒 ({(t / timings['total']) * 100:.1f}%)")

    # 打印基本信息
    print(f"\n集合数量: {n_sets}")
    print(f"两两组合数: {len(results)}")

    # 打印部分结果示例
    print("\n示例相似度结果(前10对):")
    for i, j, sim in results[:10]:
        print(f"集合 {i:3} 和 集合 {j:3} 的相似度: {sim:.4f}")

    # 结果验证
    validate_results(results, list_of_sets)

    # with open("all_tokenize_res.txt", "r", encoding="utf-8") as file:
    #     dataset_dict = json.load(file)
    #
    # n = len(dataset_dict)
    # print("before: {}".format(n))
    # token_sets = []
    # for item in dataset_dict:
    #     token_sets.append(set(item[0]))
    # dataset_dict.append(token_sets)
    #
    # for i in range(n - 1):
    #     for j in range(i + 1, n):




