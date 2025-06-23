import json
import numpy as np
from datasketch import MinHash
from itertools import combinations
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import random


def create_minhash(set_data, num_perm=128):
    """将集合转换为MinHash对象"""
    mh = MinHash(num_perm=num_perm)
    for item in set_data:
        mh.update(str(item).encode('utf-8'))
    return mh


def calculate_similarity(pair, minhashes):
    """计算一对MinHash的相似度"""
    i, j = pair
    return (i, j, minhashes[i].jaccard(minhashes[j]))


def parallel_minhash_similarity(list_of_sets, num_perm=128, n_jobs=None):
    """
    并行计算集合列表中所有两两MinHash相似度

    参数:
        list_of_sets: 集合列表，每个元素是一个set
        num_perm: MinHash的排列数，影响精度
        n_jobs: 并行进程数，None表示使用所有CPU核心

    返回:
        (结果列表, 各阶段耗时字典)
        结果列表包含所有两两相似度的列表，每个元素是元组(i, j, similarity)
        耗时字典包含各阶段的执行时间
    """
    timings = {}

    # 1. 将集合转换为MinHash对象
    start_time = time.time()
    minhashes = [create_minhash(s, num_perm) for s in list_of_sets]
    timings['minhash_creation'] = time.time() - start_time

    # 2. 生成所有两两组合的索引
    start_time = time.time()
    n = len(list_of_sets)
    pairs = list(combinations(range(n), 2))
    timings['pair_generation'] = time.time() - start_time

    # 3. 并行计算相似度
    start_time = time.time()
    if n_jobs is None:
        n_jobs = cpu_count()

    with Pool(n_jobs) as pool:
        worker = partial(calculate_similarity, minhashes=minhashes)
        results = pool.map(worker, pairs)
    timings['parallel_computation'] = time.time() - start_time

    # 总时间
    timings['total'] = sum(timings.values())

    return results, timings


def generate_large_test_set(n_sets=100, min_size=50, max_size=400):
    """生成大规模测试集合"""
    max_item = 1000  # 假设有1000种不同的可能项目
    return [set(random.sample(range(max_item), random.randint(min_size, max_size)))
            for _ in range(n_sets)]


# 示例用法
if __name__ == "__main__":
    # 生成100个随机集合的测试数据
    print("正在生成100个随机集合...")
    list_of_sets = generate_large_test_set(n_sets=100)

    print("开始计算MinHash相似度...")
    start_total = time.time()

    # 计算所有两两MinHash相似度
    similarities, timings = parallel_minhash_similarity(list_of_sets)

    total_time = time.time() - start_total
    print(f"\n总计算完成时间: {total_time:.4f}秒")

    # 打印各阶段耗时
    print("\n各阶段耗时分析:")
    for stage, t in timings.items():
        print(f"{stage:20}: {t:.4f}秒 ({t / total_time * 100:.1f}%)")

    # 打印组合数信息
    n = len(list_of_sets)
    print(f"\n集合数量: {n}")
    print(f"两两组合数: {n * (n - 1) // 2}")

    # 打印部分结果示例
    print("\n示例相似度结果(前10对):")
    for i, j, sim in similarities[:10]:
        print(f"集合 {i:3} 和 集合 {j:3} 的相似度: {sim:.3f}")

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




