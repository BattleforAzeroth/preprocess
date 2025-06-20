import json
import hashlib
from itertools import combinations
from multiprocessing import Pool
from tqdm import tqdm  # 进度条（可选，安装：pip install tqdm）

class ParallelMinHash:
    def __init__(self, num_hashes=128, n_jobs=-1):
        """
        Args:
            num_hashes (int): MinHash 哈希函数数量
            n_jobs (int): 并行进程数（-1 表示使用所有 CPU）
        """
        self.num_hashes = num_hashes
        self.n_jobs = n_jobs
        self.signatures = []  # 存储所有 MinHash 签名

    def compute_signature(self, items):
        """计算单个集合的 MinHash 签名"""
        signature = [float('inf')] * self.num_hashes
        for item in items:
            for i in range(self.num_hashes):
                h = int(hashlib.sha256(f"{i}_{item}".encode()).hexdigest(), 16)
                if h < signature[i]:
                    signature[i] = h
        return signature

    def fit(self, data):
        """预计算所有集合的 MinHash 签名"""
        print("Computing MinHash signatures...")
        with Pool(self.n_jobs) as pool:
            self.signatures = list(tqdm(
                pool.imap(self.compute_signature, data),
                total=len(data)
            ))

    def similarity(self, sig1, sig2):
        """计算两个签名的相似度"""
        similar = sum(a == b for a, b in zip(sig1, sig2))
        return similar / self.num_hashes

    def pairwise_similarities(self):
        """并行计算所有两两相似度"""
        print("Computing pairwise similarities...")
        indices = list(range(len(self.signatures)))
        pairs = list(combinations(indices, 2))  # 所有两两组合

        with Pool(self.n_jobs) as pool:
            results = list(tqdm(
                pool.starmap(
                    lambda i, j: (i, j, self.similarity(self.signatures[i], self.signatures[j])),
                    pairs
                ),
                total=len(pairs)
            ))

        return results

# 示例用法
if __name__ == "__main__":
    # 示例数据（每个集合代表一个用户的兴趣）
    data = [
        {"apple", "banana", "orange"},
        {"apple", "orange", "grape"},
        {"pear", "kiwi", "melon"},
        {"apple", "banana", "grape"}
    ]
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

    # 初始化并行 MinHash
    minhash = ParallelMinHash(num_hashes=128, n_jobs=-1)
    minhash.fit(data)  # 预计算所有签名

    # 计算所有两两相似度
    similarities = minhash.pairwise_similarities()

    # 打印结果
    print("\nPairwise MinHash Similarities:")
    for i, j, sim in similarities:
        print(f"Set {i} & Set {j}: {sim:.3f}")



