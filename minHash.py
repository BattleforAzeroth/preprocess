import json
import torch
import numpy as np
from typing import List, Set
import time
import random
from dataclasses import dataclass

from utils import file_io


@dataclass
class MinHashConfig:
    num_perm: int = 128  # 哈希函数的数量
    seed: int = 42  # 随机种子
    max_element: int = 100000  # 元素的最大值


class MinHashGenerator:
    def __init__(self, config: MinHashConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)

        # 生成哈希函数的参数 (a, b) 和大质数
        self.hash_params, self.prime = self._generate_hash_params()

    def _generate_hash_params(self):
        """生成哈希函数族: h(x) = (a * x + b) % prime"""
        prime = (1 << 61) - 1  # 一个大质数

        a = self.rng.randint(1, prime, size=(self.config.num_perm, 1), dtype=np.uint64)
        b = self.rng.randint(0, prime, size=(self.config.num_perm, 1), dtype=np.uint64)

        return torch.from_numpy(np.hstack([a, b])).to(torch.long), prime

    def generate_random_sets(self, num_sets: int, max_size: int = 300) -> List[Set[int]]:
        """生成随机集合"""
        sets = []
        for _ in range(num_sets):
            size = random.randint(1, max_size)
            elements = set(random.randint(1, self.config.max_element) for _ in range(size))
            sets.append(elements)
        return sets

    def minhash(self, sets: List[Set[int]], device='cuda', batch_size=1000):
        """
        计算所有集合的MinHash签名（分批处理)

        参数:
            sets: 集合列表
            device: 计算设备 ('cuda' 或 'cpu')
            batch_size: 批处理大小

        返回:
            signatures: 形状为 [num_sets, num_perm] 的签名矩阵
        """
        if not sets:
            return torch.zeros((0, self.config.num_perm), dtype=torch.long)

        # 获取所有唯一元素并建立元素到索引的映射
        print("收集所有唯一元素...")
        all_elements = set()
        for s in sets:
            all_elements.update(s)
        element_list = sorted(all_elements)
        element_to_idx = {e: i for i, e in enumerate(element_list)}
        num_elements = len(element_list)

        # 将哈希参数移到设备上
        hash_params = self.hash_params.to(device)

        # 预分配签名矩阵
        signatures = torch.full((len(sets), self.config.num_perm),
                                torch.iinfo(torch.long).max,
                                device='cpu')

        # 分批处理
        print(f"开始分批处理，共{len(sets)}个集合，每批{batch_size}个...")
        for batch_start in range(0, len(sets), batch_size):
            batch_end = min(batch_start + batch_size, len(sets))
            batch_sets = sets[batch_start:batch_end]

            print(f"处理批次 {batch_start // batch_size + 1}/{(len(sets) - 1) // batch_size + 1}...", end='\r')

            # 准备当前批次的稀疏矩阵
            row_indices = []
            col_indices = []

            for local_idx, s in enumerate(batch_sets):
                global_idx = batch_start + local_idx
                for e in s:
                    row_indices.append(local_idx)
                    col_indices.append(element_to_idx[e])

            if not row_indices:
                batch_signatures = torch.full((len(batch_sets), self.config.num_perm),
                                              torch.iinfo(torch.long).max,
                                              device=device)
            else:
                # 创建稀疏矩阵 (使用COO格式)并合并
                indices = torch.tensor([row_indices, col_indices], dtype=torch.long)
                values = torch.ones(len(row_indices), dtype=torch.uint8)

                set_matrix = torch.sparse_coo_tensor(
                    indices, values, (len(batch_sets), num_elements),
                    dtype=torch.uint8, device=device
                ).coalesce()

                # 计算所有元素的哈希值
                element_indices = torch.arange(num_elements, device=device)

                # 预分配当前批次的签名矩阵
                batch_signatures = torch.full((len(batch_sets), self.config.num_perm),
                                              torch.iinfo(torch.long).max,
                                              device=device)

                # 对每个哈希函数进行处理
                for i in range(self.config.num_perm):
                    a, b = hash_params[i, 0], hash_params[i, 1]
                    current_hashes = (a * element_indices + b) % self.prime

                    # 获取稀疏矩阵的非零元素
                    rows = set_matrix.indices()[0]
                    cols = set_matrix.indices()[1]

                    if len(rows) > 0:
                        data = current_hashes[cols]
                        min_hashes = torch.zeros(len(batch_sets), device=device, dtype=torch.long) + torch.iinfo(
                            torch.long).max
                        min_hashes.scatter_reduce_(0, rows, data, reduce='amin')
                        batch_signatures[:, i] = min_hashes

            # 将当前批次的签名存回主矩阵
            signatures[batch_start:batch_end] = batch_signatures.cpu()

        return signatures

    def similarity(self, signatures: torch.Tensor, device='cuda', batch_size=1000):
        """
        分批计算所有签名对之间的相似度

        参数:
            signatures: MinHash签名矩阵 [num_sets, num_perm]
            device: 计算设备
            batch_size: 批处理大小

        返回:
            sim_matrix: 相似度矩阵 [num_sets, num_sets]
        """
        if len(signatures) == 0:
            return torch.zeros((0, 0))

        num_sets = len(signatures)
        sim_matrix = torch.zeros((num_sets, num_sets), dtype=torch.float32)

        # 分批处理行
        for i_start in range(0, num_sets, batch_size):
            i_end = min(i_start + batch_size, num_sets)
            batch_i = signatures[i_start:i_end].to(device)

            # 分批处理列
            for j_start in range(0, num_sets, batch_size):
                j_end = min(j_start + batch_size, num_sets)
                batch_j = signatures[j_start:j_end].to(device)

                # 计算当前批次之间的相似度
                equal = (batch_i.unsqueeze(1) == batch_j.unsqueeze(0)).sum(dim=2)
                batch_sim = equal.float() / self.config.num_perm

                # 存回主矩阵
                sim_matrix[i_start:i_end, j_start:j_end] = batch_sim.cpu()

                print(f"处理相似度矩阵块 ({i_start}-{i_end})×({j_start}-{j_end})", end='\r')

        print("\n相似度矩阵计算完成!")
        return sim_matrix


def main(sets):
    # 配置
    output_matrix = "./out/similarity_matrix.pt"
    config = MinHashConfig(
        num_perm=128,
        seed=42,
        max_element=100000
    )
    minhash = MinHashGenerator(config)

    # # 生成50,000个随机集合
    # print("生成50,000个随机集合...")
    # start_time = time.time()
    # sets = minhash.generate_random_sets(num_sets=50000, max_size=300)
    # print(f"生成集合耗时: {time.time() - start_time:.2f}秒")
    # print(f"示例集合大小: {len(sets[0])}, {len(sets[1])}, {len(sets[2])}")

    # 计算MinHash签名
    print("\n计算MinHash签名...")
    start_time = time.time()
    signatures = minhash.minhash(sets, device='cuda', batch_size=1000)
    print(f"\nMinHash签名计算总耗时: {time.time() - start_time:.2f}秒")
    print(f"签名矩阵形状: {signatures.shape}")

    # 计算相似度矩阵
    print("\n计算相似度矩阵...")
    start_time = time.time()
    sim_matrix = minhash.similarity(signatures, device='cuda', batch_size=1000)
    print(f"相似度矩阵计算总耗时: {time.time() - start_time:.2f}秒")
    print(f"相似度矩阵形状: {sim_matrix.shape}")

    # 保存结果
    print("\n保存结果...")
    torch.save(sim_matrix, output_matrix)
    print(f"相似度矩阵结果已保存到{output_matrix}")


if __name__ == "__main__":
    file = "./out/all_tokenize_res.txt"
    dataset_dict = file_io.read_dict(file)
    token_sets = []
    for k, v in dataset_dict.items():
        token_sets.append(set(v[0]))
    main(token_sets)
