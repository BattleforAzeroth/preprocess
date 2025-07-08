from utils import file_io
import random


def shuffle_and_split_dict(data_dict, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    打乱字典并按比例划分为训练集、验证集和测试集

    参数:
        data_dict: 要划分的字典
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例

    返回:
        train_dict, val_dict, test_dict: 划分后的三个字典
    """
    # 检查比例总和是否为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例总和必须为1"

    # 将字典的键转换为列表并打乱
    keys = list(data_dict.keys())
    random.shuffle(keys)

    # 计算各集合的大小
    total_size = len(keys)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    # 划分键
    train_keys = keys[:train_size]
    val_keys = keys[train_size:train_size + val_size]
    test_keys = keys[train_size + val_size:]

    # 创建对应的字典
    train_dict = {k: data_dict[k] for k in train_keys}
    val_dict = {k: data_dict[k] for k in val_keys}
    test_dict = {k: data_dict[k] for k in test_keys}

    return train_dict, val_dict, test_dict


if __name__ == '__main__':
    input_file = "./out/all_tokenize_res.txt"
    file_io.read_dict(input_file)
    dataset_dict = file_io.read_dict(input_file)
    train, val, test = shuffle_and_split_dict(dataset_dict)
    file_io.write_dict2json(train, "./out/train_tokenize_res.txt")
    file_io.write_dict2json(val, "./out/val_tokenize_res.txt")
    file_io.write_dict2json(test, "./out/test_tokenize_res.txt")