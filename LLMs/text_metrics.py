import csv
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk import word_tokenize
import numpy as np
from difflib import SequenceMatcher


def calculate_bleu(reference, hypothesis):
    """
    计算BLEU-4分数
    """
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference], hypothesis, smoothing_function=smoothie)


def calculate_es(reference, hypothesis):
    """
    计算精确匹配(Exact Match)分数
    """
    return 1.0 if reference == hypothesis else 0.0


def calculate_em(reference, hypothesis):
    """
    计算编辑相似度(Edit Similarity)分数
    """
    return SequenceMatcher(None, reference, hypothesis).ratio()


def process_csv_file(file_path):
    """
    处理CSV文件，计算每对文本的BLEU-4、ES和EM分数
    """
    bleu_scores = []
    es_scores = []
    em_scores = []

    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) < 3:
                continue  # 跳过不完整的行

            ref_text = row[1].strip()  # 第二列作为参考文本
            hyp_text = row[2].strip()  # 第三列作为假设文本

            # 分词
            ref_tokens = word_tokenize(ref_text)
            hyp_tokens = word_tokenize(hyp_text)

            # 计算各项指标
            bleu = calculate_bleu(ref_tokens, hyp_tokens)
            es = calculate_es(ref_text, hyp_text)
            em = calculate_em(ref_text, hyp_text)

            bleu_scores.append(bleu*100)
            es_scores.append(es*100)
            em_scores.append(em*100)

    # 计算平均值
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
    avg_es = np.mean(es_scores) if es_scores else 0
    avg_em = np.mean(em_scores) if em_scores else 0

    return {
        'average_bleu': avg_bleu,
        'average_es': avg_es,
        'average_em': avg_em,
        'all_bleu_scores': bleu_scores,
        'all_es_scores': es_scores,
        'all_em_scores': em_scores
    }


# 使用示例
if __name__ == "__main__":
    # 替换为你的CSV文件路径
    csv_file_path = 'code_pairs_with_pred.csv'

    results = process_csv_file(csv_file_path)

    print(f"Average BLEU-4 score: {results['average_bleu']:.4f}")
    print(f"Average Exact Match (ES) score: {results['average_es']:.4f}")
    print(f"Average Edit Similarity (EM) score: {results['average_em']:.4f}")
