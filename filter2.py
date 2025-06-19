import pandas as pd

from utils import csv
import xiangsi as xs
import time


input_file = r'D:\source\Dataset\distinct.csv'
output_file = r'D:\source\Dataset\filtered2.csv'
df = csv.read_csv(input_file)

# 计算每行的字符长度，并过滤掉超过 20,000 的行
# max_length = 20000
# filtered_df1 = df[df["text"].str.len() <= max_length]

# 过滤掉不包含关键字的行
# keywords = ["always", "assign", "always_ff", "always_comb", "always_latch"]
# mask = filtered_df1["text"].str.contains('|'.join(keywords), case=False, regex=True)
token_seqs = []
for no in range(len(df)):
    code = df["text"].iloc[no]

    flag = True
    for seq in token_seqs:
        # 相似度度量
        # 开始计时
        # start_time = time.time()

        if xs.minhash(seq, code) > 0.75:
            flag = False

        # 结束计时
        # end_time = time.time()
        # 计算并打印执行时间
        # print(f"执行时间：{end_time - start_time}秒")

    if flag:
        token_seqs.append(code)


filtered_df = pd.DataFrame(token_seqs, columns=["text"])
csv.write_csv(filtered_df, output_file)
