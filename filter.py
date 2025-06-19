from utils import csv


input_file = r'D:\source\Dataset\distinct.csv'
output_file = r'D:\source\Dataset\filtered.csv'
df = csv.read_csv(input_file)

# 计算每行的字符长度，并过滤掉超过 20,000 的行
max_length = 20000
filtered_df1 = df[df["text"].str.len() <= max_length]

# 过滤掉不包含关键字的行
keywords = ["always", "assign", "always_ff", "always_comb", "always_latch"]
mask = filtered_df1["text"].str.contains('|'.join(keywords), case=False, regex=True)

filtered_df2 = filtered_df1[mask]
csv.write_csv(filtered_df2, output_file)