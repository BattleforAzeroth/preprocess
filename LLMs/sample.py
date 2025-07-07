from utils import file_io
import pandas as pd
import re
import random


def remove_comments(code):
    # 移除/* */类型的多行注释
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    # 移除//类型的单行注释
    code = re.sub(r'//.*', '', code)
    return code


def process_code_file(code):
    # 移除注释
    clean_code = remove_comments(code)
    # 分割成行并过滤掉空行（包括仅含空格的行）
    lines = [line for line in clean_code.split('\n') if line.strip()]
    return lines


def extract_xy(lines):
    if not lines:
        return None, None

    if len(lines) <= 10:
        x = lines[:-1]
        y = lines[-1]
    else:
        # 随机选择起始位置，确保能取到连续的11行
        start = random.randint(0, len(lines) - 11)
        selected_lines = lines[start:start + 11]
        x = selected_lines[:10]
        y = selected_lines[-1]

    # 将x中的行合并为一个字符串，用换行符连接
    x_str = '\n'.join(x)
    return x_str, y


def process_dataframe(input_df, code_column='code'):
    results = []

    for _, row in input_df.iterrows():
        code = row[code_column]
        lines = process_code_file(code)
        x, y = extract_xy(lines)

        if x is not None and y is not None:
            results.append({'x': x, 'y': y})

    output_df = pd.DataFrame(results)
    return output_df


if __name__ == "__main__":
    input_file = r'D:\source\Dataset\filtered.csv'
    df = file_io.read_csv(input_file)

    output_df = process_dataframe(df, code_column='text')

    # 保存到CSV文件
    output_df.to_csv('code_pairs.csv', index=False)
    print("处理完成，结果已保存到 code_pairs.csv")

