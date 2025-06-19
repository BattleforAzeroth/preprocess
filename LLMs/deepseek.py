from openai import OpenAI
import pandas as pd
client = OpenAI(
    base_url="https://api.deepseek.com/",
    api_key="sk-1a39859dc5a44c618418658a462a898c"
)

def call_api(code_snippet):
    try:
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                        "role": "system",
                        "content": "你是一个Verilog代码专家，现在给你若干行Verilog代码，请输出下一行你认为最有可能出现的代码，注意只能输出一行代码，只返回这一行代码的结果，结果不要带换行符"
                },
                {
                        "role": "user",
                        "content": code_snippet
                }
            ],
            temperature=0.0,
            stream=False
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"API调用失败: {e}")
        return None


# 主处理函数
def process_csv_with_api(input_csv, output_csv, output_txt="api_outputs.txt"):
    # 读取CSV文件
    df = pd.read_csv(input_csv)

    if 'x' not in df.columns:
        raise ValueError("CSV文件中缺少x列")

    df['pred'] = None

    with open(output_txt, 'w', encoding='utf-8') as txt_file:
        for index, row in df.iterrows():
            code_snippet = row['x']
            pred = call_api(code_snippet)
            df.at[index, 'pred'] = pred

            # 只将API输出写入txt文件
            txt_file.write(f"{pred}\n")

            if index % 10 == 0:
                print(f"已处理 {index + 1}/{len(df)} 行")

    df.to_csv(output_csv, index=False)
    print(f"处理完成，结果已保存到 {output_csv}")
    print(f"API输出已保存到 {output_txt}")


if __name__ == "__main__":
    input_file = "code_pairs.csv"
    output_file = "code_pairs_with_pred.csv"
    output_txt = "api_outputs.txt"

    process_csv_with_api(input_file, output_file)
