import pandas as pd
import csv

def read_csv(file):
    csv.field_size_limit(500 * 1024 * 1024)
    df = pd.read_csv(
        file,
        header=0,
        quoting=csv.QUOTE_ALL,
        engine="python"
    )
    return df


def write_csv(df, file):
    df.to_csv(
        file,
        index=False,
        header=True,
        encoding='utf-8',
        quoting=csv.QUOTE_ALL
    )