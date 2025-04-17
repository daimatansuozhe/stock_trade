import pandas as pd


def load_sentences(file_path):
    # header=None：表示原始文件没有表头行（第一行就是数据，不是列名）
    df = pd.read_csv(file_path, sep='@', names=["sentence", "label"],encoding='ISO-8859-1')
    df["sentence"] = df["sentence"].str.strip()  # 去除前后空格
    df["label"]=df["label"].str.strip()
    sentence_label = df["sentence"].tolist()
    return sentence_label,df["label"]