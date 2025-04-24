import pandas as pd


#数据预处理，df1共包含两列df1['sentence'],df1['label]
def dfc_load(file_path):
    df = pd.read_csv(file_path)
    df1 = pd.DataFrame()

    #sentence为标题➕正文⬇️
    # df1['sentence'] = df['标题'] + ':' + df['正文']
    #sentence仅为标题⬇️
    df1['sentence'] = df['标题']
    df1['label'] = df['正负面']

    # header=None：表示原始文件没有表头行（第一行就是数据，不是列名）
    # df = pd.read_csv(file_path, sep='@', names=["sentence", "label"],encoding='ISO-8859-1')
    # df["sentence"] = df["sentence"].str.strip()  # 去除前后空格
    # df["label"]=df["label"].str.strip()
    # sentence_label = df1["sentence"].tolist()
    # label=[]
    # for i in df1['label_num']:
    #     label.append('Negative') if i==0 else label.append('Positive')
    # df1['label']=pd.DataFrame(label)
    #
    df1.to_csv("./train_dataset.csv",index=False)
    return print("文件保存在./train_dataset.csv")

dfc_load("/Users/lv/PycharmProjects/sentiment_analysis/dataset/Dataset_finance_chinese/train_data.csv")