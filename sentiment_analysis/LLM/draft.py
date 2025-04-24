import pandas as pd

df = pd.read_csv('../dataset/Dataset_finance_chinese/test_data.csv')
df1 = pd.DataFrame()

#sentence为标题➕正文⬇️
# df1['sentence'] = df['标题'] + ':' + df['正文']
#sentence仅为标题⬇️
df1['sentence'] = df['标题']
df1['label_num']= df['正负面']
df1.to_csv('./test.csv')