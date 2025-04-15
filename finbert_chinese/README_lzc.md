# FinBERT: Financial Sentiment Analysis with BERT

### 1、输入：提供input_sentences.csv文件，位于finbert_chinese目录下，列名为'sentence'
例如：

sentence
此外宁德时代上半年实现出口约2GWh，同比增加200%+。
今日股市波动剧烈，投资者信心不足。


### 2、运行predict.py


### 3、输出：生成finbert_ch_pre.csv
内容如下：
sentence,prediction,sentiment_score（softmax之后的预测概率/置信度）
此外宁德时代上半年实现出口约2GWh，同比增加200%+。,Positive,0.9989
今日股市波动剧烈，投资者信心不足。,Negative,0.9988

### 后续作为特征输入可以是（positive/ neutral / negative）对应（-1，0，1）*置信度