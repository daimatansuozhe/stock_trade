# FinBERT: Financial Sentiment Analysis with BERT

### 提供`input_sentences.csv`文件，位于`finbert_chinese`目录下，仅有列名为'sentence'，运行`predict.py`执行预测任务，输出sentence，label，score
#### input：

sentence
此外宁德时代上半年实现出口约2GWh，同比增加200%+。
今日股市波动剧烈，投资者信心不足。
#### output：生成finbert_ch_pre.csv
sentence,prediction,sentiment_score（softmax之后的预测概率/置信度）
此外宁德时代上半年实现出口约2GWh，同比增加200%+。,Positive,0.9989
今日股市波动剧烈，投资者信心不足。,Negative,0.9988

### 阶段性结论
对中文文本的预测效果不好（其实也是好事，这样就有提升）
> `sentence`:明天包涨的
> `label`：neutral

后续采用大模型gpt预测看看效果



### 后续作为特征输入--`情绪因子`=（positive/ neutral / negative）对应（-1，0，1）*置信度