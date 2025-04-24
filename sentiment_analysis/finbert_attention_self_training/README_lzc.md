# FinBERT: Financial Sentiment Analysis with BERT

### 程序逻辑
提供`input_sentences.csv`文件，位于`finbert_chinese`目录下，仅有列名为'sentence'，运行`predict.py`执行预测任务，输出sentence，label，score

---

#### input：

sentence
此外宁德时代上半年实现出口约2GWh，同比增加200%+。
今日股市波动剧烈，投资者信心不足。
#### output：生成finbert_ch_pre.csv
sentence,prediction,sentiment_score（softmax之后的预测概率/置信度）
此外宁德时代上半年实现出口约2GWh，同比增加200%+。,Positive,0.9989
今日股市波动剧烈，投资者信心不足。,Negative,0.9988

---

### 阶段性结论
对中文文本的预测效果不好（其实也是好事，这样就有提升）
> `sentence`:明天包涨的
> `label`：neutral

LLM可作为创新点，效果还不错



### 后续作为特征输入--`情绪因子`=（positive/ neutral / negative）对应（-1，0，1）*置信度


---

## 程序说明
`data_preparation` 数据预处理，处理。txt格式 （csv需修改）

`predict` 主程序，直接运行即可

`mswa` 多尺度窗口注意力机制 （输入输出维度相同）

# 模型训练
## 程序说明
### `train` 📃
运行时需查看路径，以及`train_dataset`的格式
>需包含两列 sentence 和 label  ，列名对应

### `model_save` 📁
`logs`日志记录
>logging_steps=10,
这意味着每隔 10 步（logging_steps=10）就会记录一次日志，保存到 logs 文件

`checkpoint-**`保存的训练快照，可从快照中恢复训练参数
`。bin`训练完成后生成的最优参数存档