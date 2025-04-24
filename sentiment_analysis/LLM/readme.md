# deepseek-v3/r1 情感分析

## choose model 
`def sentiment_analysis(model="deepseek-chat")`

## 参数修改
设置 client:调用的api_key和api——url,custom:用户提示词,sys:系统提示词
>client = openai.OpenAI(api_key="sk-11da4c508da341088bbf4ec4afc66bf8",base_url='https://api.deepseek.com') 

>input_path='test.csv'

>df=pd.read_csv(input_path)

>custom = (
    "请以简洁的方式分析这段话的情绪倾向（positive、negative）以及情感得分（置信度）保留三位小数：\n{text}\n"
    "请输出格式如下：\nlabel：xxx\nscore：xxx"
)

> sys="你现在是金融领域的情感分析专家"

> temperature=0.3


## 输出 
`res.to_csv("./out/pre.csv",index=False)`保存到 `pre.csv`


report = classification_report(y_label,y_pre,digits=4, target_names=["negative","positive"])

`target_names=["negative","positive"]` 输出最后报告里的列名，但是要求预测值和标签`y_num`,`y_label`在两个类别都有值，不然会报错

`print(report)`输出

              precision    recall  f1-score   support

    negative     1.0000    1.0000    1.0000         3
    positive     1.0000    1.0000    1.0000         1
 
    accuracy                         1.0000         4
    macro avg     1.0000    1.0000    1.0000         4 
    weighted avg     1.0000    1.0000    1.0000         4
