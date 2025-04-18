import pandas as pd
import openai
from tqdm import tqdm
from sklearn.metrics import classification_report


def sentiment_analysis(text, sys_prompt="You are a helpful assistant",custom_prompt=None, model="deepseek-chat", temperature=0.3):
    #默认用户提示词
    if custom_prompt is None:
        custom_prompt = (
            "请判断以下句子的情感倾向（积极、消极或中性），并简要说明理由：\n"
            f"句子：{text}\n"
            "情感："
        )
    else:
        custom_prompt = custom_prompt.format(text=text)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": custom_prompt}
        ],
        temperature=temperature
    )
    #返回str：“123”
    return response.choices[0].message.content.strip()

# 设置 client:调用的api_key和api——url,custom:用户提示词,sys:系统提示词
client = openai.OpenAI(api_key="sk-11da4c508da341088bbf4ec4afc66bf8",base_url='https://api.deepseek.com')
input_path='test.csv'
df=pd.read_csv(input_path)
custom = (
    "请以简洁的方式分析这段话的情绪倾向（positive、negative）以及情感得分（置信度）保留三位小数：\n{text}\n"
    "请输出格式如下：\nlabel：xxx\nscore：xxx"
)
sys="你现在是金融领域的情感分析专家"
temperature=0.3

#情感分析
result=[]
for texts in tqdm(df['sentence']):
    result.append(sentiment_analysis(texts,sys_prompt=sys,custom_prompt=custom,temperature=temperature)) #'label：positive \nscore：0.923'

#结果处理成dict
result2dic=[]
for i in result:
    item={}
    lines=i.split("\n")
    for line in lines:
        key,value=line.split("：")
        item[key.strip()]=value.strip()
    result2dic.append(item)

#先转df再转csv
res=pd.DataFrame(result2dic)
res.to_csv("./out/pre.csv",index=False)

#准确率
acc=0
correct=0
# 情感映射表
label_map = {
    "negative": 0,
    "positive": 1

}

# 新增数值列
res["label_num"] = res["label"].map(label_map)
y_pre = res["label_num"].tolist()
y_label = df["label_num"].tolist()

for pre , label in zip(res['label_num'],df['label_num']):
    if pre==label: correct+=1

acc=correct/df['label_num'].shape[0]
report = classification_report(y_label,y_pre,digits=4, target_names=["negative","positive"])
print(f"acc:{acc:.4f}")
print(f"report:\n{report}")
