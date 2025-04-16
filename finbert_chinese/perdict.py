from transformers import AutoModelForSequenceClassification, BertTokenizerFast,AutoConfig
import pandas as pd
import os
import numpy as np
import torch

# ========== 1. 模型与Tokenizer加载 ==========
#联网调用yiyanghkust/finbert-tone-chinese，github中未上传local_model，选这个
# model_path = "yiyanghkust/finbert-tone-chinese"
# model = AutoModelForSequenceClassification.from_pretrained(model_path)
# tokenizer = BertTokenizerFast.from_pretrained(model_path)
# classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer,top_k=None,device=-1)




# 分类函数
def classify(texts):
    result=[]
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = logits.argmax().item()
            confidence = torch.softmax(logits, dim=1)[0, predicted_class_id].item()
        label = config.id2label.get(predicted_class_id, str(predicted_class_id))
        result.append({
            "Text": text,
            "Predicted Label": label,
            "Confidence": round(confidence, 4)
        })
        df=pd.DataFrame(result)
    return df


def write_to_csv(df, path="./output",file="finbert_ch_pre1.csv"):
    df.to_csv(os.path.join(path,file), index=False)
    print(f"✅ 分类结果已保存至 {os.path.join(path,file)}")




if __name__=="__main__":
    #本地配置储存path
    local_path = './local_model/finbert_chinese'
    config = AutoConfig.from_pretrained(local_path)
    #加载预训练的tokenizer，model，“trust_remote_code=True”：model采用modeling_finbert.py
    tokenizer = BertTokenizerFast.from_pretrained(local_path)
    model = AutoModelForSequenceClassification.from_pretrained(local_path, config=config, trust_remote_code=True)

    input_file = "input_sentences.csv"  # 假设CSV中有列名 sentence
    df_input = pd.read_csv(input_file)
    texts = df_input["sentence"].tolist()  #['此外宁德时代上半年实现出口约2GWh，同比增加200%+。', '今日股市波动剧烈，投资者信心不足。']

    model.eval()
    #情感分析判断
    df=classify(texts)
    write_to_csv(df)


