from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, BertTokenizerFast
import pandas as pd
import os
# ========== 1. 模型与Tokenizer加载 ==========
#联网调用yiyanghkust/finbert-tone-chinese，github中未上传local_model，选这个
# model_path = "yiyanghkust/finbert-tone-chinese"
# model = AutoModelForSequenceClassification.from_pretrained(model_path)
# tokenizer = BertTokenizerFast.from_pretrained(model_path)
local_path='./local_model/finbert_chinese'
model = AutoModelForSequenceClassification.from_pretrained(local_path)
tokenizer = BertTokenizerFast.from_pretrained(local_path)
#device=0 表示第 0 块 GPU,device=-1 表示使用CPU
classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer,top_k=None,device=-1)

# ========== 2. 读取CSV文件 ==========
input_file = "input_sentences.csv"  # 假设CSV中有列名 sentence
df_input = pd.read_csv(input_file)

# 检查是否存在 sentence 列
if "sentence" not in df_input.columns:
    raise ValueError("CSV文件必须包含'sentence'这一列。")

# ========== 3. 情感预测 ==========
results = []
for sentence in df_input["sentence"]:
    prediction_scores = classifier(sentence)[0]
    top_result = max(prediction_scores, key=lambda x: x['score'])
    results.append({
        'sentence': sentence,
        'prediction': top_result['label'],
        'sentiment_score': round(top_result['score'], 4)
    })

# ========== 4. 保存为新的CSV文件 ==========
df_output = pd.DataFrame(results)
if not os.path.exists('./output'):
    os.mkdir('./output')

output_file = "./output/finbert_ch_pre.csv"
df_output.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"✅ 情感分析完成，结果已保存为：{output_file}")
