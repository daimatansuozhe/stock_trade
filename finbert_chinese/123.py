from transformers import AutoModelForSequenceClassification, BertTokenizerFast

model_path = "yiyanghkust/finbert-tone-chinese"
local_dir = "./local_model/finbert_chinese"

# 下载并保存模型和tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.save_pretrained(local_dir)

tokenizer = BertTokenizerFast.from_pretrained(model_path)
tokenizer.save_pretrained(local_dir)
