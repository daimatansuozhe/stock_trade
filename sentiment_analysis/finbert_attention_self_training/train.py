import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoConfig,
    BertTokenizerFast,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1️⃣ 配置路径
local_model_path = "local_model/finbert_chinese"
train_data_path = "train_dataset.csv"  # 需包含列 sentence 和 label

# 2️⃣ 加载 tokenizer 和 config
tokenizer = BertTokenizerFast.from_pretrained(local_model_path)
config = AutoConfig.from_pretrained(local_model_path)

# 3️⃣ 加载并转换数据集
df = pd.read_csv(train_data_path)
df = df[["sentence", "label"]].dropna()

# pandas 转 datasets
dataset = Dataset.from_pandas(df)

# 4️⃣ 分词处理函数
def tokenize_function(example):
    return tokenizer(
        example["sentence"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 5️⃣ 加载模型
model = BertForSequenceClassification.from_pretrained(local_model_path, config=config)

# 6️⃣ 指定训练参数，主要参数：
training_args = TrainingArguments(
    output_dir="../model_save/output_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="../model_save/logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",  # 或 eval_accuracy、eval_f1 等
    greater_is_better=False,  # 如果是最小化 loss，设为 False
    evaluation_strategy="epoch",  # 每个 epoch 结束后进行评估
    early_stopping_patience=2,  # 早停参数设置，如果在 2 个 epoch 内没有进展，则停止训练
)

# 7️⃣ 自定义评估函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# 8️⃣ 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # 加入 EarlyStopping 回调，早听执行
)

# 9️⃣ 开始训练
trainer.train()
