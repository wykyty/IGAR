import os
import torch
import wandb
import numpy as np
from datasets import load_dataset
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

def train_generative_retrieval():
    os.environ["WANDB_PROJECT"] = "generative-retrieval-t5-base" 
    wandb.init(name="t5-base-gr") 

    model_name = "t5_base"  
    # data_file = "./data/seq2seq_indexing_data.jsonl" 
    data_file = "./data/seq2seq_train_data.jsonl" 
    output_dir = "./model/gr_t5_base_model_2"
    
    print("1. 加载 Tokenizer 和 Model...")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    print("2. 加载数据集...")
    dataset = load_dataset("json", data_files=data_file, split="train")
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    print("3. 数据预处理 (Tokenization)...")
    def preprocess_function(examples):
        inputs = examples["input_text"]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)
        labels = tokenizer(text_target=examples["target_text"], max_length=16, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tokenized = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    eval_tokenized = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names)

    # --- 核心新增：计算 Exact Match ---
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        
        # 1. 处理预测结果：如果是 tuple，取第一个元素
        if isinstance(preds, tuple):
            preds = preds[0]
            
        # 2. 将 -100 替换回 pad_token_id，否则 tokenizer 无法 decode
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # 3. 将 token IDs 解码回字符串形式的 docid
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 去除首尾空格，防止因为空格导致匹配失败
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        # 4. 计算 Exact Match (EM)
        # 只有当生成的 docid 和真实的 docid 完全一致时，才算正确 (1)，否则为错 (0)
        exact_matches = [
            1 if pred == label else 0 
            for pred, label in zip(decoded_preds, decoded_labels)
        ]
        
        em_score = sum(exact_matches) / len(exact_matches)

        # 返回的字典会被自动记录到 WandB 中
        return {"exact_match": em_score}

    print("4. 配置训练参数...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",  # 每个 epoch 结束时执行 compute_metrics
        learning_rate=3e-4,          
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,          
        num_train_epochs=5,          
        predict_with_generate=True,   # 必须为 True，否则 eval_preds 里不是生成的 tokens
        fp16=torch.cuda.is_available(), 
        logging_steps=10,            
        report_to="wandb",           
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    print("5. 初始化 Trainer 并开始训练...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics # 注入评测函数
    )

    trainer.train()

    print(f"6. 训练完成，保存最终模型到 {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    wandb.finish() 

if __name__ == "__main__":
    train_generative_retrieval()