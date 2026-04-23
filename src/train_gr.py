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
    os.environ["WANDB_PROJECT"] = "generative-retrieval-t5-large-irlab-slurm" 
    wandb.init(name="t5-large-gr") 

    model_name = "/labmount/public/models/google/google-t5/t5-large"  
    train_file = "./data/t5_train.jsonl" 
    dev_file = "./data/t5_dev.jsonl"
    output_dir = "./model/t5_large_gr"
    
    print("1. 加载 Tokenizer 和 Model...")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    print("2. 加载数据集...")
    train_dataset = load_dataset("json", data_files={"train": train_file})["train"]
    eval_dataset = load_dataset("json", data_files={"eval": dev_file})["eval"]

    print("3. 数据预处理 (Tokenization)...")
    def preprocess_function(examples):
        inputs = examples["input_text"]
        outputs = examples["output_text"]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)
        labels = tokenizer(outputs, max_length=16, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tokenized = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    eval_tokenized = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names)

    print("4. 配置训练参数...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",  
        save_strategy="epoch",        
        learning_rate=3e-4,             # 【关键修改 2】大模型通常建议稍微调低学习率，比如从 3e-4 降到 1e-4
        per_device_train_batch_size=8,  # 【关键修改 3】降低单卡 batch_size 防止 OOM
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,  # 【关键修改 4】梯度累积：8 * 4 = 32，等效保持 32 的 global batch size
        gradient_checkpointing=True,    # 【关键修改 5】开启梯度检查点，用时间换空间，省下大量显存
        weight_decay=0.01,
        save_total_limit=3,          
        num_train_epochs=30,          
        bf16=torch.cuda.is_available(), # 4090 必须开 bf16
        logging_steps=10,            
        report_to="wandb", 
        ddp_find_unused_parameters=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    print("5. 初始化 Trainer 并开始训练...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    print(f"6. 训练完成，保存最终模型到 {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    wandb.finish() 

if __name__ == "__main__":
    train_generative_retrieval()