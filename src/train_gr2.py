import os
import torch
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

    model_name = "/home/aizoo/data/workspace/wangyikang/models/t5-large"  
    train_file = "./data/t5_train.jsonl" 
    output_dir = "./model/t5_large_gr"
    
    print("1. 加载 Tokenizer 和 Model...")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    print("2. 加载数据集 (专注训练集记忆)...")
    train_dataset = load_dataset("json", data_files={"train": train_file})["train"]

    print("3. 数据预处理 (Tokenization)...")
    def preprocess_function(examples):
        inputs = examples["input_text"]
        outputs = examples["target_text"]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)
        labels = tokenizer(outputs, max_length=16, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tokenized = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)

    print("4. 配置单张 A800 专属训练参数...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        save_strategy="epoch",        
        learning_rate=3e-4,             
        
        # ================= 1. 80GB 显存暴击配置 =================
        per_device_train_batch_size=64, # A800 显存管够，直接拉大 Batch Size
        gradient_accumulation_steps=4,  # 等效 Global Batch Size = 256
        gradient_checkpointing=False,   # 【关键修改】关掉！显存够大时不需要用时间换空间，关掉能大幅提升训练速度
        
        # ================= 2. Ampere 架构算力释放 =================
        bf16=True,                      # A800 原生完美支持 bf16
        tf32=True,                      # 【关键修改】提速神器，开启 TensorFloat-32 矩阵加速
        
        # ================= 3. 数据加载优化 =================
        dataloader_num_workers=8,       # 充分利用集群节点的 CPU 多线程喂数据
        dataloader_pin_memory=True,     # 加速 CPU 到 GPU 的数据搬运
        
        weight_decay=0.01,
        save_total_limit=3,          
        num_train_epochs=30,          
        logging_steps=10,            
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    print("5. 初始化 Trainer 并开始训练...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    print(f"6. 训练完成，保存最终模型到 {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    

if __name__ == "__main__":
    train_generative_retrieval()