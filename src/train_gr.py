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

os.environ["HF_DATASETS_LOCK"] = "off"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train_generative_retrieval():
    # os.environ["WANDB_PROJECT"] = "generative-retrieval-t5-large-irlab-slurm" 
    # wandb.init(name="t5-large-gr") 

    model_name = "~/data/workspace/wangyikang/models/t5-large"  
    train_file = "./data/t5_train.jsonl" 
    dev_file = "./data/t5_dev.jsonl"
    output_dir = "./model/t5_large_gr"
    
    print("1. 加载 Tokenizer 和 Model...")
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    print("2. 加载数据集...")
    train_dataset = load_dataset("json", data_files={"train": train_file})["train"]
    eval_dataset = load_dataset("json", data_files={"eval": dev_file})["eval"]

    print("3. 数据预处理 (Tokenization)...")
    def preprocess_function(examples): 
        inputs = examples["input_text"]
        outputs = examples["target_text"]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)
        labels = tokenizer(outputs, max_length=16, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tokenized = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    eval_tokenized = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names)

    # print("4. 配置训练参数...")
    # training_args = Seq2SeqTrainingArguments(
    #     output_dir=output_dir,
    #     eval_strategy="epoch",  
    #     save_strategy="epoch",        
    #     learning_rate=3e-4,             # 【关键修改 2】大模型通常建议稍微调低学习率，比如从 3e-4 降到 1e-4
    #     per_device_train_batch_size=8,  # 【关键修改 3】降低单卡 batch_size 防止 OOM
    #     per_device_eval_batch_size=8,
    #     gradient_accumulation_steps=4,  # 【关键修改 4】梯度累积：8 * 4 = 32，等效保持 32 的 global batch size
    #     gradient_checkpointing=True,    # 【关键修改 5】开启梯度检查点，用时间换空间，省下大量显存
    #     weight_decay=0.01,
    #     save_total_limit=3,          
    #     num_train_epochs=30,          
    #     bf16=torch.cuda.is_available(), # 4090 必须开 bf16
    #     logging_steps=10,            
    #     report_to="wandb", 
    #     ddp_find_unused_parameters=False,
    # )
    print("4. 配置 T5-Large 训练参数 (2张 4090 适配版)...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # ================= 1. 核心训练参数优化 =================
        learning_rate=3e-4,           # 保持不变
        num_train_epochs=30,          # 保持不变
        weight_decay=0.01,
        
        # ================= 2. 显存与 Batch Size 平衡 =================
        # 单卡维持 16 不变（刚好榨干 24G 显存）
        # 【关键修改】为了维持原本 256 的全局 Batch Size，梯度累加步数需要翻倍
        # 等效全局 Batch Size = 16 (单卡) * 8 (累加) * 2 (卡数) = 256
        per_device_train_batch_size=16, 
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=8,  # <--- 从 4 改为 8
        
        # ================= 3. 4090 专属算力释放 =================
        fp16=False,
        bf16=True,                    # 保持开启 bfloat16
        tf32=True,                    # 保持开启 tf32
        
        # ================= 4. 数据流加载优化 =================
        dataloader_num_workers=2,     # <--- 从 4 改为 2，匹配卡数，避免 CPU 线程抢占
        dataloader_pin_memory=True,   
        
        # ================= 5. 评估与保存策略 =================
        eval_strategy="epoch",        
        # save_strategy="epoch",
        # load_best_model_at_end=True,
        # metric_for_best_model="loss",
        # save_total_limit=3,
        logging_steps=100,
        
        # ================= 6. 核心提速配置 =================
        predict_with_generate=False,  
        
        report_to="tensorboard",
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
    
    # wandb.finish() 

if __name__ == "__main__":
    train_generative_retrieval()