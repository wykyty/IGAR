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

    # model_name = "~/data/workspace/wangyikang/models/t5-large"  
    model_name = "/labmount/public/models/google/google-t5/t5-large"
    train_file = "./data/t5_train.jsonl" 
    dev_file = "./data/t5_dev.jsonl"
    output_dir = "./models/t5_large_igar"
    
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

    # print("4. 配置 T5-Large 训练参数...")
    # training_args = Seq2SeqTrainingArguments(
    #     output_dir=output_dir,
    #     overwrite_output_dir=True,
        
    #     # ================= 1. 核心训练参数优化 =================
    #     learning_rate=3e-4,           # 模型变大，退回最稳妥的黄金学习率
    #     num_train_epochs=30,          # 提升训练轮数，确保 40 万文档被死死记住
    #     weight_decay=0.01,
        
    #     # ================= 2. 显存与 Batch Size 平衡 =================
    #     # 实际单卡吃 16 的显存，但每 4 步才更新一次梯度
    #     # 加上 4 张卡的 DDP 并行，等效全局 Batch Size = 16 * 4步 * 4卡 = 256
    #     per_device_train_batch_size=16, 
    #     per_device_eval_batch_size=16,
    #     gradient_accumulation_steps=3, 
        
    #     # ================= 3. 4090 专属算力释放 =================
    #     fp16=False,
    #     bf16=True,                    # 4090 完美支持 bfloat16，防溢出且收敛极稳
    #     tf32=True,                    # 开启 TensorFloat-32 矩阵加速
        
    #     # ================= 4. 数据流加载优化 =================
    #     dataloader_num_workers=6,
    #     dataloader_pin_memory=True,   # 加速 CPU 内存向 GPU 显存的传输
        
    #     # ================= 5. 评估与保存策略 =================
    #     eval_strategy="epoch",        # 踩坑提示：新版 transformers 用 eval_strategy 替代了 evaluation_strategy
    #     save_strategy="epoch",
    #     load_best_model_at_end=True,
    #     metric_for_best_model="loss",
    #     save_total_limit=3,
    #     logging_steps=100,
        
    #     # ================= 6. 核心提速配置 =================
    #     # 严禁在训练验证时生成文本，只测 Loss，将评估时间从几十分钟缩短到几秒
    #     predict_with_generate=False,  
        
    #     # report_to="wandb",
    # )

    from transformers import Seq2SeqTrainingArguments

    print("4. 配置 T5-Large 训练参数 (2x RTX 4090 性能拉满版)...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # ================= 1. 核心训练参数优化 =================
        learning_rate=3e-4,           
        num_train_epochs=30,          
        weight_decay=0.01,
        warmup_ratio=0.05,               
        lr_scheduler_type="cosine",      
        
        # ================= 2. 显存与 Batch Size 极限压榨 =================
        # 【核心修改】为了维持原来的收敛效果（全局 Batch Size = 256）
        # 现在的数学等式变成了：16(单卡BS) * 8(累加步数) * 2(卡数) = 256
        per_device_train_batch_size=16,  
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=8,   # 【修改】从 4 提升到 8，弥补少掉的两张卡
        # gradient_checkpointing=True,     # 依然保持开启，用时间换空间防爆显存
        
        # ================= 3. 4090 算力与底层加速释放 =================
        fp16=False,
        bf16=True,                    
        tf32=True,                    
        optim="adamw_torch_fused",       # Fused 算子融合，提速极快
        # torch_compile=True,              # PyTorch 2.x 图编译加速
        
        # ================= 4. 数据流与分布式 (DDP) 优化 =================
        dataloader_num_workers=4,        # 【修改】双卡对应调低 CPU worker 数量，4~6 均可，避免 CPU 上下文切换开销过大
        dataloader_pin_memory=True,   
        ddp_find_unused_parameters=False,# 多卡必关，省下巨量通信时间
        
        # ================= 5. 评估与保存策略 =================
        eval_strategy="epoch",        
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        save_total_limit=3,
        logging_steps=50,                
        
        # ================= 6. 核心提速配置 =================
        predict_with_generate=False,  
        
        # report_to="wandb",
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