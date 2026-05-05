#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
T5-Large 单卡训练脚本 (优化版)
适用于 2x4090 环境，但只使用单卡
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"

def print_flush(*args, **kwargs):
    """强制刷新输出"""
    print(*args, **kwargs, flush=True)

def train_generative_retrieval():
    print_flush("\n" + "="*60)
    print_flush("T5-Large 单卡训练 (优化版)")
    print_flush("="*60)
    
    # 配置路径
    model_name = "/labmount/public/models/google/google-t5/t5-large"  
    train_file = "./data/t5_train.jsonl" 
    dev_file = "./data/t5_dev.jsonl"
    output_dir = "./model/t5_large_gr_single"
    
    # ==================== GPU 配置 ====================
    print_flush("\n0. GPU 信息:")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print_flush(f"   可用GPU数量: {gpu_count}")
        # 使用第一张卡
        torch.cuda.set_device(0)
        print_flush(f"   使用GPU: {torch.cuda.get_device_name(0)}")
        
        # 显示显存信息
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print_flush(f"   总显存: {total_memory:.1f} GB")
        
        # 清空缓存
        torch.cuda.empty_cache()
    else:
        print_flush("   警告: 未检测到GPU，使用CPU训练")
    
    # ==================== 加载模型 ====================
    print_flush("\n1. 加载 Tokenizer 和 Model...")
    
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    print_flush("   ✓ Tokenizer 加载完成")
    
    # 加载模型（单卡模式）
    model = T5ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # 使用bf16节省显存
        device_map="auto"             # 自动放置到GPU
    )
    
    # 启用梯度检查点（关键优化）
    model.gradient_checkpointing_enable()
    print_flush("   ✓ 梯度检查点已启用")
    
    # 显示模型信息
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print_flush(f"   ✓ 模型参数: {total_params:.1f}M (可训练: {trainable_params:.1f}M)")
    
    # ==================== 加载数据集 ====================
    print_flush("\n2. 加载数据集...")
    
    # 使用内存映射加速加载
    train_dataset = load_dataset(
        "json", 
        data_files={"train": train_file}, 
        split="train",
        keep_in_memory=False  # 避免内存爆炸
    )
    
    eval_dataset = load_dataset(
        "json", 
        data_files={"eval": dev_file}, 
        split="eval",
        keep_in_memory=False
    )
    
    print_flush(f"   ✓ 训练集: {len(train_dataset):,} 样本")
    print_flush(f"   ✓ 验证集: {len(eval_dataset):,} 样本")
    
    # 显示数据样本
    print_flush(f"\n   数据样例:")
    print_flush(f"     Input: {train_dataset[0]['input_text'][:80]}...")
    print_flush(f"     Output: {train_dataset[0]['target_text']}")
    
    # ==================== 数据预处理 ====================
    print_flush("\n3. 数据预处理 (Tokenization)...")
    print_flush("   使用多进程加速...")
    
    def preprocess_function(examples):
        """tokenization函数"""
        inputs = examples["input_text"]
        outputs = examples["target_text"]
        
        # 编码输入（不填充，后续使用collator动态填充）
        model_inputs = tokenizer(
            inputs, 
            max_length=128, 
            truncation=True, 
            padding=False
        )
        
        # 编码输出
        labels = tokenizer(
            outputs, 
            max_length=16, 
            truncation=True, 
            padding=False
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # 处理训练集
    print_flush("   处理训练集...")
    train_tokenized = train_dataset.map(
        preprocess_function, 
        batched=True, 
        batch_size=1000,
        remove_columns=train_dataset.column_names,
        num_proc=4,                    # 使用4个进程
        load_from_cache_file=True,     # 缓存结果
        desc="Tokenizing train"
    )
    
    # 处理验证集
    print_flush("   处理验证集...")
    eval_tokenized = eval_dataset.map(
        preprocess_function, 
        batched=True, 
        batch_size=1000,
        remove_columns=eval_dataset.column_names,
        num_proc=4,
        load_from_cache_file=True,
        desc="Tokenizing eval"
    )
    
    print_flush("   ✓ Tokenization 完成")
    
    # 显示tokenization后的长度统计
    avg_input_len = sum(len(x) for x in train_tokenized['input_ids'][:1000]) / 1000
    avg_label_len = sum(len(x) for x in train_tokenized['labels'][:1000]) / 1000
    print_flush(f"   平均输入长度: {avg_input_len:.1f}")
    print_flush(f"   平均输出长度: {avg_label_len:.1f}")
    
    # ==================== 训练配置 ====================
    print_flush("\n4. 配置训练参数...")
    
    # 根据显存调整batch size
    # 24GB显存，T5-Large + bf16 + 梯度检查点，可以开到48-64
    per_device_batch_size = 48
    gradient_accumulation_steps = 2
    effective_batch_size = per_device_batch_size * gradient_accumulation_steps
    
    # 计算训练步数
    num_epochs = 30
    steps_per_epoch = len(train_tokenized) // effective_batch_size
    total_steps = steps_per_epoch * num_epochs
    
    print_flush(f"   训练配置:")
    print_flush(f"     每卡Batch Size: {per_device_batch_size}")
    print_flush(f"     梯度累积步数: {gradient_accumulation_steps}")
    print_flush(f"     有效Batch Size: {effective_batch_size}")
    print_flush(f"     每Epoch步数: {steps_per_epoch:,}")
    print_flush(f"     总训练步数: {total_steps:,}")
    
    # 学习率调度
    warmup_steps = int(total_steps * 0.1)  # 10% warmup
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # 基础训练参数
        learning_rate=3e-4,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        
        # Batch配置
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        
        # 混合精度优化
        bf16=True,                     # 4090支持bf16
        fp16=False,
        tf32=True,                     # 启用TF32加速
        
        # 数据加载优化
        dataloader_num_workers=4,      # 多进程加载
        dataloader_pin_memory=True,    # 内存锁定
        dataloader_drop_last=True,     # 丢弃不完整batch
        dataloader_prefetch_factor=2,  # 预取数据
        
        # 评估和保存
        eval_strategy="epoch",         # 每个epoch评估
        save_strategy="epoch",         # 每个epoch保存
        save_total_limit=2,            # 只保留最后2个checkpoint
        logging_steps=50,              # 每50步打印日志
        logging_first_step=True,
        eval_on_start=False,           # 开始时不评估
        
        # 其他优化
        predict_with_generate=False,   # 评估时不生成（节省时间）
        generation_max_length=16,      # 生成最大长度
        generation_num_beams=1,        # 不使用beam search
        report_to="none",              # 禁用wandb/tensorboard
        remove_unused_columns=True,    # 删除未使用的列
        include_tokens_per_second=True, # 显示训练速度
        
        # 性能优化
        gradient_checkpointing=False,  # 已在模型上启用，这里不用设置
        optim="adamw_torch_fused",     # 使用融合优化器
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        
        # 调试选项
        disable_tqdm=False,            # 显示进度条
        load_best_model_at_end=True,   # 训练结束加载最佳模型
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, 
        model=model,
        label_pad_token_id=-100,       # 忽略padding的loss
        pad_to_multiple_of=8           # 8的倍数提高效率
    )
    
    # ==================== 准备训练 ====================
    print_flush("\n5. 初始化 Trainer...")
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    print_flush("   ✓ Trainer 初始化完成")
    
    # ==================== 开始训练 ====================
    print_flush("\n6. 开始训练！")
    print_flush("="*60)
    print_flush("训练监控:")
    print_flush("  - 每50步打印一次loss")
    print_flush("  - 每个epoch后评估并保存模型")
    print_flush("  - 预计每个epoch: 2-3小时")
    print_flush("="*60 + "\n")
    
    # 记录开始时间
    import time
    start_time = time.time()
    
    # 开始训练
    train_result = trainer.train()
    
    # 计算总时间
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    
    print_flush("\n" + "="*60)
    print_flush(f"训练完成！总耗时: {hours}小时 {minutes}分钟")
    print_flush("="*60)
    
    # 打印最终结果
    print_flush(f"\n最终训练结果:")
    print_flush(f"  最终 Loss: {train_result.training_loss:.4f}")
    print_flush(f"  训练步数: {train_result.global_step}")
    
    # ==================== 保存模型 ====================
    print_flush("\n7. 保存最终模型...")
    
    # 保存模型和tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print_flush(f"   ✓ 模型已保存到: {output_dir}")
    
    # 保存训练参数
    with open(os.path.join(output_dir, "training_args.txt"), "w") as f:
        f.write(f"训练时间: {hours}小时 {minutes}分钟\n")
        f.write(f"最终Loss: {train_result.training_loss:.4f}\n")
        f.write(f"训练步数: {train_result.global_step}\n")
        f.write(f"Batch Size: {per_device_batch_size}\n")
        f.write(f"梯度累积: {gradient_accumulation_steps}\n")
        f.write(f"学习率: 3e-4\n")
        f.write(f"Epochs: {num_epochs}\n")
    
    print_flush("\n" + "="*60)
    print_flush("所有任务完成！")
    print_flush("="*60)

if __name__ == "__main__":
    train_generative_retrieval()