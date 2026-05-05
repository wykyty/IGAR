# train_gr.py  2 * 4090, 42h
import os
import json
import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

class JSONLDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, max_input_len=128, max_target_len=16):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        
        # 读取所有数据
        self.data = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        print(f"加载了 {len(self.data)} 条数据从 {jsonl_file}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['input_text']
        target_text = item['target_text']
        
        # Tokenize
        inputs = self.tokenizer(
            input_text, 
            max_length=self.max_input_len, 
            truncation=True, 
            padding='max_length',
            return_tensors='pt'
        )
        labels = self.tokenizer(
            target_text,
            max_length=self.max_target_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels['input_ids'].squeeze()
        }

def train_generative_retrieval():
    # 设置环境
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # 为每个进程设置缓存
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    cache_dir = f"/tmp/hf_cache_{os.getpid()}_{local_rank}"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    
    model_name = "/labmount/public/models/google/google-t5/t5-large"
    train_file = "./data/t5_train.jsonl"
    dev_file = "./data/t5_dev.jsonl"
    output_dir = f"./model/t5_large_igar_2"
    
    print("1. 加载 Tokenizer 和 Model...")
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    print("2. 加载数据集...")
    train_dataset = JSONLDataset(train_file, tokenizer)
    eval_dataset = JSONLDataset(dev_file, tokenizer)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # ========== 核心提速参数 ==========
        learning_rate=3e-4,
        num_train_epochs=20,
        weight_decay=0.01,
        
        # 增大 batch size（4090 24GB 显存）
        per_device_train_batch_size=16,  # 从 8 增加到 16
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=8,   # 从 16 减到 8，保持总 batch size = 16*8*2=256
        
        # 启用更多加速
        bf16=True,
        tf32=True,
        
        # ========== 关键提速设置 ==========
        dataloader_num_workers=8,        # 增加数据加载线程
        dataloader_pin_memory=True,
        remove_unused_columns=False,      # 避免不必要的列检查
        
        # 优化器和调度器
        optim="adamw_torch_fused",        # 融合优化器，更快
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        
        # 减少日志频率
        logging_steps=200,                # 减少日志输出
        
        # 评估和保存
        eval_strategy="steps",            # 改为按步数评估
        eval_steps=500,                   # 每500步评估一次
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        save_total_limit=2,               # 减少保存的模型数量
        
        predict_with_generate=False,
        local_rank=local_rank if torch.cuda.is_available() else -1,
        ddp_find_unused_parameters=False,
        report_to="none",
        
        # ========== 梯度检查点（如果显存不够）==========
        # gradient_checkpointing=True,    # 如果显存不够就取消注释
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
    
    print("4. 开始训练...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    train_generative_retrieval()