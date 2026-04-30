import json
import random
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

def check_indexing_memorization(model_path, corpus_file, num_samples=100, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"1. 正在加载模型和分词器: {model_path} ...")
    tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()

    print(f"2. 正在从 {corpus_file} 中随机抽取 {num_samples} 篇文档...")
    all_docs = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            all_docs.append(json.loads(line))
            
    random.seed(seed)
    sampled_docs = random.sample(all_docs, min(num_samples, len(all_docs)))

    print("3. 开始进行记忆准确率 (Exact Match) 测试...")
    correct_count = 0
    
    for doc in tqdm(sampled_docs, desc="Testing Memorization"):
        title = doc.get("title", "")
        text = doc.get("text", "")
        gold_docid = str(doc.get("docid", ""))
        
        # 严格按照训练时的格式构建 Prompt
        input_text = f"Doc: {title} {text}"
        
        # 为了防止爆显存和加快速度，限制一下输入长度 (和训练时保持一致，比如 512)
        inputs = tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).to(device)
        
        with torch.no_grad():
            # 记忆任务不需要复杂的 Beam Search，直接 Greedy Search (num_beams=1) 即可
            outputs = model.generate(
                inputs["input_ids"],
                max_length=16,
                num_beams=1 
            )
            
        pred_docid = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        if pred_docid == gold_docid:
            correct_count += 1
        else:
            # 打印几个错误的例子看看它到底生成了什么鬼东西
            if correct_count + 1 <= 5: # 只打印前几个错的，防止刷屏
                print(f"\n[错题分析] Title: {title[:30]}...")
                print(f"   -> 标准答案 (Gold) : {gold_docid}")
                print(f"   -> 模型生成 (Pred) : {pred_docid}")

    accuracy = correct_count / len(sampled_docs) * 100
    print("\n" + "="*40)
    print("🧠 模型底层记忆力 (Indexing) 诊断报告")
    print("="*40)
    print(f"测试样本数: {len(sampled_docs)}")
    print(f"完全命中数: {correct_count}")
    print(f"记忆准确率: {accuracy:.2f}%")
    print("="*40)

if __name__ == "__main__":
    # ⚠️ 请将这里的路径替换为你目前训练好的最优 checkpoint 文件夹路径
    MODEL_PATH = "./models/t5_igar_model/checkpoint-xxxx" 
    CORPUS_FILE = "./data/2wiki_corpus_semantic_ids.jsonl"
    
    check_indexing_memorization(MODEL_PATH, CORPUS_FILE, num_samples=100)