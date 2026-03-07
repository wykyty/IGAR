from transformers import AutoModelForCausalLM, AutoTokenizer

# 指向你刚才导出的模型路径
MODEL_PATH = "/home/aizoo/data/workspace/LLaMA-Factory/saves/merged/qwen2.5-1.5b-sft-merged" 

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")

# 模拟一个指令+查询
instruction = "Find documents that describe the latest timeline or updated status of the event."
query = "梯度下降算法在2024年的最新优化变体有哪些？"

# 构造 Prompt (格式要和 LLaMA-Factory 训练时用的 template 一致，Qwen 默认是 ChatML)
messages = [
    {"role": "user", "content": f"{instruction}\nInput: {query}"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50) # DocID 很短，50就够

print(tokenizer.decode(outputs[0], skip_special_tokens=True))