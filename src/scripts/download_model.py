from modelscope.hub.api import HubApi
api = HubApi()
api.login('ms-f612db35-951e-4d7e-a1f0-31c65466b6da') # 确保这一行不报错

from modelscope.hub.snapshot_download import snapshot_download

# 指定模型 ID 和版本（默认 master）
# cache_dir 可以指定下载到哪个文件夹，不填则默认在 ~/.cache/modelscope
model_dir = snapshot_download("wykyty/Qwen2.5-1.5b-sft-gr", cache_dir="/data/wyh/IGAR/model")

print(f"模型已下载到: {model_dir}")