from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import snapshot_download

api = HubApi()
api.login('ms-4f8dadc8-460b-4a10-95bd-6081964224c1') # 确保这一行不报错

# 指定模型 ID 和版本（默认 master）
# cache_dir 可以指定下载到哪个文件夹，不填则默认在 ~/.cache/modelscope

# model_dir = snapshot_download("wykyty/Qwen2.5-1.5b-sft-gr", cache_dir="/data/wyh/IGAR/model")
# print(f"模型已下载到: {model_dir}")


LOCAL_MODEL_DIR = './model/t5_large_igar'

# 上传模型
api.upload_folder(
    repo_id="wykyty/t5-large-gr",
    folder_path=LOCAL_MODEL_DIR
)