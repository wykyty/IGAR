import logging
import os

# 必须在 import deepsearcher 之前或初始化 config 之前设置
os.environ["OPENAI_API_KEY"] = "sk-tTo3MNJgAsRIvFgyuRCWfUKSVkBpIgBtPZi7yKTGGAmspl5D" 

from deepsearcher.offline_loading import load_from_local_files
from deepsearcher.online_query import query
from deepsearcher.configuration import Configuration, init_config

httpx_logger = logging.getLogger("httpx")  # disable openai's logger output
httpx_logger.setLevel(logging.WARNING)

current_dir = os.path.dirname(os.path.abspath(__file__))

config = Configuration()  # Customize your config here


# 配置 LLM
config.set_provider_config("llm", "OpenAI", {
    "model": "gpt-4o-mini",           
    "base_url": "http://123.129.219.111:3000/v1", 
    "api_key": "sk-tTo3MNJgAsRIvFgyuRCWfUKSVkBpIgBtPZi7yKTGGAmspl5D",
})

# 配置 Embedding
config.set_provider_config("embedding", "OpenAIEmbedding", {
    "model": "text-embedding-ada-002",
    "base_url": "http://123.129.219.111:3000/v1", 
    "api_key": "sk-tTo3MNJgAsRIvFgyuRCWfUKSVkBpIgBtPZi7yKTGGAmspl5D",
})

init_config(config=config)

# You should clone the milvus docs repo to your local machine first, execute:
# git clone https://github.com/milvus-io/milvus-docs.git
# Then replace the path below with the path to the milvus-docs repo on your local machine
# import glob
# all_md_files = glob.glob('xxx/milvus-docs/site/en/**/*.md', recursive=True)
# load_from_local_files(paths_or_directory=all_md_files, collection_name="milvus_docs", collection_description="All Milvus Documents")

# Hint: You can also load a single file, please execute it in the root directory of the deep searcher project
load_from_local_files(
    paths_or_directory=os.path.join(current_dir, "data/WhatisMilvus.pdf"),
    collection_name="milvus_docs",
    collection_description="All Milvus Documents",
    # force_new_collection=True, # If you want to drop origin collection and create a new collection every time, set force_new_collection to True
)

question = "Write a report comparing Milvus with other vector databases."

final_answer, all_retrieved_results, consumed_token = query(question, max_iter=1)
print(f"Consumed tokens: {consumed_token}")

# save to file


