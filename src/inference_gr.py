import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# ==========================================
# 1. 定义前缀树 (Trie) 结构，用于受限解码
# ==========================================
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class DocidTrie:
    def __init__(self, tokenizer, valid_docids):
        self.root = TrieNode()
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        
        print("正在构建合法 DocID 的 Trie 树...")
        for docid in valid_docids:
            # 将文本形式的 docid 转化为 token ID 序列 (不加特殊符)
            tokens = tokenizer.encode(docid, add_special_tokens=False)
            self._insert(tokens)

    def _insert(self, tokens):
        node = self.root
        for token in tokens:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
        node.is_end = True

    def get_allowed_tokens(self, prefix_tokens):
        """给定已生成的 prefix，返回下一步允许生成的 token 列表"""
        node = self.root
        for token in prefix_tokens:
            if token not in node.children:
                return [] # 理论上不会走到这里
            node = node.children[token]
            
        allowed = list(node.children.keys())
        # 如果当前节点是某个合法 docid 的结尾，允许模型输出 EOS (结束符)
        if node.is_end:
            allowed.append(self.eos_token_id)
        return allowed

# ==========================================
# 2. 推理主函数
# ==========================================
def retrieve_top_k(model_path, corpus_file, queries, top_k=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("1. 加载模型与分词器...")
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()
    
    print("2. 加载 Corpus 并提取所有合法的 docid...")
    corpus = []
    valid_docids = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                doc = json.loads(line)
                corpus.append(doc)
                valid_docids.append(doc["docid"])
    
    # 建立映射表，方便后续通过 docid 查原文
    docid_to_title = {doc["docid"]: doc["title"] for doc in corpus}
    docid_to_text = {doc["docid"]: doc["text"] for doc in corpus}

    print("3. 初始化受限解码 Trie 树...")
    trie = DocidTrie(tokenizer, valid_docids)

    # 定义受限解码回调函数
    def prefix_allowed_tokens_fn(batch_id, input_ids):
        # input_ids 包含了目前 Decoder 已经生成的所有 token (作为一维 Tensor)
        prefix = input_ids.tolist()
        
        # T5 的 Decoder 输入默认以 pad_token_id (通常是 0) 开始，我们需要跳过它
        if prefix and prefix[0] == trie.pad_token_id:
            prefix = prefix[1:]
            
        allowed_tokens = trie.get_allowed_tokens(prefix)
        
        # 防止意外断路，提供 EOS 兜底
        if not allowed_tokens:
            return [trie.eos_token_id]
        return allowed_tokens

    print("\n4. 开始推理...")
    for query in queries:
        input_text = f"Retrieve docid for: {query}"
        inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True).to(device)

        with torch.no_grad():
            # 使用 Beam Search 和受限解码
            outputs = model.generate(
                **inputs,
                max_length=16,
                num_beams=top_k,               # Beam 宽度
                num_return_sequences=top_k,    # 返回的序列数量 (必须 <= num_beams)
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn, # 开启受限解码
                early_stopping=True
            )

        # 解码生成的 docids
        generated_docids = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        print(f"\nQuery: {query}")
        print("-" * 40)
        for i, docid in enumerate(generated_docids):
            docid = docid.strip()
            title = docid_to_title.get(docid, "⚠️ 未知文档 (这在开启受限解码后不应出现)")
            text = docid_to_text.get(docid, "")
            print(f"Rank {i+1}: ID [{docid}] -> Title: {title}\n Text: {text}\n")

if __name__ == "__main__":
    # 填入你刚才训练保存的模型路径
    trained_model_dir = "./model/t5_large_igar" 
    corpus_data_path = "./data/corpus_with_ids.jsonl" 
    
    # 测试一些新的 Query
    test_queries = [
        # "Who was born later, Lisbeth Cathrine Amalie Rose or Princess Raiyah Bint Hussein?",
        # "When did the Battle of Lincoln happen?",
        # "Which country the director of film One Law For The Woman is from?",
        "When did John V, Prince Of Anhalt-Zerbst's father die?"
    ]
    
    retrieve_top_k(trained_model_dir, corpus_data_path, test_queries, top_k=10)

"""
1. 加载模型与分词器...
2. 加载 Corpus 并提取所有合法的 docid...
3. 初始化受限解码 Trie 树...
正在构建合法 DocID 的 Trie 树...

4. 开始推理...

Query: Who was born later, Lisbeth Cathrine Amalie Rose or Princess Raiyah Bint Hussein?
----------------------------------------
Rank 1: ID [0.8.8.1.0] -> Title: Marie Leszczyńska
"text": "Maria Karolina Zofia Felicja Leszczyńska (23 June 1703 – 24 June 1768), also known as Marie Leczinska , was a Polish princess and French queen consort. The daughter of King Stanisław Leszczyński—Stanislaus I of Poland (later Duke of Lorraine)–and Catherine Opalińska, she married King Louis XV of France and became queen consort of France. She served in that role for 42 years from 1725 until her death in 1768, the longest service of any queen of France, and was popular due to her generosity and piety. She was the grandmother of Louis XVI, Louis XVIII and Charles X of France."

Rank 2: ID [0.8.8.8.0] -> Title: Catherine of Bosnia, Grand Princess of Hum
"text": "\" For other people named Catherine of Bosnia, see Catherine of Bosnia( disambiguation).\" Catherine of Bosnia( Bosnian:\" Katarina Kotromanić Stjepanova\")( b. 1294- d. 1355) was sister of Stephen II, Ban of Bosnia."

Rank 3: ID [0.8.8.6.0] -> Title: Princess Elisabetta of Belgium
"text": "Princess Elisabetta of Belgium, Archduchess of Austria- Este( née\" Nob.\" Elisabetta Rosboch von Wolkenstein on 9 September 1987) is the wife of Prince Amedeo of Belgium, Archduke of Austria- Este."


Query: When did the Battle of Lincoln happen?
----------------------------------------
Rank 1: ID [6.4.7.1] -> Title: Howard Bretherton
Rank 2: ID [4.5.9.1.0] -> Title: Edmund of Langley, 1st Duke of York
Rank 3: ID [4.5.9.2.0] -> Title: Humphrey Stafford, 1st Earl of Devon



Query: Which country the director of film One Law For The Woman is from?
----------------------------------------
Rank 1: ID [6.4.5.2.0] -> Title: Sidney Olcott
"text": "Sidney Olcott( September 20, 1872 – December 16, 1949) was a Canadian- born film producer, director, actor and screenwriter."

Rank 2: ID [1.7.9.1.0] -> Title: Fernando Cortés
"text": "Fernando \"Papi\" Cortés (October 4, 1909 – 1979) was a Puerto Rican film actor, writer and director. He was born in San Juan, Puerto Rico, but he spent most of his adult life in Mexico City, where he died. On 1932, while in New York City, Fernando Cortés married Puerto Rican childhood friend María del Pilar Cordero, who adopted the stage name of Mapy Cortés. The couple soon traveled to Spain with a Cuban theatrical troupe. They worked on the Spanish stage, radio and film until the outbreak of the Civil War in 1936. Fernando progressively began to take a backseat as actor and baritone and focused on promoting the career of his wife Mapy, who became a noted \"vedette\" (showgirl with star status) in Barcelona. After the Spanish Civil War interrupted their careers, the couple worked in New York, San Juan, Buenos Aires, Havana and Caracas, occasionally starring in movies. They arrived to Mexico City in late 1940 and made their stage debut at the Teatro Follies, in a show headlined by the popular Mexican comedian Cantinflas. Despite early struggles to become household names, Mapy achieved Mexican film stardom in late 1941 and the couple settled in Mexico City. Initially, Fernando Cortés played supporting roles in his wife's films. He then made a successful debut as director with \"La pícara Susana\" (1945) a comedy vehicle for his wife. On March 1954, Fernando and Mapy Cortés returned to Puerto Rico to help launch local television. Cortés became the first director at WKAQ-TV, Channel 2, and the couple co-starred in \"Mapy y Papi\", the first Puerto Rican sitcom. Despite their success on local TV, the couple returned the following year to Mexico City, which offered more opportunities. The couple starred in a Mexican version of their Puerto Rican sitcom and Mapy returned to the stage. Fernando Cortés became known as a reliable director of Mexican comedies on stage, television and film. After writing and directing star vehicles for his wife Mapy in the 1940s and comedians like Resortes and Tin-Tan in the 1950s, Fernando Cortés produced and directed Puerto Rican co-productions in the 1960s and launched the film career of La India María in the 1970s."

Rank 3: ID [6.4.5.6.0] -> Title: Frank Lloyd
"text": "Frank William George Lloyd( 2 February 1886 – 10 August 1960) was a British- born American film director, actor, scriptwriter, and producer. He was among the founders of the Academy of Motion Picture Arts and Sciences, and was its president from 1934 to 1935."



Query: Which country the director of film One Law For The Woman is from?
----------------------------------------
Rank 1: ID [5.4.6.8.7.0.0] -> Title: The Veiled Woman
 Text: The Veiled Woman is a 1929 American drama film directed by Emmett J. Flynn and starring Lia Torá and Walter McGrail.

Rank 2: ID [5.4.6.8.7.4.0] -> Title: Swamp Woman
 Text: Swamp Woman is a 1941 American film directed by Elmer Clifton.

Rank 3: ID [5.6.4.2.6.4.0.0] -> Title: Attack of the 50 Ft. Woman (1993 film)
 Text: Attack of the 50 Ft. Woman is a 1993 television film, it is a remake of the 1958 film of the same name. Directed by Christopher Guest and starring Daryl Hannah and Daniel Baldwin, the film premiered on HBO on December 11, 1993, and was later theatrically released in the United Kingdom, France and Germany.

 

 1. 加载模型与分词器...
2. 加载 Corpus 并提取所有合法的 docid...
3. 初始化受限解码 Trie 树...
正在构建合法 DocID 的 Trie 树...

4. 开始推理...

Query: Which country the director of film One Law For The Woman is from?
----------------------------------------
Rank 1: ID [5.6.4.6.1.6.0] -> Title: One Law for the Woman
 Text: One Law for the Woman is a 1924 American silent western film directed by Dell Henderson and starring Cullen Landis, Mildred Harris and Cecil Spooner.

Rank 2: ID [5.4.5.2.3.1] -> Title: Dell Henderson
 Text: George Delbert "Dell" Henderson (July 5, 1877 – December 2, 1956) was a Canadian-American actor, director, and writer. He began his long and prolific film career in the early days of silent film.

Rank 3: ID [4.6.9.7.1.4.0] -> Title: J. Gordon Edwards
 Text: James Gordon Edwards( June 24, 1867 – December 31, 1925) was an American film director, producer, and writer who began his career as a stage actor and stage director.


"""
