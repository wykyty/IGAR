from __future__ import annotations

import json
from typing import List, Optional, Union

import numpy as np
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from deepsearcher.utils import log
from deepsearcher.vector_db.base import RetrievalResult
from deepsearcher.vector_db.milvus import Milvus


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False


class DocidTrie:
    def __init__(self, tokenizer: T5Tokenizer, valid_doc_ids: List[str]):
        self.root = TrieNode()
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        for docid in valid_doc_ids:
            tokens = tokenizer.encode(docid, add_special_tokens=False)
            self._insert(tokens)

    def _insert(self, tokens: List[int]) -> None:
        node = self.root
        for token in tokens:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
        node.is_end = True

    def get_allowed_tokens(self, prefix_tokens: List[int]) -> List[int]:
        node = self.root
        for token in prefix_tokens:
            if token not in node.children:
                return []
            node = node.children[token]

        allowed_tokens = list(node.children.keys())
        if node.is_end:
            allowed_tokens.append(self.eos_token_id)
        return allowed_tokens


class ConstrainedGenerativeRetrievalDB(Milvus):
    """
    Generative retrieval backed by Milvus using plain trie-constrained decoding.

    This class does not apply any agentic lookahead or external reranking during
    decoding. It only restricts generation to valid doc ids and then fetches the
    matched chunks from Milvus by the `reference` field.
    """

    def __init__(
        self,
        gr_model_path: str,
        valid_doc_ids: Optional[List[str]] = None,
        prompt_template: str = "Query: {query}",
        input_max_length: int = 128,
        output_max_length: int = 16,
        model_device: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if not gr_model_path:
            raise ValueError("`gr_model_path` is required for constrained generative retrieval.")

        self.valid_doc_ids = [str(docid) for docid in (valid_doc_ids or [])]
        self.valid_doc_id_set = set(self.valid_doc_ids)
        self.prompt_template = prompt_template
        self.input_max_length = input_max_length
        self.output_max_length = output_max_length
        self.model_device = model_device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = T5Tokenizer.from_pretrained(gr_model_path)
        if self.model_device == "cuda":
            self.llm = T5ForConditionalGeneration.from_pretrained(
                gr_model_path,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            self.llm = T5ForConditionalGeneration.from_pretrained(gr_model_path).to(
                self.model_device
            )
        self.llm.eval()

        self.trie = DocidTrie(self.tokenizer, self.valid_doc_ids)

        if not self.valid_doc_ids:
            log.warning(
                "ConstrainedGenerativeRetrievalDB was initialized without valid_doc_ids; "
                "decoding will not be able to generate usable doc ids."
            )

    def _prefix_allowed_tokens_fn(self, batch_id: int, input_ids: torch.LongTensor) -> List[int]:
        del batch_id

        prefix_tokens = input_ids.tolist()
        if prefix_tokens and prefix_tokens[0] == self.trie.pad_token_id:
            prefix_tokens = prefix_tokens[1:]

        allowed_tokens = self.trie.get_allowed_tokens(prefix_tokens)
        if not allowed_tokens:
            return [self.trie.eos_token_id]
        return allowed_tokens

    def _build_prompt(self, instruct: str, query: str) -> str:
        try:
            return self.prompt_template.format(instruct=instruct, query=query)
        except KeyError:
            log.warning(
                "prompt_template only supports known placeholders; falling back to query-only prompt."
            )
            return f"Query: {query}"

    def _generate_docids(self, instruct: str, query: str, top_k: int = 10) -> List[str]:
        if not query or not self.valid_doc_ids:
            return []

        prompt = self._build_prompt(instruct=instruct, query=query)

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.input_max_length,
                truncation=True,
            ).to(self.model_device)

            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs,
                    max_length=self.output_max_length,
                    num_beams=max(top_k, 1),
                    num_return_sequences=max(top_k, 1),
                    prefix_allowed_tokens_fn=self._prefix_allowed_tokens_fn,
                    early_stopping=True,
                )

            generated_docids = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            normalized_docids = [
                docid.strip()
                for docid in generated_docids
                if docid.strip() and docid.strip() in self.valid_doc_id_set
            ]
            return list(dict.fromkeys(normalized_docids))
        except Exception as exc:
            log.error(f"failed to generate constrained doc ids: {exc}")
            return []

    def search_data(
        self,
        collection: Optional[str],
        vector: Union[np.ndarray, List[float]],
        query_text: Optional[str] = None,
        instruct: Optional[str] = None,
        top_k: int = 10,
        *args,
        **kwargs,
    ) -> List[RetrievalResult]:
        del vector, args, kwargs

        if not collection:
            collection = self.default_collection

        safe_instruct = instruct or "Retrieve relevant documents for the query."
        log.color_print("[STEP 1] Generate candidate doc ids with constrained decoding")
        log.color_print(f"Query: {query_text}")
        target_doc_ids = self._generate_docids(safe_instruct, query_text or "", top_k)

        if not target_doc_ids:
            return []

        log.color_print("\n[STEP 2] Fetch matched chunks from Milvus", color="cyan")
        serialized_ids = [json.dumps(docid, ensure_ascii=False) for docid in target_doc_ids]
        expr = f"reference in [{', '.join(serialized_ids)}]"
        log.color_print(f"Filter Expression: {expr}", color="blue")

        try:
            raw_results = self.client.query(
                collection_name=collection,
                filter=expr,
                output_fields=["text", "reference", "metadata"],
                limit=len(target_doc_ids),
            )
        except Exception as exc:
            log.color_print(f"Milvus query failed: {exc}")
            return []

        result_by_reference = {
            item.get("reference", ""): item for item in raw_results if item.get("reference")
        }

        results: List[RetrievalResult] = []
        for docid in target_doc_ids[:top_k]:
            item = result_by_reference.get(docid)
            if not item:
                continue

            text = item.get("text", "")
            preview = text.replace("\n", " ")
            log.color_print(
                f" Result [{len(results) + 1}] | ID: {docid} | Content: {preview[:60]}...",
                color="white",
            )
            results.append(
                RetrievalResult(
                    embedding=[],
                    text=text,
                    reference=item.get("reference", ""),
                    score=1.0,
                    metadata=item.get("metadata", {}),
                )
            )

        if not results:
            log.color_print("No generated doc ids were found in Milvus.", color="yellow")

        return results
