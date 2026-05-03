import argparse
import json
import logging
import os
import re
import string
from collections import Counter
from datetime import datetime

from dotenv import load_dotenv
from tqdm import tqdm

from deepsearcher.configuration import Configuration, init_config
from deepsearcher.offline_loading import load_from_local_files
from deepsearcher.online_query import naive_rag_query, query


DEFAULT_TOP_K = (1, 5, 10)


def normalize_answer(text):
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def compute_f1(prediction, gold_answer):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold_answer).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def parse_args():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace = os.path.dirname(current_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output_dir = os.path.join(workspace, "output", timestamp)

    parser = argparse.ArgumentParser(description="Evaluate retrieval, QA, and token efficiency.")
    parser.add_argument("--query-mode", choices=["rag", "naive_rag"], default="naive_rag")
    parser.add_argument("--max-iter", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=200, help="Use 0 for full dataset.")
    parser.add_argument("--top-k", type=str, default="1,5,10")
    parser.add_argument("--reload-data", action="store_true")
    parser.add_argument("--corpus-file", type=str, default=os.path.join(workspace, "data", "corpus_with_ids.jsonl"))
    parser.add_argument("--dev-file", type=str, default=os.path.join(workspace, "data", "2wiki", "dev.json"))
    parser.add_argument("--output-dir", type=str, default=default_output_dir)
    parser.add_argument("--output-file", type=str, default="")
    parser.add_argument("--invalid-log-file", type=str, default="")

    args = parser.parse_args()

    if args.max_iter < 1:
        parser.error("--max-iter must be >= 1")
    if args.max_samples < 0:
        parser.error("--max-samples must be >= 0")

    top_k_values = []
    for part in args.top_k.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            parser.error("--top-k must contain positive integers")
        top_k_values.append(value)
    if not top_k_values:
        parser.error("--top-k must contain at least one integer")

    args.top_k_values = tuple(dict.fromkeys(top_k_values))
    args.max_samples = None if args.max_samples == 0 else args.max_samples
    args.output_file = args.output_file or os.path.join(args.output_dir, "evaluation_results.json")
    args.invalid_log_file = args.invalid_log_file or os.path.join(args.output_dir, "invalid_queries_log.json")
    return args


def setup_pipeline(corpus_file, reload_data=False):
    load_dotenv()
    base_url = os.getenv("BASE_URL")
    api_key = os.getenv("API_KEY")
    model_name = os.getenv("MODEL_NAME", "LongCat-Flash-Lite")

    if not base_url or not api_key:
        raise ValueError("Missing BASE_URL or API_KEY in environment variables.")

    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
    vllm_logger = logging.getLogger("vllm")
    vllm_logger.propagate = False
    vllm_logger.handlers.clear()
    handler = logging.FileHandler("vllm_server_log.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    vllm_logger.setLevel(logging.INFO)
    vllm_logger.addHandler(handler)

    config = Configuration()
    config.set_provider_config("llm", "OpenAI", {
        "model": model_name,
        "base_url": base_url,
        "api_key": api_key,
    })
    config.set_provider_config("embedding", "SentenceTransformerEmbedding", {
        "model": "BAAI/bge-large-en-v1.5",
        "batch_size": 256,
    })
    config.set_provider_config("file_loader", "JsonFileLoader", {
        "text_key": "text",
        "id_key": "docid",
    })
    init_config(config=config)

    if reload_data:
        print(f"Reloading corpus from: {corpus_file}")
        load_from_local_files(
            paths_or_directory=corpus_file,
            collection_name="wiki",
            collection_description="2wiki corpus",
        )
    else:
        print("Skipping corpus reload. Using existing retrieval store state.")


def evaluate(args):
    print("1. Loading corpus title -> docid mapping...")
    title_to_docid = {}
    with open(args.corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            title = doc.get("title")
            docid = doc.get("docid")
            if title and docid is not None:
                title_to_docid[str(title)] = str(docid)
    print(f"   - Loaded {len(title_to_docid)} mappings")

    print(f"2. Loading dev set from {args.dev_file} ...")
    with open(args.dev_file, "r", encoding="utf-8") as f:
        dev_data = json.load(f)
    if args.max_samples:
        dev_data = dev_data[:args.max_samples]
    print(f"   - Loaded {len(dev_data)} samples")

    details = []
    invalid_queries = []
    retrieval_sums = {"mrr": 0.0}
    for k in args.top_k_values:
        retrieval_sums[f"partial_hits@{k}"] = 0.0
        retrieval_sums[f"strict_hits@{k}"] = 0.0
    total_em = 0.0
    total_f1 = 0.0
    total_tokens = 0.0

    print("3. Running evaluation...")
    for item in tqdm(dev_data, desc="Evaluating"):
        question = item["question"]
        gold_answer = str(item.get("answer", ""))
        gold_titles = list(dict.fromkeys(fact[0] for fact in item.get("supporting_facts", []) if fact))
        missing_titles = [title for title in gold_titles if title not in title_to_docid]

        if missing_titles:
            invalid_queries.append({
                "question": question,
                "gold_titles": gold_titles,
                "missing_titles_in_corpus": missing_titles,
                "reason": "Some supporting fact titles are missing from the corpus mapping.",
            })
            continue

        gold_docids = [title_to_docid[title] for title in gold_titles]

        try:
            if args.query_mode == "rag":
                response = query(question, max_iter=args.max_iter)
            else:
                response = naive_rag_query(question)

            if not isinstance(response, tuple):
                raise ValueError(f"Unexpected query response type: {type(response)}")

            if len(response) == 3:
                pred_answer, retrieved_results, consumed_token = response
            elif len(response) == 2:
                pred_answer, retrieved_results = response
                consumed_token = 0
            else:
                raise ValueError(f"Unexpected query response length: {len(response)}")
        except Exception as exc:
            invalid_queries.append({
                "question": question,
                "gold_titles": gold_titles,
                "reason": f"RAG pipeline failed: {exc}",
            })
            continue

        pred_answer = str(pred_answer)
        consumed_token = float(consumed_token or 0)
        pred_docids = []
        for res in retrieved_results:
            metadata = getattr(res, "metadata", None) or {}
            docid = metadata.get("docid") or getattr(res, "reference", "")
            if docid:
                pred_docids.append(str(docid))

        gold_set = set(gold_docids)
        first_relevant_rank = None
        for idx, docid in enumerate(pred_docids, start=1):
            if docid in gold_set:
                first_relevant_rank = idx
                break
        mrr = 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank

        retrieval_metrics = {"mrr": mrr}
        retrieval_sums["mrr"] += mrr
        for k in args.top_k_values:
            overlap = gold_set.intersection(set(pred_docids[:k]))
            partial = float(len(overlap) > 0)
            strict = float(len(overlap) == len(gold_set) and len(gold_set) > 0)
            retrieval_metrics[f"partial_hits@{k}"] = partial
            retrieval_metrics[f"strict_hits@{k}"] = strict
            retrieval_sums[f"partial_hits@{k}"] += partial
            retrieval_sums[f"strict_hits@{k}"] += strict

        exact_match = float(normalize_answer(pred_answer) == normalize_answer(gold_answer))
        f1 = compute_f1(pred_answer, gold_answer)
        total_em += exact_match
        total_f1 += f1
        total_tokens += consumed_token

        details.append({
            "question": question,
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "gold_titles": gold_titles,
            "gold_docids": gold_docids,
            "pred_docids": pred_docids,
            "retrieval_metrics": retrieval_metrics,
            "qa_metrics": {
                "exact_match": exact_match,
                "f1": f1,
            },
            "efficiency_metrics": {
                "consumed_token": consumed_token,
            },
        })

        if len(details) <= 3:
            print(f"\n[Debug] Question: {question[:80]}...")
            print(f"   Gold docids: {gold_docids}")
            print(f"   Pred docids: {pred_docids[:10]}")
            print(f"   Gold answer: {gold_answer}")
            print(f"   Pred answer: {pred_answer}")
            print(f"   Tokens: {consumed_token}")

    total_valid = len(details)
    summary = {
        "total_valid_queries": total_valid,
        "retrieval": {},
        "downstream_qa": {},
        "efficiency": {},
    }

    if total_valid > 0:
        for k in args.top_k_values:
            summary["retrieval"][f"partial_hits@{k}"] = retrieval_sums[f"partial_hits@{k}"] / total_valid
            summary["retrieval"][f"strict_hits@{k}"] = retrieval_sums[f"strict_hits@{k}"] / total_valid
        summary["retrieval"]["mrr"] = retrieval_sums["mrr"] / total_valid
        summary["downstream_qa"]["exact_match"] = total_em / total_valid
        summary["downstream_qa"]["f1"] = total_f1 / total_valid
        summary["efficiency"]["avg_consumed_token"] = total_tokens / total_valid
        summary["efficiency"]["total_consumed_token"] = total_tokens
    else:
        for k in args.top_k_values:
            summary["retrieval"][f"partial_hits@{k}"] = 0.0
            summary["retrieval"][f"strict_hits@{k}"] = 0.0
        summary["retrieval"]["mrr"] = 0.0
        summary["downstream_qa"]["exact_match"] = 0.0
        summary["downstream_qa"]["f1"] = 0.0
        summary["efficiency"]["avg_consumed_token"] = 0.0
        summary["efficiency"]["total_consumed_token"] = 0.0

    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Valid queries: {total_valid}")
    print("\nRetrieval Metrics")
    for k in args.top_k_values:
        print(f"  Partial Hits@{k}: {summary['retrieval'][f'partial_hits@{k}'] * 100:.2f}%")
        print(f"  Strict Hits@{k}:  {summary['retrieval'][f'strict_hits@{k}'] * 100:.2f}%")
    print(f"  MRR: {summary['retrieval']['mrr']:.4f}")
    print("\nDownstream QA Metrics")
    print(f"  Exact Match: {summary['downstream_qa']['exact_match'] * 100:.2f}%")
    print(f"  F1:          {summary['downstream_qa']['f1'] * 100:.2f}%")
    print("\nEfficiency Metrics")
    print(f"  Avg consumed tokens:   {summary['efficiency']['avg_consumed_token']:.2f}")
    print(f"  Total consumed tokens: {summary['efficiency']['total_consumed_token']:.2f}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "dev_file": args.dev_file,
                "corpus_file": args.corpus_file,
                "query_mode": args.query_mode,
                "max_iter": args.max_iter,
                "max_samples": args.max_samples,
                "top_k_values": list(args.top_k_values),
                "invalid_queries": len(invalid_queries),
            },
            "summary": summary,
            "details": details,
        }, f, ensure_ascii=False, indent=4)

    with open(args.invalid_log_file, "w", encoding="utf-8") as f:
        json.dump(invalid_queries, f, ensure_ascii=False, indent=4)

    print(f"Saved evaluation results to: {args.output_file}")
    print(f"Saved invalid query log to: {args.invalid_log_file}")


def main():
    args = parse_args()
    setup_pipeline(args.corpus_file, reload_data=args.reload_data)
    evaluate(args)


if __name__ == "__main__":
    main()
