import argparse
import json
import logging
import os
import re
import string
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

from deepsearcher.configuration import Configuration, init_config
from deepsearcher.offline_loading import load_from_local_files
from deepsearcher.online_query import naive_rag_query, query


LOGGER = logging.getLogger(__name__)
DEFAULT_TOP_K_VALUES = (1, 5, 10)


def setup_environment() -> Tuple[str, str, str]:
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
    vllm_file_handler = logging.FileHandler("vllm_server_log.log", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    vllm_file_handler.setFormatter(formatter)
    vllm_logger.setLevel(logging.INFO)
    vllm_logger.addHandler(vllm_file_handler)

    return base_url, api_key, model_name


def build_paths() -> Dict[str, str]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace = os.path.dirname(current_dir)
    output_root = os.path.join(workspace, "output")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_root, timestamp)
    return {
        "workspace": workspace,
        "output_root": output_root,
        "output_dir": output_dir,
        "corpus_path": os.path.join(workspace, "data", "corpus_with_ids.jsonl"),
        "dev_json_path": os.path.join(workspace, "data", "2wiki", "dev.json"),
        "output_file": os.path.join(output_dir, "evaluation_results.json"),
        "invalid_log_file": os.path.join(output_dir, "invalid_queries_log.json"),
    }


def parse_top_k_values(raw_value: str) -> Tuple[int, ...]:
    values: List[int] = []
    for part in raw_value.split(","):
        cleaned = part.strip()
        if not cleaned:
            continue
        value = int(cleaned)
        if value <= 0:
            raise ValueError("top-k values must be positive integers.")
        values.append(value)

    if not values:
        raise ValueError("At least one top-k value must be provided.")

    return tuple(dict.fromkeys(values))


def setup_retrieval_pipeline(corpus_path: str, reload_data: bool = False) -> None:
    base_url, api_key, model_name = setup_environment()

    config = Configuration()
    config.set_provider_config(
        "llm",
        "OpenAI",
        {
            "model": model_name,
            "base_url": base_url,
            "api_key": api_key,
        },
    )
    config.set_provider_config(
        "embedding",
        "SentenceTransformerEmbedding",
        {
            "model": "BAAI/bge-large-en-v1.5",
            "batch_size": 256,
        },
    )
    config.set_provider_config(
        "file_loader",
        "JsonFileLoader",
        {
            "text_key": "text",
            "id_key": "docid",
        },
    )

    init_config(config=config)
    if reload_data:
        print(f"Reloading corpus into retrieval store from: {corpus_path}")
        load_from_local_files(
            paths_or_directory=corpus_path,
            collection_name="wiki",
            collection_description="2wiki corpus",
        )
    else:
        print("Skipping corpus reload. Using existing retrieval store state.")


def load_title_to_docid_mapping(corpus_file: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            title = doc.get("title")
            docid = doc.get("docid")
            if title and docid is not None:
                mapping[str(title)] = str(docid)
    return mapping


def load_dev_data(dev_file: str, max_samples: Optional[int]) -> List[Dict[str, Any]]:
    with open(dev_file, "r", encoding="utf-8") as f:
        dev_data = json.load(f)
    if max_samples:
        return dev_data[:max_samples]
    return dev_data


def extract_gold_titles(item: Dict[str, Any]) -> List[str]:
    titles = []
    for fact in item.get("supporting_facts", []):
        if fact and len(fact) > 0:
            titles.append(str(fact[0]))
    return list(dict.fromkeys(titles))


def extract_pred_docids(retrieved_results: Sequence[Any]) -> List[str]:
    pred_docids: List[str] = []
    for res in retrieved_results:
        metadata = getattr(res, "metadata", None) or {}
        docid = metadata.get("docid") or getattr(res, "reference", "")
        if docid:
            pred_docids.append(str(docid))
    return pred_docids


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    def lower(value: str) -> str:
        return value.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def compute_exact_match(prediction: str, gold_answer: str) -> float:
    return float(normalize_answer(str(prediction)) == normalize_answer(str(gold_answer)))


def compute_f1(prediction: str, gold_answer: str) -> float:
    pred_tokens = normalize_answer(str(prediction)).split()
    gold_tokens = normalize_answer(str(gold_answer)).split()

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


def compute_retrieval_metrics(
    gold_docids: Sequence[str],
    pred_docids: Sequence[str],
    top_k_values: Sequence[int],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    gold_set = set(gold_docids)

    first_relevant_rank: Optional[int] = None
    for idx, docid in enumerate(pred_docids, start=1):
        if docid in gold_set:
            first_relevant_rank = idx
            break

    metrics["mrr"] = 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank

    for k in top_k_values:
        top_k_preds = list(pred_docids[:k])
        top_k_set = set(top_k_preds)
        overlap = gold_set.intersection(top_k_set)
        metrics[f"partial_hits@{k}"] = float(len(overlap) > 0)
        metrics[f"strict_hits@{k}"] = float(len(overlap) == len(gold_set) and len(gold_set) > 0)

    return metrics


def unpack_query_response(response: Any) -> Tuple[str, Sequence[Any], int]:
    if not isinstance(response, tuple):
        raise ValueError(f"Unexpected query response type: {type(response)}")

    if len(response) == 3:
        final_answer, retrieved_results, consumed_token = response
        return str(final_answer), retrieved_results, int(safe_float(consumed_token))

    if len(response) == 2:
        final_answer, retrieved_results = response
        return str(final_answer), retrieved_results, 0

    raise ValueError(f"Unexpected query response length: {len(response)}")


def run_query(question: str, query_mode: str, max_iter: int = 1) -> Tuple[str, Sequence[Any], int]:
    normalized_mode = query_mode.strip().lower()
    if normalized_mode == "rag":
        return unpack_query_response(query(question, max_iter=max_iter))
    if normalized_mode in {"naive_rag", "naive-rag", "naiverag"}:
        return unpack_query_response(naive_rag_query(question))
    raise ValueError(f"Unsupported query_mode: {query_mode}")


def build_invalid_entry(
    question: str,
    reason: str,
    gold_titles: Optional[Sequence[str]] = None,
    missing_titles: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"question": question, "reason": reason}
    if gold_titles is not None:
        entry["gold_titles"] = list(gold_titles)
    if missing_titles is not None:
        entry["missing_titles_in_corpus"] = list(missing_titles)
    return entry


def evaluate_single_sample(
    item: Dict[str, Any],
    title_to_docid: Dict[str, str],
    query_mode: str,
    max_iter: int,
    top_k_values: Sequence[int],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    question = item["question"]
    gold_answer = str(item.get("answer", ""))
    gold_titles = extract_gold_titles(item)

    missing_titles = [title for title in gold_titles if title not in title_to_docid]
    if missing_titles:
        invalid_entry = build_invalid_entry(
            question=question,
            reason="Some supporting fact titles are missing from the corpus mapping.",
            gold_titles=gold_titles,
            missing_titles=missing_titles,
        )
        return None, invalid_entry

    gold_docids = [title_to_docid[title] for title in gold_titles]

    try:
        final_answer, retrieved_results, consumed_token = run_query(
            question=question,
            query_mode=query_mode,
            max_iter=max_iter,
        )
    except Exception as exc:
        invalid_entry = build_invalid_entry(
            question=question,
            reason=f"RAG pipeline failed: {exc}",
            gold_titles=gold_titles,
        )
        return None, invalid_entry

    pred_docids = extract_pred_docids(retrieved_results)
    retrieval_metrics = compute_retrieval_metrics(gold_docids, pred_docids, top_k_values)
    exact_match = compute_exact_match(final_answer, gold_answer)
    f1_score = compute_f1(final_answer, gold_answer)

    result = {
        "question": question,
        "gold_answer": gold_answer,
        "pred_answer": final_answer,
        "gold_titles": gold_titles,
        "gold_docids": gold_docids,
        "pred_docids": pred_docids,
        "retrieval_metrics": retrieval_metrics,
        "qa_metrics": {
            "exact_match": exact_match,
            "f1": f1_score,
        },
        "efficiency_metrics": {
            "consumed_token": safe_float(consumed_token),
        },
    }
    return result, None


def aggregate_metrics(
    detailed_results: Sequence[Dict[str, Any]],
    top_k_values: Sequence[int],
) -> Dict[str, Any]:
    total = len(detailed_results)
    summary: Dict[str, Any] = {
        "total_valid_queries": total,
        "retrieval": {},
        "downstream_qa": {},
        "efficiency": {},
    }

    if total == 0:
        for k in top_k_values:
            summary["retrieval"][f"partial_hits@{k}"] = 0.0
            summary["retrieval"][f"strict_hits@{k}"] = 0.0
        summary["retrieval"]["mrr"] = 0.0
        summary["downstream_qa"]["exact_match"] = 0.0
        summary["downstream_qa"]["f1"] = 0.0
        summary["efficiency"]["avg_consumed_token"] = 0.0
        summary["efficiency"]["total_consumed_token"] = 0.0
        return summary

    for k in top_k_values:
        summary["retrieval"][f"partial_hits@{k}"] = sum(
            item["retrieval_metrics"][f"partial_hits@{k}"] for item in detailed_results
        ) / total
        summary["retrieval"][f"strict_hits@{k}"] = sum(
            item["retrieval_metrics"][f"strict_hits@{k}"] for item in detailed_results
        ) / total

    summary["retrieval"]["mrr"] = sum(
        item["retrieval_metrics"]["mrr"] for item in detailed_results
    ) / total

    summary["downstream_qa"]["exact_match"] = sum(
        item["qa_metrics"]["exact_match"] for item in detailed_results
    ) / total
    summary["downstream_qa"]["f1"] = sum(
        item["qa_metrics"]["f1"] for item in detailed_results
    ) / total

    total_tokens = sum(
        item["efficiency_metrics"]["consumed_token"] for item in detailed_results
    )
    summary["efficiency"]["avg_consumed_token"] = total_tokens / total
    summary["efficiency"]["total_consumed_token"] = total_tokens

    return summary


def print_summary(summary: Dict[str, Any], top_k_values: Sequence[int]) -> None:
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Valid queries: {summary['total_valid_queries']}")
    print("\nRetrieval Metrics")
    for k in top_k_values:
        partial = summary["retrieval"][f"partial_hits@{k}"] * 100
        strict = summary["retrieval"][f"strict_hits@{k}"] * 100
        print(f"  Partial Hits@{k}: {partial:.2f}%")
        print(f"  Strict Hits@{k}:  {strict:.2f}%")
    print(f"  MRR: {summary['retrieval']['mrr']:.4f}")

    print("\nDownstream QA Metrics")
    print(f"  Exact Match: {summary['downstream_qa']['exact_match'] * 100:.2f}%")
    print(f"  F1:          {summary['downstream_qa']['f1'] * 100:.2f}%")

    print("\nEfficiency Metrics")
    print(f"  Avg consumed tokens:   {summary['efficiency']['avg_consumed_token']:.2f}")
    print(f"  Total consumed tokens: {summary['efficiency']['total_consumed_token']:.2f}")
    print("=" * 60)


def save_results(
    output_file: str,
    invalid_log_file: str,
    summary: Dict[str, Any],
    detailed_results: Sequence[Dict[str, Any]],
    invalid_queries_log: Sequence[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": metadata,
                "summary": summary,
                "details": detailed_results,
            },
            f,
            ensure_ascii=False,
            indent=4,
        )

    with open(invalid_log_file, "w", encoding="utf-8") as f:
        json.dump(list(invalid_queries_log), f, ensure_ascii=False, indent=4)


def evaluate_dataset(
    dev_file: str,
    corpus_file: str,
    output_file: str,
    invalid_log_file: str,
    max_samples: Optional[int] = 200,
    query_mode: str = "naive_rag",
    max_iter: int = 1,
    top_k_values: Sequence[int] = DEFAULT_TOP_K_VALUES,
) -> Dict[str, Any]:
    print("1. Building title -> docid mapping...")
    title_to_docid = load_title_to_docid_mapping(corpus_file)
    print(f"   - Loaded {len(title_to_docid)} title mappings from corpus")

    print(f"2. Loading dev set from {dev_file} ...")
    dev_data = load_dev_data(dev_file, max_samples=max_samples)
    print(f"   - Loaded {len(dev_data)} evaluation samples")

    detailed_results: List[Dict[str, Any]] = []
    invalid_queries_log: List[Dict[str, Any]] = []

    print("3. Running evaluation...")
    for item in tqdm(dev_data, desc="Evaluating"):
        result, invalid_entry = evaluate_single_sample(
            item=item,
            title_to_docid=title_to_docid,
            query_mode=query_mode,
            max_iter=max_iter,
            top_k_values=top_k_values,
        )

        if invalid_entry is not None:
            invalid_queries_log.append(invalid_entry)
            continue

        if result is not None:
            detailed_results.append(result)
            if len(detailed_results) <= 3:
                print(f"\n[Debug] Question: {result['question'][:80]}...")
                print(f"   Gold docids: {result['gold_docids']}")
                print(f"   Pred docids: {result['pred_docids'][:10]}")
                print(f"   Gold answer: {result['gold_answer']}")
                print(f"   Pred answer: {result['pred_answer']}")
                print(f"   Tokens: {result['efficiency_metrics']['consumed_token']}")

    summary = aggregate_metrics(detailed_results, top_k_values=top_k_values)
    print_summary(summary, top_k_values=top_k_values)

    metadata = {
        "dev_file": dev_file,
        "corpus_file": corpus_file,
        "max_samples": max_samples,
        "query_mode": query_mode,
        "max_iter": max_iter,
        "invalid_queries": len(invalid_queries_log),
    }
    save_results(
        output_file=output_file,
        invalid_log_file=invalid_log_file,
        summary=summary,
        detailed_results=detailed_results,
        invalid_queries_log=invalid_queries_log,
        metadata=metadata,
    )

    print(f"Saved evaluation results to: {output_file}")
    print(f"Saved invalid query log to: {invalid_log_file}")

    return {
        "metadata": metadata,
        "summary": summary,
        "details": detailed_results,
        "invalid_queries": invalid_queries_log,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    paths = build_paths()
    parser = argparse.ArgumentParser(description="Evaluate DeepSearcher retrieval and downstream QA metrics.")
    parser.add_argument(
        "--query-mode",
        choices=["rag", "naive_rag"],
        default="naive_rag",
        help="Choose the query pipeline to evaluate.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1,
        help="Maximum number of iterations for the agentic RAG pipeline.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Number of dev samples to evaluate. Use 0 to evaluate the full dataset.",
    )
    parser.add_argument(
        "--top-k",
        type=str,
        default="1,5,10",
        help="Comma-separated top-k values for retrieval metrics, for example: 1,5,10",
    )
    parser.add_argument(
        "--reload-data",
        action="store_true",
        help="Reload the corpus into the retrieval store before evaluation.",
    )
    parser.add_argument(
        "--dev-file",
        type=str,
        default=paths["dev_json_path"],
        help="Path to the dev dataset JSON file.",
    )
    parser.add_argument(
        "--corpus-file",
        type=str,
        default=paths["corpus_path"],
        help="Path to the corpus JSONL file.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=paths["output_file"],
        help="Path to the evaluation results JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=paths["output_dir"],
        help="Directory for evaluation outputs. Defaults to output/<timestamp>/",
    )
    parser.add_argument(
        "--invalid-log-file",
        type=str,
        default=paths["invalid_log_file"],
        help="Path to the invalid query log JSON file.",
    )
    parser.set_defaults(
        default_output_file=paths["output_file"],
        default_invalid_log_file=paths["invalid_log_file"],
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.max_iter < 1:
        parser.error("--max-iter must be >= 1")
    if args.max_samples < 0:
        parser.error("--max-samples must be >= 0")

    try:
        top_k_values = parse_top_k_values(args.top_k)
    except ValueError as exc:
        parser.error(str(exc))

    max_samples = None if args.max_samples == 0 else args.max_samples
    output_file_overridden = args.output_file != args.default_output_file
    invalid_log_overridden = args.invalid_log_file != args.default_invalid_log_file
    if args.output_dir and not output_file_overridden:
        args.output_file = os.path.join(args.output_dir, "evaluation_results.json")
    if args.output_dir and not invalid_log_overridden:
        args.invalid_log_file = os.path.join(args.output_dir, "invalid_queries_log.json")

    setup_retrieval_pipeline(args.corpus_file, reload_data=args.reload_data)
    evaluate_dataset(
        dev_file=args.dev_file,
        corpus_file=args.corpus_file,
        output_file=args.output_file,
        invalid_log_file=args.invalid_log_file,
        max_samples=max_samples,
        query_mode=args.query_mode,
        max_iter=args.max_iter,
        top_k_values=top_k_values,
    )


if __name__ == "__main__":
    main()
