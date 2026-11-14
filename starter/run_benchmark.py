"""Compare DAY07 retrievers and optionally run the agent on the same split."""
import argparse
import json
import os
from contextlib import closing
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from pymilvus import MilvusClient

from .agentic import run_agent
from .baseline import hit, pack_context
from .live_agent import LexicalRetriever, MilvusRetriever, OllamaController, RerankRetriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--method", action="append", choices=("dense", "lexical", "rerank"))
    parser.add_argument("--agent", action="store_true")
    parser.add_argument("--questions", type=Path, default=Path("data/vla_benchmark.json"))
    parser.add_argument("--output-root", type=Path, default=Path("starter/results"))
    args = parser.parse_args()
    cases = [case for case in json.loads(args.questions.read_text(encoding="utf-8"))
             if case["split"] == args.split]
    if not cases:
        parser.error("No questions for the requested split.")
    load_dotenv()
    url = os.environ["OLLAMA_URL"]
    collection = os.environ["MILVUS_COLLECTION"]
    chat = os.environ["OLLAMA_CHAT_MODEL"]
    embedding = os.environ["OLLAMA_EMBED_MODEL"]
    controller = OllamaController(url, chat)
    budget = 3200
    mode = "agent" if args.agent else "retrieval"
    with closing(MilvusClient(os.environ["MILVUS_URI"], timeout=30)) as client:
        client.load_collection(collection)
        dense = MilvusRetriever(client, collection, url, embedding)
        for method in dict.fromkeys(args.method or ("dense", "lexical", "rerank")):
            retriever = dense
            if method == "lexical":
                retriever = LexicalRetriever(client, collection)
            elif method == "rerank":
                retriever = RerankRetriever(dense, url, chat)
            output = args.output_root / f"{mode}-{args.split}-{method}"
            output.mkdir(parents=True, exist_ok=True)
            traces = []
            with (output / "traces.jsonl").open("x", encoding="utf-8") as stream:
                for case in cases:
                    question, filters = case["question"], case.get("filter")
                    started = perf_counter()
                    try:
                        if args.agent:
                            result = run_agent(question, controller, retriever, filters=filters,
                                               top_k=3, budget=budget, max_searches=2)
                            searches = [event for event in result["events"] if event["event"] == "search"]
                            if searches:
                                final = searches[-1]
                                gold = case.get("gold", [])
                                context_ids = set(final["context_ids"])
                                result["context_evidence"] = any(
                                    row["id"] in context_ids and hit(row, gold)
                                    for event in searches for row in event["results"]
                                )
                            else:
                                result["context_evidence"] = False
                        else:
                            ranked = retriever.search(question, filters=filters, top_k=3)
                            context, ids = pack_context(ranked, budget=budget)
                            gold = case.get("gold", [])
                            result = {
                                "status": "ok", "top3": ranked, "context": context, "context_ids": ids,
                                "hit_at_1": any(hit(row, gold) for row in ranked[:1]),
                                "evidence_hit_at_3": any(hit(row, gold) for row in ranked),
                                "context_evidence": any(hit(row, gold) for row in ranked if row["id"] in ids),
                            }
                            if isinstance(retriever, RerankRetriever):
                                result["ranking"] = retriever.last_ranking
                    except Exception as exc:
                        result = {"status": "error", "error": str(exc)}
                    trace = {"query_id": case["id"], "question": question, "filter": filters,
                             "answerable": case["answerable"], "elapsed_seconds": perf_counter() - started,
                             **result}
                    traces.append(trace)
                    stream.write(json.dumps(trace, ensure_ascii=False) + "\n")
                    stream.flush()
                    print(method, case["id"], trace["status"], flush=True)
            eligible = [trace for trace in traces if trace["answerable"] and trace["status"] != "error"]
            summary = {"method": method, "split": args.split, "mode": mode, "collection": collection,
                       "embedding_model": embedding, "chat_model": chat if args.agent or method == "rerank" else None,
                       "context_budget_chars": budget, "top_k": 3, "api_cost": 0,
                       "elapsed_seconds": sum(trace["elapsed_seconds"] for trace in traces),
                       "errors": sum(trace["status"] == "error" for trace in traces), "cases": len(traces)}
            metrics = ("context_evidence",) if args.agent else (
                "hit_at_1", "evidence_hit_at_3", "context_evidence",
            )
            for metric in metrics:
                summary[metric] = {"n": sum(trace[metric] for trace in eligible), "N": len(eligible)}
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
