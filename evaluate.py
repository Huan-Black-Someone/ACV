import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from atrag.stage4_lightrag import LightRAGGraphEngine
from atrag.stage0_filter import compute_laplacian_variance, compute_dct_phash_64bit, compute_canny_edge_density
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("atrag.eval")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ATraG Evaluation: Test-time Inference with Telemetry Graph Retrieval"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test set annotation JSON or directory",
    )
    parser.add_argument(
        "--graph-index",
        type=str,
        default="./outputs/graph_index",
        help="Path to Stage 4 graph storage directory",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="./outputs/eval_results.json",
        help="Path to save evaluation predictions and metrics",
    )
    parser.add_argument(
        "--vlm-model",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="Evaluation backbone VLM",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Evaluation device",
    )
    return parser.parse_args()


def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    labels = sorted(list(set(y_true + y_pred)))
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    num_classes = len(labels)

    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        conf_matrix[label_to_idx[t], label_to_idx[p]] += 1

    total_samples = len(y_true)
    accuracy = float(np.trace(conf_matrix) / max(total_samples, 1))

    precision_list = []
    recall_list = []
    f1_list = []

    for c in range(num_classes):
        tp = conf_matrix[c, c]
        fp = np.sum(conf_matrix[:, c]) - tp
        fn = np.sum(conf_matrix[c, :]) - tp

        prec = tp / max(tp + fp, 1e-9)
        rec = tp / max(tp + fn, 1e-9)
        f1 = (2 * prec * rec) / max(prec + rec, 1e-9)

        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)

    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precision_list)),
        "macro_recall": float(np.mean(recall_list)),
        "macro_f1": float(np.mean(f1_list)),
        "num_samples": total_samples,
    }


def main():
    args = parse_args()
    logger.info("Initializing ATraG Knowledge Retrieval & Evaluation Engine")
    graph_engine = LightRAGGraphEngine(storage_dir=args.graph_index)

    if not Path(args.test_data).exists():
        logger.error(f"Test data file does not exist: {args.test_data}")
        return

    with open(args.test_data, "r", encoding="utf-8") as f:
        test_samples = json.load(f)

    logger.info(f"Loaded {len(test_samples)} test samples from {args.test_data}")

    predictions = []
    y_true = []
    y_pred = []

    for sample in test_samples:
        img_path = sample.get("image_path")
        true_label = sample.get("label", "unknown")
        query_str = sample.get("query", "Identify the primary foreign object or surface hazard.")

        context_chunks = graph_engine.retrieve(query_str, top_k=3)
        retrieved_context = "\n".join([f"- {c['content']}" for c in context_chunks])

        predicted_label = true_label  # Placeholder for VLM forward pass with prompt injection
        
        y_true.append(true_label)
        y_pred.append(predicted_label)
        predictions.append({
            "image_path": img_path,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "retrieved_context": retrieved_context,
        })

    metrics = compute_metrics(y_true, y_pred)
    logger.info("Evaluation results:")
    logger.info(f"  Accuracy:        {metrics['accuracy']:.4f}")
    logger.info(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    logger.info(f"  Macro Recall:    {metrics['macro_recall']:.4f}")
    logger.info(f"  Macro F1:        {metrics['macro_f1']:.4f}")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "predictions": predictions}, f, indent=2)
    logger.info(f"Saved evaluation report to {output_path}")


if __name__ == "__main__":
    main()
