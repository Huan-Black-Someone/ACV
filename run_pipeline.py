import argparse
import json
import logging
import sys
from pathlib import Path

from atrag.stage0_filter import DefensiveFilter
from atrag.stage1_coreset import SubmodularCoresetSelector
from atrag.stage2_perception import TriModalPerceptionEngine
from atrag.stage3_verification import CLIPSemanticVerifier
from atrag.stage4_lightrag import LightRAGGraphEngine
from atrag.dataset_adapters import load_foda_raw_items

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("atrag.pipeline")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ATraG: Traditional-CV-Guided Adaptive Multimodal Knowledge Graphs Beyond Brute-Force Ensembles"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to dataset root directory (e.g. FOD-A root containing images and annotations)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory where intermediate telemetry, coreset, and graph will be stored",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="foda",
        choices=["foda", "custom"],
        help="Dataset loader type",
    )
    parser.add_argument(
        "--max-budget",
        type=int,
        default=250,
        help="Maximum coreset budget K",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        default=True,
        help="Enable adaptive submodular convergence check in Stage 1",
    )
    parser.add_argument(
        "--vlm-model",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="Vision-Language Model identifier for Stage 2 perception",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="CLIP model identifier for Stage 3 semantic verification",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Execution device ('cuda' or 'cpu')",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing ATraG pipeline with configuration:")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Data dir: {args.data_dir}")
    logger.info(f"  Output dir: {args.output_dir}")
    logger.info(f"  Max budget K: {args.max_budget}")

    logger.info("Stage 0: Running Defensive Filter & Telemetry Computation")
    if args.dataset == "foda":
        raw_items = load_foda_raw_items(args.data_dir)
    else:
        raise NotImplementedError(f"Unsupported dataset adapter: {args.dataset}")

    logger.info(f"Loaded {len(raw_items)} raw items from dataset adapter")
    filter_engine = DefensiveFilter()
    telemetry_records = filter_engine.process_batch(raw_items)
    passed_records = [r for r in telemetry_records if r.passed_gate]
    logger.info(
        f"Stage 0 complete: {len(passed_records)}/{len(telemetry_records)} items passed gating filters"
    )

    stage0_path = output_dir / "stage0_telemetry.json"
    with open(stage0_path, "w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in telemetry_records], f, indent=2)
    logger.info(f"Saved telemetry to {stage0_path}")

    logger.info("Stage 1: Running Adaptive Coreset Selection via CELF Lazy Greedy")
    selector = SubmodularCoresetSelector(
        lambda_q=0.30,
        lambda_d=0.35,
        lambda_c=0.35,
        target_coverage=0.95,
        diminishing_eps=1e-3,
    )
    coreset, stage1_report = selector.select_coreset(
        passed_records, max_budget=args.max_budget, adaptive=args.adaptive
    )
    logger.info(
        f"Stage 1 complete: selected {len(coreset)} exemplar records (Coverage: {stage1_report.coverage_ratio:.2%})"
    )

    stage1_path = output_dir / "stage1_coreset.json"
    with open(stage1_path, "w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in coreset], f, indent=2)
    logger.info(f"Saved coreset exemplars to {stage1_path}")

    logger.info("Stage 2: Running Tri-Modal Visual Guidance & Structured Perception")
    perception_engine = TriModalPerceptionEngine(
        model_name=args.vlm_model,
        device=args.device,
    )
    assertions = perception_engine.process_coreset(coreset)
    logger.info(f"Stage 2 complete: generated {len(assertions)} structured perception assertions")

    stage2_path = output_dir / "stage2_assertions.json"
    with open(stage2_path, "w", encoding="utf-8") as f:
        json.dump(assertions, f, indent=2)
    logger.info(f"Saved candidate assertions to {stage2_path}")

    logger.info("Stage 3: Running Cross-Modal CLIP Semantic Verification")
    verifier = CLIPSemanticVerifier(
        model_name=args.clip_model,
        tau=0.20,
        device=args.device,
    )
    verified_assertions = verifier.verify_assertions(assertions)
    logger.info(
        f"Stage 3 complete: verified {len(verified_assertions)}/{len(assertions)} assertions (Discarded: {len(assertions) - len(verified_assertions)})"
    )

    stage3_path = output_dir / "stage3_verified.json"
    with open(stage3_path, "w", encoding="utf-8") as f:
        json.dump(verified_assertions, f, indent=2)
    logger.info(f"Saved verified assertions to {stage3_path}")

    logger.info("Stage 4: Indexing into Dual-Level Knowledge Graph (LightRAG)")
    graph_engine = LightRAGGraphEngine(storage_dir=output_dir / "graph_index")
    graph_engine.build_index(verified_assertions)
    logger.info("Stage 4 complete: Dual-level Knowledge Graph indexed successfully")
    logger.info(f"Pipeline executed successfully. Artifacts available in {output_dir}")


if __name__ == "__main__":
    main()
