# ATraG: Traditional-CV-Guided Adaptive Multimodal Knowledge Graphs Beyond Brute-Force Ensembles

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)

Official implementation of **ATraG** (*Traditional-CV-Guided Adaptive Multimodal Knowledge Graphs Beyond Brute-Force Ensembles*). ATraG addresses catastrophic hallucination, visual feature degeneration, and knowledge pollution in Vision-Language Models (VLMs) deployed in safety-critical, high-uncertainty operational environments without relying on computationally prohibitive brute-force model ensembles.

---

## Architecture Overview

```
[Raw Input Stream]
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Stage 0: 3-Tier Defensive Filter & Telemetry Sensor    │
│  - Gate 1: Spatial Dimension (H, W >= 32px)            │
│  - Gate 2: Sharpness Gating (Var(Laplacian) >= s_var)   │
│  - Gate 3: Perceptual Telemetry: z = [q, b, c, e, d]   │
└──────────────────────────┬─────────────────────────────┘
                           │ Candidate Pool V
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: Adaptive Submodular Coreset Selection         │
│  - Submodular Maximization: F(S) = λ_q Q + λ_d D + λ_c C│
│  - CELF Lazy Greedy Acceleration (Max-Heap)            │
│  - Adaptive Convergence Gating (Coverage >= 95%, ΔF<ε) │
└──────────────────────────┬─────────────────────────────┘
                           │ Optimal Coreset S*
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 2: Tri-Modal Visual Guidance Perception          │
│  - Modality 1: Salient ROI Spatial Cropping            │
│  - Modality 2: CLAHE Contrast-Enhanced Channel         │
│  - Modality 3: Bilateral Filtered Boundary Topology    │
│  - Joint Feature Fusion + Structured Reasoning Prompt  │
└──────────────────────────┬─────────────────────────────┘
                           │ Candidate Assertions
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 3: Cross-Modal Semantic Verification             │
│  - Zero-Shot CLIP Verification: sim_CLIP(I, T) >= 0.20 │
│  - Rejection of Hallucinated / Contradictory Tokens    │
└──────────────────────────┬─────────────────────────────┘
                           │ Evidence-Grounded Assertions
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 4: Dual-Level Knowledge Graph Indexing (LightRAG)│
│  - High-Level Subgraph Relations & Semantic Taxonomies │
│  - Low-Level Entity & Factual Graph Structure          │
│  - SHA-256 Provenance Anchor Binding Φ                 │
└──────────────────────────┬─────────────────────────────┘
                           │ Indexed Multimodal Knowledge Base
                           ▼
                 [Test-Time Inference]
      Online Retrieval via Telemetry -> Prompt Injection -> Accurate Prediction
```

---

## Key Algorithmic Stages

1. **Stage 0: 3-Tier Defensive Filtering & Telemetry Computation**
   Filters degenerate artifacts (extreme blur, chromatic collapse, out-of-distribution frames) and computes a 5-dimensional perceptual telemetry signature $z_i = [q_i, b_i, c_i, e_i, d_i]$ for each candidate frame.
2. **Stage 1: Adaptive Submodular Coreset Selection**
   Formulates exemplar distillation as submodular set function maximization $F(S) = \lambda_q Q(S) + \lambda_d D(S) + \lambda_c C(S)$, accelerated by Cost-Effective Lazy Forward (CELF) greedy optimization with adaptive convergence termination.
3. **Stage 2: Tri-Modal Visual Guidance & Structured Perception**
   Synthesizes a visual composite consisting of Salient Region-of-Interest (ROI), CLAHE contrast-enhanced illumination, and Bilateral Canny boundary topology, guiding the VLM backbone through a structured schema.
4. **Stage 3: Cross-Modal CLIP Semantic Verification**
   Calculates mutual cross-modal alignment scores between ROI visual embeddings and extracted assertions, discarding hallucinations below calibrated threshold $\tau = 0.20$.
5. **Stage 4: Dual-Level LightRAG Indexing & Provenance Anchor Binding**
   Organizes verified factual nodes into entity-level and relationship-level knowledge graphs, cryptographically bound to operational telemetry via SHA-256 provenance anchors $\Phi$.

---

## Installation

```bash
# Clone repository
git clone https://github.com/Huan-Black-Someone/ACV.git
cd ACV

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

---

## Quickstart

### 1. End-to-End Pipeline Execution
Run all 5 stages from raw dataset to indexed knowledge graph:

```bash
python run_pipeline.py \
    --data-dir /path/to/dataset \
    --output-dir ./outputs \
    --dataset foda \
    --max-budget 250 \
    --adaptive
```

### 2. Test-time Inference & Evaluation
Evaluate predictions on target test samples with knowledge graph retrieval:

```bash
python evaluate.py \
    --test-data /path/to/test_annotations.json \
    --graph-index ./outputs/graph_index \
    --output-file ./outputs/eval_results.json
```

---

## Project Structure

```
ACV/
├── atrag/
│   ├── __init__.py              # Package entrypoint
│   ├── stage0_filter.py         # Defensive filtering & telemetry extraction
│   ├── stage1_coreset.py        # Submodular CELF coreset selection
│   ├── stage2_perception.py     # Tri-modal guidance & VLM structured perception
│   ├── stage3_verification.py   # Cross-modal CLIP verification
│   ├── stage4_lightrag.py       # Dual-level LightRAG graph indexing & provenance
│   └── dataset_adapters.py      # Dataset loaders (FOD-A, custom formats)
├── run_pipeline.py              # CLI entrypoint for complete pipeline
├── evaluate.py                  # Evaluation and metrics benchmarking
├── requirements.txt             # Required Python dependencies
├── setup.py                     # Package setup script
├── .gitignore                   # Ignored directories and build artifacts
└── README.md                    # Project documentation
```

---

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
