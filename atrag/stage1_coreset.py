import heapq
import math
from typing import Sequence, List, Dict, Tuple, Optional
import numpy as np

from .stage0_filter import TelemetryRecord, phash_hamming_similarity


class SubmodularCoresetSelector:
    def __init__(
        self,
        lambda_quality: float = 0.25,
        lambda_diversity: float = 0.40,
        lambda_coverage: float = 0.35,
        saturation_alpha: float = 2.0,
    ):
        self.lambda_q = lambda_quality
        self.lambda_d = lambda_diversity
        self.lambda_c = lambda_coverage
        self.alpha = saturation_alpha

    @staticmethod
    def _extract_evidence_keys(rec: TelemetryRecord) -> List[str]:
        xmin, ymin, xmax, ymax = rec.bbox
        area_ratio = ((xmax - xmin) * (ymax - ymin)) / max(1.0, float(rec.width * rec.height))
        if area_ratio < 0.02:
            scale_tag = "scale:tiny"
        elif area_ratio < 0.10:
            scale_tag = "scale:medium"
        else:
            scale_tag = "scale:large"

        return [
            f"class:{rec.label}",
            f"class_weather:{rec.label}_{rec.weather}",
            f"class_light:{rec.label}_{rec.light}",
            f"class_scale:{rec.label}_{scale_tag}",
            f"env:{rec.weather}_{rec.light}_{scale_tag}",
        ]

    @staticmethod
    def _build_similarity_matrix(records: Sequence[TelemetryRecord]) -> np.ndarray:
        n = len(records)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float32)

        feats = np.zeros((n, 4), dtype=np.float64)
        for idx, r in enumerate(records):
            feats[idx, 0] = math.log1p(max(0.0, r.q_laplacian_var))
            feats[idx, 1] = r.b_mean_brightness
            feats[idx, 2] = r.c_rms_contrast
            feats[idx, 3] = r.e_canny_edge_density

        mean = feats.mean(axis=0, keepdims=True)
        std = feats.std(axis=0, keepdims=True) + 1e-8
        norm_feats = (feats - mean) / std

        sq_dists = np.sum((norm_feats[:, None, :] - norm_feats[None, :, :]) ** 2, axis=-1)
        rbf_sim = np.exp(-0.25 * sq_dists).astype(np.float32)

        labels = [r.label for r in records]
        phashes = [r.d_dct_phash_hex for r in records]
        sim_mat = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            sim_mat[i, i] = 1.0
            for j in range(i + 1, n):
                p_sim = phash_hamming_similarity(phashes[i], phashes[j])
                cls_bonus = 1.0 if labels[i] == labels[j] else 0.25
                val = float(cls_bonus * (0.65 * rbf_sim[i, j] + 0.35 * p_sim))
                sim_mat[i, j] = val
                sim_mat[j, i] = val
        return sim_mat

    def select(self, candidates: Sequence[TelemetryRecord], budget_k: Optional[int] = None) -> Tuple[List[TelemetryRecord], Dict]:
        if budget_k is None or budget_k <= 0:
            return self.select_adaptive(candidates)

        valid_pool = [r for r in candidates if r.passed_gate]
        n = len(valid_pool)
        if n == 0:
            return [], {"selected_count": 0, "total_valid_pool": n}

        k_target = min(budget_k, n)
        sim_matrix = self._build_similarity_matrix(valid_pool)

        item_evidences = [self._extract_evidence_keys(r) for r in valid_pool]
        all_evidence_keys = sorted({ev for ev_list in item_evidences for ev in ev_list})
        ev_to_idx = {ev: i for i, ev in enumerate(all_evidence_keys)}
        num_ev = max(1, len(all_evidence_keys))

        item_ev_indices = [[ev_to_idx[ev] for ev in ev_list] for ev_list in item_evidences]
        q_scores = np.array([r.composite_quality_score for r in valid_pool], dtype=np.float64)

        current_max_sim = np.zeros(n, dtype=np.float64)
        current_ev_counts = np.zeros(num_ev, dtype=np.float64)
        selected_mask = np.zeros(n, dtype=bool)

        def marginal_gain(idx: int) -> float:
            dq = q_scores[idx] / float(k_target)
            diff_sim = np.maximum(0.0, sim_matrix[:, idx] - current_max_sim)
            dd = float(np.sum(diff_sim)) / float(n)
            ev_idxs = item_ev_indices[idx]
            old_cov = np.log1p(self.alpha * current_ev_counts[ev_idxs])
            new_cov = np.log1p(self.alpha * (current_ev_counts[ev_idxs] + 1.0))
            dc = float(np.sum(new_cov - old_cov)) / float(num_ev)
            return float(self.lambda_q * dq + self.lambda_d * dd + self.lambda_c * dc)

        heap = []
        for idx in range(n):
            gain = marginal_gain(idx)
            heapq.heappush(heap, (-gain, 0, idx))

        selected_indices = []
        trajectory = []
        total_f_val = 0.0
        lazy_evals = 0

        for step in range(1, k_target + 1):
            while heap:
                neg_gain, last_updated_step, best_idx = heapq.heappop(heap)
                if selected_mask[best_idx]:
                    continue
                if last_updated_step == step - 1:
                    true_gain = -neg_gain
                    selected_mask[best_idx] = True
                    selected_indices.append(best_idx)
                    current_max_sim = np.maximum(current_max_sim, sim_matrix[:, best_idx])
                    for ev_i in item_ev_indices[best_idx]:
                        current_ev_counts[ev_i] += 1.0
                    total_f_val += true_gain
                    trajectory.append({
                        "step": step,
                        "image_id": valid_pool[best_idx].image_id,
                        "label": valid_pool[best_idx].label,
                        "marginal_gain": round(true_gain, 6),
                        "cumulative_objective": round(total_f_val, 6),
                    })
                    break
                else:
                    lazy_evals += 1
                    recomputed_gain = marginal_gain(best_idx)
                    heapq.heappush(heap, (-recomputed_gain, step - 1, best_idx))

        selected_records = [valid_pool[i] for i in selected_indices]
        covered_classes = sorted({r.label for r in selected_records})
        stats = {
            "total_candidates": len(candidates),
            "total_valid_pool": n,
            "budget_k": k_target,
            "selected_count": len(selected_records),
            "covered_classes_count": len(covered_classes),
            "final_objective_F": round(total_f_val, 6),
            "lazy_evaluations_count": lazy_evals,
            "naive_evaluations_saved": max(0, (n * k_target) - (n + lazy_evals)),
            "trajectory_head": trajectory[:10],
        }
        return selected_records, stats

    def select_adaptive(
        self,
        candidates: Sequence[TelemetryRecord],
        max_ceiling_k: int = 200,
        epsilon_gain: float = 1e-4,
        target_coverage_ratio: float = 0.95,
    ) -> Tuple[List[TelemetryRecord], Dict]:
        valid_pool = [r for r in candidates if r.passed_gate]
        n = len(valid_pool)
        if n == 0:
            return [], {
                "total_candidates": len(candidates),
                "total_valid_pool": 0,
                "adaptive_k": 0,
                "selected_count": 0,
                "covered_classes_count": 0,
                "stopping_reason": "Empty Pool",
                "final_objective_F": 0.0,
                "evidence_coverage_ratio": 0.0,
            }

        sim_matrix = self._build_similarity_matrix(valid_pool)
        item_evidences = [self._extract_evidence_keys(r) for r in valid_pool]
        all_evidence_keys = sorted({ev for ev_list in item_evidences for ev in ev_list})
        ev_to_idx = {ev: i for i, ev in enumerate(all_evidence_keys)}
        num_ev = max(1, len(all_evidence_keys))

        item_ev_indices = [[ev_to_idx[ev] for ev in ev_list] for ev_list in item_evidences]
        q_scores = np.array([r.composite_quality_score for r in valid_pool], dtype=np.float64)

        current_max_sim = np.zeros(n, dtype=np.float64)
        current_ev_counts = np.zeros(num_ev, dtype=np.float64)
        selected_mask = np.zeros(n, dtype=bool)

        def marginal_gain(idx: int) -> float:
            dq = q_scores[idx] / float(max_ceiling_k)
            diff_sim = np.maximum(0.0, sim_matrix[:, idx] - current_max_sim)
            dd = float(np.sum(diff_sim)) / float(n)
            ev_idxs = item_ev_indices[idx]
            old_cov = np.log1p(self.alpha * current_ev_counts[ev_idxs])
            new_cov = np.log1p(self.alpha * (current_ev_counts[ev_idxs] + 1.0))
            dc = float(np.sum(new_cov - old_cov)) / float(num_ev)
            return float(self.lambda_q * dq + self.lambda_d * dd + self.lambda_c * dc)

        heap = []
        for idx in range(n):
            gain = marginal_gain(idx)
            heapq.heappush(heap, (-gain, 0, idx))

        selected_indices = []
        trajectory = []
        total_f_val = 0.0
        lazy_evals = 0
        stopping_reason = "Reached Ceiling K"

        step = 0
        while step < max_ceiling_k and heap:
            step += 1
            best_gain_found = False

            while heap:
                neg_gain, last_updated_step, best_idx = heapq.heappop(heap)
                if selected_mask[best_idx]:
                    continue

                if last_updated_step == step - 1:
                    true_gain = -neg_gain
                    if true_gain < epsilon_gain:
                        stopping_reason = f"Marginal Gain Saturation (gain={true_gain:.6f} < epsilon={epsilon_gain})"
                        best_gain_found = False
                        break

                    selected_mask[best_idx] = True
                    selected_indices.append(best_idx)
                    current_max_sim = np.maximum(current_max_sim, sim_matrix[:, best_idx])
                    for ev_i in item_ev_indices[best_idx]:
                        current_ev_counts[ev_i] += 1.0
                    total_f_val += true_gain

                    trajectory.append({
                        "step": step,
                        "image_id": valid_pool[best_idx].image_id,
                        "label": valid_pool[best_idx].label,
                        "marginal_gain": round(true_gain, 6),
                        "cumulative_objective": round(total_f_val, 6),
                    })
                    best_gain_found = True
                    break
                else:
                    lazy_evals += 1
                    recomputed_gain = marginal_gain(best_idx)
                    heapq.heappush(heap, (-recomputed_gain, step - 1, best_idx))

            if not best_gain_found:
                break

            covered_evidence_ratio = np.count_nonzero(current_ev_counts > 0) / float(num_ev)
            if covered_evidence_ratio >= target_coverage_ratio:
                stopping_reason = f"Evidence Coverage Target Reached ({covered_evidence_ratio*100:.1f}% >= {target_coverage_ratio*100:.1f}%)"
                break

        selected_records = [valid_pool[i] for i in selected_indices]
        covered_classes = sorted({r.label for r in selected_records})
        stats = {
            "total_candidates": len(candidates),
            "total_valid_pool": n,
            "adaptive_k": len(selected_records),
            "budget_k": len(selected_records),
            "selected_count": len(selected_records),
            "covered_classes_count": len(covered_classes),
            "stopping_reason": stopping_reason,
            "final_objective_F": round(total_f_val, 6),
            "evidence_coverage_ratio": round(float(np.count_nonzero(current_ev_counts > 0) / float(num_ev)), 4),
            "lazy_evaluations_count": lazy_evals,
            "naive_evaluations_saved": max(0, (n * len(selected_records)) - (n + lazy_evals)),
            "trajectory_head": trajectory[:10],
        }
        return selected_records, stats

    def evaluate_subset_objective(self, candidates: Sequence[TelemetryRecord], subset_ids: Sequence[str]) -> float:
        valid_pool = [r for r in candidates if r.passed_gate]
        n = len(valid_pool)
        id_to_idx = {r.image_id: i for i, r in enumerate(valid_pool)}
        chosen = [id_to_idx[s] for s in subset_ids if s in id_to_idx]
        if n == 0 or not chosen:
            return 0.0

        k = len(chosen)
        sim_matrix = self._build_similarity_matrix(valid_pool)
        item_evidences = [self._extract_evidence_keys(r) for r in valid_pool]
        all_evidence_keys = sorted({ev for ev_list in item_evidences for ev in ev_list})
        ev_to_idx = {ev: i for i, ev in enumerate(all_evidence_keys)}
        num_ev = max(1, len(all_evidence_keys))

        q_scores = np.array([r.composite_quality_score for r in valid_pool], dtype=np.float64)
        q_val = float(np.sum(q_scores[chosen])) / float(k)
        sub_sim = sim_matrix[:, chosen]
        d_val = float(np.mean(np.max(sub_sim, axis=1)))

        ev_counts = np.zeros(num_ev, dtype=np.float64)
        for idx in chosen:
            for ev in item_evidences[idx]:
                ev_counts[ev_to_idx[ev]] += 1.0
        c_val = float(np.mean(np.log1p(self.alpha * ev_counts)))

        return float(self.lambda_q * q_val + self.lambda_d * d_val + self.lambda_c * c_val)
