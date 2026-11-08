import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class LightRAGGraphEngine:
    def __init__(
        self,
        working_dir: Path,
        embedding_dim: int = 768,
        similarity_threshold: float = 0.65,
    ):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold

        self.entities_path = self.working_dir / "vdb_entities.json"
        self.relationships_path = self.working_dir / "vdb_relationships.json"
        self.chunks_path = self.working_dir / "vdb_chunks.json"

        self.entities: Dict[str, Dict[str, Any]] = self._load_store(self.entities_path)
        self.relationships: List[Dict[str, Any]] = self._load_store(self.relationships_path, default=[])
        self.chunks: List[Dict[str, Any]] = self._load_store(self.chunks_path, default=[])

    @staticmethod
    def _load_store(path: Path, default=None):
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return default if default is not None else {}

    def _save_stores(self):
        with open(self.entities_path, "w", encoding="utf-8") as f:
            json.dump(self.entities, f, indent=2, ensure_ascii=False)
        with open(self.relationships_path, "w", encoding="utf-8") as f:
            json.dump(self.relationships, f, indent=2, ensure_ascii=False)
        with open(self.chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)

    @staticmethod
    def compute_provenance_anchor(
        image_id: str,
        bbox: Tuple[int, int, int, int],
        telemetry: Dict[str, Any],
        raw_bytes: Optional[bytes] = None,
    ) -> str:
        raw_signature = f"{image_id}_{bbox}_{telemetry.get('q_laplacian_var', 0.0)}"
        if raw_bytes is not None:
            raw_signature += f"_{len(raw_bytes)}"
        return hashlib.sha256(raw_signature.encode("utf-8")).hexdigest()[:16].upper()

    def insert_fact_chunk(
        self,
        doc_id: str,
        text_chunk: str,
        provenance_anchor: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        chunk_record = {
            "chunk_id": doc_id,
            "anchor_phi": provenance_anchor,
            "content": text_chunk,
            "metadata": metadata or {},
        }
        self.chunks.append(chunk_record)

    def insert_entity(
        self,
        entity_name: str,
        entity_type: str,
        description: str,
        embedding: Optional[List[float]] = None,
        anchor_phi: Optional[str] = None,
    ):
        self.entities[entity_name] = {
            "type": entity_type,
            "description": description,
            "embedding": embedding or [0.0] * self.embedding_dim,
            "anchor_phi": anchor_phi,
        }

    def insert_relationship(
        self,
        src_entity: str,
        tgt_entity: str,
        relation_type: str,
        description: str,
        embedding: Optional[List[float]] = None,
    ):
        rel_record = {
            "source": src_entity,
            "target": tgt_entity,
            "relation": relation_type,
            "description": description,
            "embedding": embedding or [0.0] * self.embedding_dim,
        }
        self.relationships.append(rel_record)

    def persist(self):
        self._save_stores()

    def query_dual_level(
        self,
        query_text: str,
        query_embedding: Optional[List[float]] = None,
        top_entities: int = 5,
        top_relations: int = 5,
    ) -> Dict[str, Any]:
        low_level_hits = []
        high_level_hits = []

        q_terms = set(query_text.lower().split())

        for ent_name, ent_data in self.entities.items():
            desc = ent_data.get("description", "").lower()
            overlap = sum(1 for t in q_terms if t in desc or t in ent_name.lower())
            score = overlap / max(1, len(q_terms))
            if score > 0.1:
                low_level_hits.append({
                    "entity": ent_name,
                    "type": ent_data.get("type"),
                    "description": ent_data.get("description"),
                    "score": round(score, 4),
                    "anchor_phi": ent_data.get("anchor_phi"),
                })
        low_level_hits.sort(key=lambda x: x["score"], reverse=True)
        low_level_hits = low_level_hits[:top_entities]

        for rel in self.relationships:
            desc = f"{rel['source']} {rel['relation']} {rel['target']} {rel.get('description', '')}".lower()
            overlap = sum(1 for t in q_terms if t in desc)
            score = overlap / max(1, len(q_terms))
            if score > 0.1:
                high_level_hits.append({
                    "path": f"({rel['source']}) --[{rel['relation']}]--> ({rel['target']})",
                    "description": rel.get("description"),
                    "score": round(score, 4),
                })
        high_level_hits.sort(key=lambda x: x["score"], reverse=True)
        high_level_hits = high_level_hits[:top_relations]

        return {
            "low_level_entities": low_level_hits,
            "high_level_subgraphs": high_level_hits,
        }

    def format_prompt_context(self, dual_retrieval_result: Dict[str, Any]) -> str:
        entities = dual_retrieval_result.get("low_level_entities", [])
        relations = dual_retrieval_result.get("high_level_subgraphs", [])

        lines = ["[Dual-Level Knowledge Graph Retrieval Context]"]
        if entities:
            lines.append("Low-Level Entity Grounding:")
            for e in entities:
                lines.append(f"  • {e['entity']} ({e['type']}): {e['description']}")

        if relations:
            lines.append("High-Level Causal & Regulatory Traversals:")
            for r in relations:
                lines.append(f"  • {r['path']} ({r.get('description', '')})")

        return "\n".join(lines)
