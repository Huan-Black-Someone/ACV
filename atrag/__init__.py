"""
ATraG: Traditional-CV-Guided Adaptive Multimodal Knowledge Graphs Beyond Brute-Force Ensembles
"""

__version__ = "1.0.0"

from .stage0_filter import DefensiveFilter, TelemetryRecord
from .stage1_coreset import SubmodularCoresetSelector
from .stage2_perception import TriModalPerceptionEngine
from .stage3_verification import CLIPSemanticVerifier
from .stage4_lightrag import LightRAGGraphEngine
from .dataset_adapters import get_dataset_adapter

__all__ = [
    "DefensiveFilter",
    "TelemetryRecord",
    "SubmodularCoresetSelector",
    "TriModalPerceptionEngine",
    "CLIPSemanticVerifier",
    "LightRAGGraphEngine",
    "get_dataset_adapter",
]
