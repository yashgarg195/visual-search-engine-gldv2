"""Geometric Verification and Stage-2 Reranking Package.

Optional, pluggable module for second-stage geometric verification
of visual search candidates.
"""

from src.reranking.geometric_verifier import GeometricVerificationResult, GeometricVerifier
from src.reranking.pipeline import GeometricReranker

__all__ = [
    "GeometricVerifier",
    "GeometricVerificationResult",
    "GeometricReranker",
]
