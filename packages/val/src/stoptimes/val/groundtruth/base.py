"""GroundTruthDetector protocol — single source of the detector contract. Phase P4.

Re-exports the Protocol from stoptimes.val.domain.interfaces so that detector
implementations and callers can import from a single, stable location without
coupling directly to the domain layer.
"""

from __future__ import annotations

from stoptimes.val.domain.interfaces import GroundTruthDetector

__all__ = ["GroundTruthDetector"]
