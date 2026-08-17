"""
Landmark name constants and JSON → canonical name mappings.

Training / inference use exactly these names, in this order.
Output dimension = NUM_LANDMARKS * 2.
"""

from __future__ import annotations

from typing import Dict

# Keep in sync with the trained Swin checkpoint head.
LANDMARK_NAMES = [
    "cmb31",
    "cb31",
    "cmt10",
    "cb11",
    "csp21",
    "cmtc1",
    "csp30",
    "cmb10",
    "cmtc0",
    "cb10",
    "cmt20",
    "csp10",
    "cmt11",
    "csp31",
    "cmb11",
    "cb30",
    "csp11",
    "cmb30",
    "csp20",
    "cmt21",
    "cmtc_mMPFA0",
    "cmtc_mMPFA1",
    "labelBL0",
    "labelBL1",
    "labelBR0",
    "labelBR1",
    "labelTL0",
    "labelTL1",
    "labelTR0",
    "labelTR1",
    "labelForCircleSetToBot0",
    "labelForCircleSetToBot1",
    "labelBottomAngle0",
    "labelBottomAngle1",
    "labelmMPFA0",
    "labelmMPFA1",
]

NUM_LANDMARKS = len(LANDMARK_NAMES)

# Per-leg mapping for unzipped JSON: two object_*.json files per case.
# Smaller object ID = right leg (suffix 0); larger = left leg (suffix 1).
# Keys are Pointizr drawingProps names.
BASE_TO_RIGHT: Dict[str, str] = {
    "csp1": "csp10",
    "csp2": "csp20",
    "csp3": "csp30",
    "cmtc": "cmtc0",
    "cmt1": "cmt10",
    "cmt2": "cmt20",
    "cmb1": "cmb10",
    "cmb3": "cmb30",
    "cb1": "cb10",
    "cb3": "cb30",
    "labelBL": "labelBL0",
    "labelBR": "labelBR0",
    "labelTL": "labelTL0",
    "labelTR": "labelTR0",
    "labelForCircleSetToBot": "labelForCircleSetToBot0",
    "labelBottomAngle": "labelBottomAngle0",
    "labelmMPFA": "labelmMPFA0",
    "cmtc_mMPFA": "cmtc_mMPFA0",
}
BASE_TO_LEFT: Dict[str, str] = {
    "csp1": "csp11",
    "csp2": "csp21",
    "csp3": "csp31",
    "cmtc": "cmtc1",
    "cmt1": "cmt11",
    "cmt2": "cmt21",
    "cmb1": "cmb11",
    "cmb3": "cmb31",
    "cb1": "cb11",
    "cb3": "cb31",
    "labelBL": "labelBL1",
    "labelBR": "labelBR1",
    "labelTL": "labelTL1",
    "labelTR": "labelTR1",
    "labelForCircleSetToBot": "labelForCircleSetToBot1",
    "labelBottomAngle": "labelBottomAngle1",
    "labelmMPFA": "labelmMPFA1",
    "cmtc_mMPFA": "cmtc_mMPFA1",
}

# Optional flat override applied after per-leg mapping (usually empty).
LANDMARK_NAME_MAPPING: Dict[str, str] = {}
