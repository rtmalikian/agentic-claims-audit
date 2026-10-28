"""Offline-vs-online eval comparison: offline = gold synthetic set,
online = reviewer-corrected production findings. Drift between the two
is the signal that the offline set no longer matches production."""
from __future__ import annotations

def precision_recall(tp: int, fp: int, fn: int) -> tuple[float, float]:
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    return round(p, 4), round(r, 4)

def compare_offline_online(offline: dict, online: dict) -> dict:
    keys = ["precision", "recall", "citation_rate"]
    drift = {k: round(online.get(k, 0) - offline.get(k, 0), 4) for k in keys}
    return {"offline": offline, "online": online, "drift": drift,
            "alert": any(abs(v) >= 0.1 for v in drift.values())}
