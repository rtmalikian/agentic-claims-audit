"""Production feedback loop: reviewer corrections on findings are stored
and converted into new regression eval cases, so every corrected mistake
becomes a test the agent must keep passing."""
from __future__ import annotations
import json, os
from .evals import EvalCase

class FeedbackStore:
    def __init__(self, path: str | None = None):
        self.path = path
        self.corrections: list[dict] = []
        if path and os.path.exists(path):
            self.corrections = json.loads(open(path).read() or "[]")

    def add_correction(self, claim_id: str, claim: dict, predicted_category: str | None,
                       corrected_category: str | None, reviewer: str, note: str = "") -> dict:
        rec = {"claim_id": claim_id, "claim": claim, "predicted_category": predicted_category,
               "corrected_category": corrected_category, "reviewer": reviewer, "note": note}
        self.corrections.append(rec); self._save(); return rec

    def _save(self):
        if self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            open(self.path, "w").write(json.dumps(self.corrections, indent=1))

    def to_eval_cases(self) -> list[EvalCase]:
        return [EvalCase(case_id=f"FB-{i+1:03d}-{c['claim_id']}", claim=c["claim"],
                         gold_category=c["corrected_category"], source="reviewer_correction")
                for i, c in enumerate(self.corrections)]

def correction_to_eval_case(rec: dict) -> EvalCase:
    return EvalCase(case_id=f"FB-{rec['claim_id']}", claim=rec["claim"],
                    gold_category=rec["corrected_category"], source="reviewer_correction")
