"""Generate evals/dataset.json from the synthetic claim set.
The eval dataset is committed BEFORE agent changes are graded against it.
Run: python evals/generate_dataset.py
"""
import json, sys
sys.path.insert(0, ".")
from aca.corpus import generate_claims
from aca.evals import build_eval_cases, agent_view

if __name__ == "__main__":
    cases = build_eval_cases(generate_claims())
    payload = [{"case_id": c.case_id, "claim_agent_view": agent_view(c),
                "gold_category": c.gold_category, "source": c.source} for c in cases]
    open("evals/dataset.json", "w").write(json.dumps(payload, indent=1))
    print(f"wrote {len(payload)} cases to evals/dataset.json")
