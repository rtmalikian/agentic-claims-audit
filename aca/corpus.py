"""Synthetic corpus + claims generator (seeded, deterministic).

SYNTHETIC DATA ONLY. The guideline/policy corpus and the claim set are
generated from templates to demonstrate retrieval and audit pipelines at
scale. They are not real payer policy, not real claims, and contain no PHI.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict

# --- controlled vocabularies used by both corpus and claim generation -----
CPT = {
    "99213": ("established patient office visit level 3", 3),
    "99214": ("established patient office visit level 4", 4),
    "99215": ("established patient office visit level 5", 5),
    "99212": ("established patient office visit level 2", 2),
    "97110": ("therapeutic exercises", 0),
    "97140": ("manual therapy techniques", 0),
    "97530": ("therapeutic activities", 0),
    "20610": ("arthrocentesis major joint", 0),
    "73562": ("radiologic examination knee 3 views", 0),
    "80053": ("comprehensive metabolic panel", 0),
    "85025": ("complete blood count with differential", 0),
    "36415": ("venipuncture collection", 0),
    "93000": ("electrocardiogram complete", 0),
    "71046": ("radiologic examination chest 2 views", 0),
    "43239": ("upper GI endoscopy with biopsy", 0),
    "45378": ("colonoscopy diagnostic", 0),
    "29881": ("knee arthroscopy with meniscectomy", 0),
    "27447": ("total knee arthroplasty", 0),
    "64483": ("transforaminal epidural injection lumbar", 0),
    "20552": ("trigger point injection single muscle", 0),
}
# code pairs that must not be billed together on the same date (bundling)
BUNDLING_PAIRS = [("36415", "80053"), ("36415", "85025"), ("97140", "97110")]
# diagnosis (ICD-10-ish) -> procedures it can support for medical necessity
NECESSITY_MAP = {
    "M17.11": ["20610", "73562", "27447", "97110", "97140", "97530"],
    "M25.561": ["73562", "20610", "97110"],
    "R07.9": ["93000", "71046"],
    "K29.70": ["43239"],
    "Z12.11": ["45378"],
    "M54.16": ["64483", "20552"],
    "S83.241A": ["29881", "73562"],
    "E11.9": ["80053", "85025"],
    "I10": ["80053", "85025", "93000"],
    "J06.9": ["99212", "99213"],
}
EVAL_CODES = ["99212", "99213", "99214", "99215"]
DOC_KINDS = ["coding_rule", "coverage_criteria", "fee_rule", "duplicate_rule", "bundling_rule"]


@dataclass
class Doc:
    doc_id: str
    kind: str
    title: str
    text: str
    codes: list[str] = field(default_factory=list)

    def to_dict(self): return asdict(self)


@dataclass
class ClaimLine:
    cpt: str
    units: int = 1
    charge: float = 100.0


@dataclass
class Claim:
    claim_id: str
    patient_token: str
    provider_id: str
    date_of_service: str
    diagnosis: str
    lines: list[ClaimLine]
    documentation: str
    documented_level: int | None
    planted_error: str | None  # gold label: None = clean
    history_match: bool = False  # a same patient/provider/code/date claim exists in history

    def to_dict(self): return asdict(self)


def generate_corpus(n_docs: int = 2100, seed: int = 42) -> list[Doc]:
    rng = random.Random(seed)
    docs: list[Doc] = []
    codes = list(CPT)
    templates = {
        "coding_rule": "Coding rule for {code} ({desc}): bill {code} only when the documentation supports {desc}. The billed level must not exceed the documented level. Claims for {code} without supporting documentation are not payable.",
        "coverage_criteria": "Coverage criteria for {code} ({desc}): {code} is covered when medical necessity is established by a supporting diagnosis and the service is documented. Services for {code} billed with an unrelated diagnosis do not meet coverage criteria.",
        "fee_rule": "Fee schedule rule for {code} ({desc}): the allowed amount for {code} is determined by the fee schedule. Duplicate payment for the same {code} on the same date of service for the same patient and provider is not allowed.",
        "duplicate_rule": "Duplicate claim rule: a claim is a duplicate when the same patient, provider, procedure code {code} and date of service were already submitted and paid. Duplicate submissions of {code} must be denied as exact duplicates.",
        "bundling_rule": "Bundling rule: when a comprehensive code and a component code are billed together on the same date, the component is bundled into the comprehensive code and is not separately payable. This rule applies to the code referenced here, {code} ({desc}).",
    }
    i = 0
    while len(docs) < n_docs:
        kind = rng.choice(DOC_KINDS)
        code = rng.choice(codes)
        desc, _lvl = CPT[code]
        variant = rng.randint(1, 9999)
        text = templates[kind].format(code=code, desc=desc)
        text += f" Reference variant {variant}. Section {i % 40 + 1}.{i % 9 + 1}."
        if kind == "bundling_rule":
            for a, b in BUNDLING_PAIRS:
                if code in (a, b):
                    other = b if code == a else a
                    text += f" Specifically, {other} is bundled into / bundles with {code} when billed on the same date."
        docs.append(Doc(doc_id=f"DOC-{len(docs)+1:04d}", kind=kind,
                        title=f"{kind.replace('_', ' ').title()} — {code}", text=text, codes=[code]))
        i += 1
    return docs


def _base_claim(rng, idx, diagnosis, cpts, level=None, doc=True):
    return Claim(
        claim_id=f"CLM-{idx:04d}", patient_token=f"PAT-{rng.randint(1000, 9999)}",
        provider_id=f"PRV-{rng.randint(10, 99)}", date_of_service="2026-01-15",
        diagnosis=diagnosis,
        lines=[ClaimLine(cpt=c, charge=float(rng.randint(50, 900))) for c in cpts],
        documentation=("Visit documented. Assessment and plan recorded. "
                       f"Documented visit level {level}. Medical decision making recorded."
                       if doc and level else
                       "Visit documented. Assessment and plan recorded. Medical decision making recorded." if doc else ""),
        documented_level=level, planted_error=None)


def generate_claims(n: int = 120, seed: int = 7) -> list[Claim]:
    """120 claims: 24 clean, 24 duplicate, 24 upcoding, 16 missing documentation,
    16 medical-necessity-questionable, 16 bundling."""
    rng = random.Random(seed)
    claims: list[Claim] = []
    idx = 1
    def add(c):
        nonlocal idx; claims.append(c); idx += 1
    for _ in range(24):  # clean
        dx = rng.choice(list(NECESSITY_MAP)); lvl = rng.choice([2, 3, 4])
        code = EVAL_CODES[lvl - 2]
        add(_base_claim(rng, idx, dx, [code], level=lvl))
    for _ in range(24):  # duplicate
        c = _base_claim(rng, idx, "E11.9", ["80053"], level=None); c.planted_error = "duplicate"; c.history_match = True; add(c)
    for _ in range(24):  # upcoding: billed 99215, documented level 3
        c = _base_claim(rng, idx, "I10", ["99215"], level=3); c.planted_error = "upcoding"; add(c)
    for _ in range(16):  # missing documentation
        c = _base_claim(rng, idx, "M17.11", ["20610"], level=None, doc=False); c.planted_error = "missing_documentation"; add(c)
    for _ in range(16):  # medical necessity: knee arthroplasty billed for upper-resp diagnosis
        c = _base_claim(rng, idx, "J06.9", ["27447"], level=None); c.planted_error = "medical_necessity"; add(c)
    for _ in range(16):  # bundling: venipuncture with a panel
        pair = rng.choice(BUNDLING_PAIRS)
        dx = "E11.9" if pair[1] in ("80053", "85025") else "M17.11"
        c = _base_claim(rng, idx, dx, list(pair), level=None); c.planted_error = "bundling"; add(c)
    rng.shuffle(claims)
    return claims[:n]
