"""Conservative, source-grounded direct V/K reward diagnostics.

The specifications are extracted from structured SFT-Train provenance, never
from the SFT answer or the RL final answer.  Free-text-only supervision has no
direct reward and must use policy continuation instead.
"""

from __future__ import annotations

import re
from typing import Any


def _words(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip() + " "


def _present(text: str, alternatives: tuple[str, ...]) -> bool:
    normalized = _words(text)
    return any(" " + _words(term).strip() + " " in normalized for term in alternatives)


def _negated(text: str, term: str) -> bool:
    escaped = re.escape(term).replace(r"\ ", r"\s+")
    return re.search(rf"\b(?:no|not|without|absent|negative for)\s+(?:\w+\s+){{0,2}}{escaped}\b", text.lower()) is not None


BREAST_FEATURES = {
    "round": ("round",),
    "oval": ("oval",),
    "irregular": ("irregular",),
    "hypoechoic": ("hypoechoic",),
    "hyperechoic": ("hyperechoic",),
    "isoechoic": ("isoechoic",),
    "anechoic": ("anechoic",),
    "heterogeneous": ("heterogeneous", "mixed echogenicity"),
    "noncircumscribed_margin": ("non circumscribed margin", "noncircumscribed margin", "not circumscribed margin", "indistinct margin", "angular margin", "microlobulated margin", "spiculated margin"),
    "circumscribed_margin": ("circumscribed margin", "well defined margin", "well circumscribed margin"),
    "angular": ("angular",),
    "indistinct": ("indistinct",),
    "microlobulated": ("microlobulated",),
    "spiculated": ("spiculated",),
    "halo": ("halo",),
    "calcifications": ("calcification", "calcifications"),
    "skin_thickening": ("skin thickening", "thickened skin"),
}

BRATS_COMPONENTS = {
    "enhancing_tumor": ("enhancing tumor", "enhancing component", "enhancement"),
    "necrotic_and_non_enhancing_tumor_core": ("necrotic", "non enhancing tumor core", "non enhancing core"),
    "peritumoral_edema": ("peritumoral edema", "peritumoural edema", "edema component", "oedema component"),
}


def _feature_present(text: str, name: str) -> bool:
    present = _present(text, BREAST_FEATURES[name])
    if name == "circumscribed_margin" and _present(text, BREAST_FEATURES["noncircumscribed_margin"]):
        return False
    return present and not any(_negated(text, term) for term in BREAST_FEATURES[name])


def _brats_feature_present(text: str, name: str) -> bool:
    if name == "enhancing_tumor":
        normalized = _words(text)
        return re.search(r"(?<!non )\benhancing\b|\benhancement\b", normalized) is not None
    return _present(text, BRATS_COMPONENTS[name]) and not any(_negated(text, term) for term in BRATS_COMPONENTS[name])


def _circumscribed_present(text: str) -> bool:
    normalized = _words(text)
    return " circumscribed " in normalized and " non circumscribed " not in normalized and " not circumscribed " not in normalized


def _breast_expected(raw: dict[str, Any]) -> list[str]:
    expected: list[str] = []
    for field, choices in (
        ("Shape", ("round", "oval", "irregular")),
        ("Echogenicity", ("hypoechoic", "hyperechoic", "isoechoic", "anechoic", "heterogeneous")),
    ):
        value = str(raw.get(field, "")).lower()
        expected.extend(choice for choice in choices if choice in value)
    margin = str(raw.get("Margin", "")).lower()
    if "not circumscribed" in margin:
        expected.append("noncircumscribed_margin")
        expected.extend(choice for choice in ("angular", "indistinct", "microlobulated", "spiculated") if choice in margin)
    elif "circumscribed" in margin:
        expected.append("circumscribed_margin")
    for field, name in (("Halo", "halo"), ("Calcifications", "calcifications"), ("Skin_thickening", "skin_thickening")):
        if str(raw.get(field, "")).lower() == "yes":
            expected.append(name)
    return list(dict.fromkeys(expected))


def score_direct_v(text: str, spec: dict[str, Any] | None) -> float | None:
    if not spec:
        return None
    kind = spec.get("kind")
    if kind == "breast_descriptors":
        expected = _breast_expected(spec["raw_descriptors"])
        if not expected:
            return None
        correct = sum(_feature_present(text, name) for name in expected)
        wrong = sum(_feature_present(text, name) for name in BREAST_FEATURES if name not in expected)
        # An unsupported finding should not be erased by listing every true feature.
        return max(0.0, min(1.0, (correct - wrong) / len(expected)))
    if kind == "brats_mask_components":
        components = {key for key, count in spec["mask_components"].items() if int(count) > 0}
        sequence = str(spec.get("sequence", "")).upper()
        if sequence == "T1":
            return 1.0 if components and _present(text, ("brain tumor region", "tumor region", "tumour region")) else 0.0
        if sequence == "T1CE":
            eligible = {"enhancing_tumor", "necrotic_and_non_enhancing_tumor_core"}
        elif sequence in ("T2", "FLAIR"):
            eligible = {"peritumoral_edema"}
        else:
            return None
        expected = components & eligible
        if not expected:
            return None
        correct = sum(_brats_feature_present(text, name) for name in expected)
        wrong = sum(_brats_feature_present(text, name) for name in BRATS_COMPONENTS if name not in components)
        return max(0.0, min(1.0, (correct - wrong) / len(expected)))
    if kind == "glas_mask_count":
        match = re.search(r"\b(\d+)\s+(?:segmented\s+)?gland(?:ular)?\s+(?:profiles?|instances?|regions?)\b", text.lower())
        if not match:
            return 0.0
        return 1.0 if int(match.group(1)) == int(spec["gland_instance_count"]) else 0.0
    raise ValueError(f"Unknown V direct specification: {kind}")


def _relation_feature(text: str, normalized_feature: str) -> bool:
    if normalized_feature == "noncircumscribed_margin":
        return _present(text, BREAST_FEATURES["noncircumscribed_margin"])
    if normalized_feature == "irregular_shape":
        return _present(text, ("irregular shape", "irregular lesion"))
    if normalized_feature == "oval_circumscribed_hypoechoic_pattern":
        return _present(text, ("oval",)) and _circumscribed_present(text)
    if normalized_feature == "anechoic_circumscribed_cystic_pattern":
        return _present(text, ("anechoic", "cystic")) and _circumscribed_present(text)
    return False


def score_direct_k(text: str, spec: dict[str, Any] | None) -> float | None:
    if not spec:
        return None
    kind = spec.get("kind")
    if kind == "breast_birads_relations":
        relations = spec["relations"]
        if not relations:
            return None
        scores = []
        for relation in relations:
            feature = _relation_feature(text, relation["normalized_feature"])
            relation_class = relation["interpretation_class"]
            if relation_class == "suspicious":
                conclusion = _present(text, ("suspicion", "suspicious", "concerning", "increased risk", "higher risk", "warrants biopsy", "tissue assessment"))
                contradiction = _present(text, ("not suspicious", "no suspicion", "clearly benign"))
            elif relation_class == "benign_leaning":
                conclusion = _present(text, ("benign leaning", "benign appearing", "benign cystic", "benign ultrasound", "low suspicion", "lower suspicion", "likely benign", "cystic pattern"))
                contradiction = _present(text, ("highly suspicious", "high risk", "warrants biopsy"))
            else:
                return None
            scores.append(1.0 if feature and conclusion and not contradiction else 0.0)
        return sum(scores) / len(scores)
    if kind == "glas_architecture_relation":
        relation = spec["relation_selection"]
        if relation == "architectural_disorder":
            good = _present(text, ("disrupted architecture", "architectural disorder", "disordered glandular architecture", "fragmentation"))
            bad = _present(text, ("preserved architecture", "well preserved architecture"))
        elif relation == "architectural_preservation":
            good = _present(text, ("preserved architecture", "preserved glandular organization", "glandular organization is preserved", "uniform gland", "limited fragmentation"))
            bad = _present(text, ("disrupted architecture", "architectural disorder"))
        else:
            return None
        return 1.0 if good and not bad else 0.0
    raise ValueError(f"Unknown K direct specification: {kind}")
