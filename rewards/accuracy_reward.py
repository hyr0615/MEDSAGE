import json
import re
from typing import Any


TAG_RE = re.compile(r"<([A-Z_]+)>\s*(.*?)\s*</\1>", re.S)


def parse_ground_truth(ground_truth: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(ground_truth, dict):
        return ground_truth
    try:
        data = json.loads(ground_truth)
        if isinstance(data, dict):
            return data
    except (TypeError, ValueError):
        pass
    return {"answer": ground_truth}


def extract_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text or "", flags=re.I | re.S)
    return match.group(1).strip() if match else ""


def strip_tags(text: str) -> str:
    return TAG_RE.sub(lambda m: " " + m.group(2) + " ", text or "")


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text).strip()
    # Keep decimals, polarity signs and separators: 1.5 != 15, ER+ != ER-.
    return text.removesuffix('.').strip()


def token_f1(pred: str, ref: str) -> float:
    p = normalize_text(pred).split()
    r = normalize_text(ref).split()
    if not p or not r:
        return 0.0
    pc = {}
    for t in p:
        pc[t] = pc.get(t, 0) + 1
    overlap = 0
    for t in r:
        if pc.get(t, 0) > 0:
            overlap += 1
            pc[t] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def contains_token_phrase(longer: str, shorter: str) -> bool:
    long_tokens = normalize_text(longer).split()
    short_tokens = normalize_text(shorter).split()
    if not short_tokens or len(short_tokens) > len(long_tokens):
        return False
    width = len(short_tokens)
    return any(long_tokens[index : index + width] == short_tokens for index in range(len(long_tokens) - width + 1))


def conclusion_or_final(text: str, require_conclusion: bool = False) -> str:
    if not isinstance(text, str):
        return ""
    opening = len(re.findall(r"<CON>", text, flags=re.I))
    closing = len(re.findall(r"</CON>", text, flags=re.I))
    if opening or closing:
        if (opening, closing) != (1, 1):
            return ""
        match = re.search(r"<CON>\s*(.*?)\s*</CON>\s*$", text, flags=re.I | re.S)
        if not match or re.search(r"</?[A-Za-z][^>]*>", match.group(1)):
            return ""
        return match.group(1).strip()
    if require_conclusion or re.search(r"</?[A-Za-z][^>]*>", text):
        return ""
    return text.strip()


def explicit_answer(text: str) -> str:
    return re.sub(r"^\s*(?:the\s+)?(?:final\s+)?answer(?:\s+is\b\s*:?\s*|\s*:\s*)", "", text.strip(), flags=re.I)


def retracts_answer(text: str) -> bool:
    subject = r"(?:(?:this|that|my|the|previous|initial)\s+)?(?:answer|choice|selection|response)|this|that"
    predicate = r"(?:(?:is|was)\s+(?:false|incorrect|wrong|invalid|not\s+(?:correct|true))|(?:isn't|wasn't|is\s+not|was\s+not)\s+(?:correct|true))"
    return bool(re.search(rf"\b(?:{subject})\s+{predicate}\b", text, re.I) or re.search(
        r"\b(?:i\s+(?:retract|withdraw|change|revise)|retracting|withdrawing|changing|revising)\s+"
        r"(?:(?:my|the|this|that|previous|initial)\s+)?(?:answer|choice|selection|response)\b", text, re.I))


def binary_choice(text: str) -> str | None:
    text = explicit_answer(text)
    if retracts_answer(text):
        return None
    tokens = re.findall(r"\b(?:yes|no)\b", text.lower())
    if len(set(tokens)) != 1:
        return None
    match = re.match(r"^(yes|no)(?=$|[\s.,!;:])", text, flags=re.I)
    if not match:
        return None
    tail = text[match.end():]
    if tail.strip() and not re.match(r"^\s*[.,!;:]", tail):
        return None
    if re.search(r"\b(?:maybe|perhaps|uncertain|not\s+(?:yes|no|correct)|incorrect|wrong|(?:yes|no)\s+is\s+(?:false|not))\b", text, re.I):
        return None
    return match.group(1).lower()


def explicit_mc_choice(text: str) -> str | None:
    text = explicit_answer(text)
    if retracts_answer(text):
        return None
    match = re.match(r"^(?:option\s+|choice\s+)?(?:\(([ABCD])\)(?=$|\s|[.,:])|([ABCD])(?=$|\s*[.):,]))", text, re.I)
    if not match:
        return None
    choice = (match.group(1) or match.group(2)).upper()
    tail = text[match.end():]
    # Distinguish option markers from articles such as "B. A lung lesion".
    option_patterns = [
        r"\b(?:option|choice|answer)\s*(?:(?:is|was|would\s+be|should\s+be|must\s+be)\s*|[:=]\s*)?\(?([abcd])\)?\b",
        r"(?:^|[.,;:\n]\s*|\b(?:and|or|versus|vs)\s+)\(?([abcd])\)?(?=\s*(?:$|[.,;:]|\band\b|\bor\b))",
        r"\(([abcd])\)",
        r"\b([abcd])\s+is\s+(?:correct|the\s+answer)\b",
        r"\b(?:actually|rather|instead|correction)\s*[:,]?\s*(?:option\s+|choice\s+)?\(?([abcd])\)?(?=\s*(?:$|[.,;:]|\bis\b|\bbecause\b))",
        r"\b(?:choose|select|pick)\s+(?:option\s+|choice\s+)?\(?([abcd])\)?(?=\s*(?:$|[.,;:]|\bis\b|\bbecause\b))",
    ]
    mentions = {c.upper() for pattern in option_patterns for c in re.findall(pattern, tail, re.I)}
    if mentions - {choice} or re.search(r"\b(?:not|maybe|perhaps|instead|incorrect|wrong)\b", tail, re.I):
        return None
    return choice


def accuracy_reward(response: str, ground_truth: str) -> float:
    gt = parse_ground_truth(ground_truth)
    ref_answer = gt.get("answer", ground_truth)
    availability = gt.get("stage_availability") or {}
    c_only_direct = (
        bool(availability.get("C"))
        and not any(availability.get(stage) for stage in ("L", "V", "K"))
    )
    require_conclusion = bool(gt.get("full_LVKC")) or (not c_only_direct and gt.get("task_type") in {
        "full_LVKC",
        "C",
    })
    pred = conclusion_or_final(response, require_conclusion=require_conclusion)
    ref = conclusion_or_final(ref_answer)
    pred_n = normalize_text(explicit_answer(pred))
    ref_n = normalize_text(ref)
    if not pred_n or not ref_n:
        return 0.0
    if ref_n in {"yes", "no"}:
        return float(binary_choice(pred) == ref_n)
    if gt.get("question_type") == "multiple_choice" or re.fullmatch(r"[ABCD]", ref.strip()):
        return float(ref.strip().upper() in {"A", "B", "C", "D"} and explicit_mc_choice(pred) == ref.strip().upper())
    # Open answers require a complete normalized match.
    return float(pred_n == ref_n)


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use reward_type=batch for MEDSAGE accuracy reward.")
    scores = [accuracy_reward(x.get("response", ""), x.get("ground_truth", "")) for x in reward_inputs]
    return [{"overall": score, "accuracy": score} for score in scores]
