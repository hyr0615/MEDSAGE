"""Align stage content with the actual generated tokens, not isolated tags."""


def decode_with_offsets(tokenizer, ids):
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if not ids:
        return text, []
    if getattr(tokenizer, "is_fast", False):
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        # A generated sequence need not be the canonical encoding of its text.
        if list(encoded["input_ids"]) == ids:
            offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
            if len(offsets) != len(ids) or any(not 0 <= a <= b <= len(text) for a, b in offsets):
                raise ValueError("Invalid tokenizer character offsets for stage weighting.")
            return text, offsets

    # Decode actual prefixes when re-encoding changes token IDs. Buffer partial
    # UTF-8 characters so all contributing byte tokens receive the same span.
    offsets = []
    start = 0
    pending = 0
    for end in range(1, len(ids) + 1):
        prefix = tokenizer.decode(ids[:end], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if not text.startswith(prefix):
            if prefix.endswith("\ufffd") and text.startswith(prefix.rstrip("\ufffd")):
                continue
            raise ValueError("Tokenizer decoding is not prefix-stable; cannot align stage weights safely.")
        if len(prefix) < start:
            raise ValueError("Non-monotonic token offsets for stage weighting.")
        offsets.extend([(start, len(prefix))] * (end - pending))
        start, pending = len(prefix), end
    if len(offsets) != len(ids) or start != len(text):
        raise ValueError("Incomplete token alignment for stage weighting.")
    return text, offsets


def assign_token_stages(offsets, spans):
    """Assign boundary-spanning tokens by greatest content overlap; tags are neutral."""
    stage_to_id = {"L": 1, "V": 2, "K": 3, "C": 4}
    result = []
    for start, end in offsets:
        overlaps = [
            (sum(max(0, min(end, b) - max(start, a)) for a, b in spans[stage]), stage_id)
            for stage, stage_id in stage_to_id.items()
        ]
        # The first stage wins a tie; each token is weighted exactly once.
        overlap, stage_id = max(overlaps, key=lambda pair: pair[0])
        result.append(stage_id if overlap else 0)
    for stage, stage_id in stage_to_id.items():
        if any(b > a for a, b in spans[stage]) and stage_id not in result:
            raise ValueError(f"Nonempty {stage} stage has no aligned tokens.")
    return result
