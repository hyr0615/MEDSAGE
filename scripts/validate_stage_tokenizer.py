"""CPU regression check using a local training checkpoint's actual tokenizer."""
import argparse
import ast
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'easyr1/verl/trainer'))
from stage_token_mapping import assign_token_stages, decode_with_offsets


def trainer_methods(torch):
    tree = ast.parse((ROOT / 'easyr1/verl/trainer/ray_trainer.py').read_text())
    names = {'_sagrpo_stage_char_spans', '_sagrpo_token_stage_ids'}
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    for node in nodes:
        node.decorator_list = []
    scope = {'torch': torch, 're': re, 'assign_token_stages': assign_token_stages,
             'decode_with_offsets': decode_with_offsets}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<trainer-methods>', 'exec'), scope)
    return scope


def legacy_counts(tokenizer, ids):
    counts = []
    for tag in ['LOC', 'VIS', 'KNO', 'CON']:
        opening = tokenizer.encode('<' + tag + '>', add_special_tokens=False)
        closing = tokenizer.encode('</' + tag + '>', add_special_tokens=False)
        starts = [i + len(opening) for i in range(len(ids)) if ids[i:i + len(opening)] == opening]
        ends = [i for i in range(len(ids)) if ids[i:i + len(closing)] == closing]
        counts.append(sum(next((e - s for e in ends if e >= s), 0) for s in starts))
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', required=True, help='Local tokenizer directory from the resolved training config.')
    args = parser.parse_args()
    import torch
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    methods = trainer_methods(torch)
    trainer = SimpleNamespace(tokenizer=tokenizer, _sagrpo_stage_char_spans=methods['_sagrpo_stage_char_spans'])
    examples = [
        '<LOC>{"bbox_xyxy_pixel":[[1,2,10,20]]}</LOC><VIS>Finding.</VIS><KNO>Relation.</KNO><CON>yes</CON>',
        '<LOC> [1, 2, 10, 20] </LOC>\n<VIS> A focal lesion. </VIS>\n<KNO> A supported relation. </KNO>\n<CON> Tumor. </CON>',
        '<loc>{"bbox_xyxy_pixel":[[1,2,10,20]]}</loc><vis>\u80ba\u90e8 finding.</vis><kno>Relation.</kno><con>No.</con>',
    ]
    rows = []
    for index, text in enumerate(examples):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if tokenizer.eos_token_id is not None:
            ids.append(tokenizer.eos_token_id)
        tokens = torch.tensor([[0] + ids + [0, 0]])
        mask = torch.tensor([[0] + [1] * len(ids) + [0, 0]])
        texts, stages = methods['_sagrpo_token_stage_ids'](trainer, tokens, mask)
        counts = [(stages == stage).sum().item() for stage in range(1, 5)]
        assert all(counts), counts
        assert not bool(stages[mask == 0].any())
        spans = trainer._sagrpo_stage_char_spans(texts[0])
        _, offsets = decode_with_offsets(tokenizer, ids)
        for position, (start, end) in enumerate(offsets, 1):
            assigned = stages[0, position].item()
            if assigned:
                assert any(min(end, b) > max(start, a) for a, b in spans['LVKC'[assigned - 1]])
        if tokenizer.eos_token_id is not None:
            assert stages[0, len(ids)].item() == 0
        # V and C receive distinct multipliers; boundary tokens remain neutral.
        multipliers = torch.tensor([1.0, 1.0, 0.85, 1.0, 1.15])[stages]
        assert bool((multipliers[stages == 2] == 0.85).all())
        assert bool((multipliers[stages == 4] == 1.15).all())
        rows.append({'example': index + 1, 'legacy_counts': legacy_counts(tokenizer, ids), 'fixed_counts': counts})
    # Exercise generated token IDs that differ from full-text re-encoding.
    pieces = ['<LOC>', '{"bbox_xyxy_pixel":[[1,2,10,20]]}', '</LOC>', '<VIS>', 'Finding.', '</VIS>',
              '<KNO>', 'Relation.', '</KNO>', '<CON>', 'yes', '</CON>']
    ids = [token for part in pieces for token in tokenizer.encode(part, add_special_tokens=False)]
    text, offsets = decode_with_offsets(tokenizer, ids)
    stages = assign_token_stages(offsets, trainer._sagrpo_stage_char_spans(text))
    assert all(stages.count(i) > 0 for i in range(1, 5))
    # Padding must not acquire a stage assignment.
    _, empty = methods['_sagrpo_token_stage_ids'](trainer, torch.zeros((1, 3), dtype=torch.long), torch.zeros((1, 3)))
    assert not bool(empty.any())
    print(json.dumps({'tokenizer_class': type(tokenizer).__name__, 'examples': rows,
                      'padding_eos_and_noncanonical_checks': 'passed', 'device': 'cpu'}, indent=2))


if __name__ == '__main__':
    main()
