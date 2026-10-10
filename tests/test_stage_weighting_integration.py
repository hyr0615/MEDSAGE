"""CPU checks of the trainer's actual stage-weight construction and PG loss."""
import ast
import os
import re
import random
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from verl.trainer import sa_outcome_credit
from verl.trainer.core_algos import compute_policy_loss
from verl.trainer.stage_token_mapping import assign_token_stages, decode_with_offsets

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def tokenizer():
    path = os.environ.get('MEDSAGE_TEST_TOKENIZER')
    if not path:
        pytest.skip('Set MEDSAGE_TEST_TOKENIZER to a local checkpoint tokenizer.')
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)


def trainer_class():
    tree = ast.parse((ROOT / 'easyr1/verl/trainer/ray_trainer.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RayPPOTrainer')
    cls.bases, cls.keywords, cls.decorator_list = [], [], []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and
                (n.name.startswith('_sagrpo_') or n.name == '_apply_stage_aware_grpo')]
    scope = dict(vars(sa_outcome_credit))
    scope.update(torch=torch, np=np, re=re, Any=Any, DataProto=object, defaultdict=defaultdict,
                 assign_token_stages=assign_token_stages, decode_with_offsets=decode_with_offsets)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<trainer-stage-methods>', 'exec'), scope)
    return scope['RayPPOTrainer']


@pytest.mark.parametrize('outcome_aware', [False, True])
def test_actual_stage_weight_path_changes_policy_gradient(tokenizer, outcome_aware):
    trainer = trainer_class()()
    trainer.tokenizer = tokenizer
    trainer.config = SimpleNamespace(algorithm=SimpleNamespace(
        sa_alpha=1.0, sa_min_weight=1.0, sa_max_weight=2.0,
        sa_component_advantages=False, sa_component_advantage_mix=0.25,
        sa_stage_gamma=1.25, sa_stage_mix=0.25, sa_stage_min_multiplier=0.85,
        sa_stage_max_multiplier=1.15, sa_use_l_divergence=False,
        sa_outcome_aware_credit=outcome_aware,
    ))
    texts = [
        '<LOC>{"bbox_xyxy_pixel":[[1,2,10,20]]}</LOC><VIS>' + visual +
        '</VIS><KNO>Same relation.</KNO><CON>' + ('yes' if i < 3 else 'no') + '</CON>'
        for i, visual in enumerate(['Small focal opacity.', 'Round dark area.', 'Bright irregular focus.',
                                   'No finding.', 'Clear field.', 'Homogeneous region.'])
    ]
    encoded = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    width = max(map(len, encoded)) + 2
    responses = torch.zeros((6, width), dtype=torch.long)
    mask = torch.zeros_like(responses)
    for i, ids in enumerate(encoded):
        responses[i, :len(ids)] = torch.tensor(ids)
        mask[i, :len(ids)] = 1
    advantages = torch.tensor([1., 1., 1., -1., -1., -1.]).unsqueeze(1).expand_as(mask).clone()
    scores = torch.zeros_like(advantages)
    scores[:3, 0] = 1.0
    batch = SimpleNamespace(batch={'responses': responses, 'response_mask': mask,
                                  'advantages': advantages, 'token_level_scores': scores,
                                  'sa_reward_accuracy': torch.tensor([1., 1., 1., 0., 0., 0.])},
                            non_tensor_batch={'uid': ['same-prompt'] * 6})
    metrics = {}
    trainer._apply_stage_aware_grpo(batch, metrics)
    weights = batch.batch['sa_token_weights']
    _, stages = trainer._sagrpo_token_stage_ids(responses, mask)
    assert all(bool((stages == i).any()) for i in range(1, 5))
    assert torch.equal(batch.batch['advantages'], advantages)
    assert bool((weights[mask == 0] == 1).all())
    assert bool((weights[stages == 1] == 1.5).all())
    assert not torch.allclose(weights[stages == 2], torch.full_like(weights[stages == 2], 1.5))

    def grad(token_weights):
        log_probs = torch.zeros_like(advantages, requires_grad=True)
        loss, _ = compute_policy_loss(torch.zeros_like(log_probs), log_probs, advantages, mask,
                                      0.2, 0.2, 0.0, 'default', 'token', token_weights)
        loss.backward()
        return log_probs.grad

    weighted, ordinary = grad(weights), grad(None)
    assert bool(torch.isfinite(weighted).all())
    assert not torch.allclose(weighted, ordinary)
    assert not bool(weighted[mask == 0].any())
    assert metrics['sagrpo/no_stage_tag_response_count'] == 0


def test_randomized_context_and_noncanonical_token_boundaries(tokenizer):
    trainer = trainer_class()()
    trainer.tokenizer = tokenizer
    randomizer = random.Random(42)
    contents = ['1.5 mm', 'x', '\u80ba\u90e8', 'caf\u00e9', '\u03b1\u03b2', 'first\nsecond', 'yes', 'ER+']
    for _ in range(40):
        text = ''.join('<' + tag + '>' + randomizer.choice(contents) + '</' + tag + '>'
                       for tag in ['LOC', 'VIS', 'KNO', 'CON'])
        boundaries = sorted(set([0, len(text)] + [randomizer.randrange(len(text)) for _ in range(12)]))
        chunks = [text[a:b] for a, b in zip(boundaries, boundaries[1:])]
        for ids in [tokenizer.encode(text, add_special_tokens=False),
                    [token for chunk in chunks for token in tokenizer.encode(chunk, add_special_tokens=False)]]:
            decoded, offsets = decode_with_offsets(tokenizer, ids)
            assert decoded == text
            stages = assign_token_stages(offsets, trainer._sagrpo_stage_char_spans(decoded))
            assert all(stages.count(i) > 0 for i in range(1, 5))
            assert len(stages) == len(ids)
