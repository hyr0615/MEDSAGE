"""Check explicit seed controls without importing the GPU generation stack."""
import argparse
import ast
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_training_seed_settings_are_explicit():
    for name in ('sft', 'dpo'):
        config = yaml.safe_load((ROOT / f'configs/{name}.yaml').read_text())
        assert config['seed'] == config['data_seed'] == 42
    rl = yaml.safe_load((ROOT / 'configs/rl.yaml').read_text())
    assert rl['data']['seed'] == 42
    assert rl['worker']['rollout']['seed'] == 42
    assert rl['worker']['actor']['clip_ratio_low'] == .2
    assert rl['worker']['actor']['clip_ratio_high'] == .2
    assert rl['worker']['actor']['clip_ratio_dual'] == 0.


@pytest.mark.parametrize('seed', [None, '1', '42'])
def test_preference_generation_requires_an_explicit_seed(monkeypatch, seed):
    tree = ast.parse((ROOT / 'dpo/scripts/generate_preferences.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'parse_args')
    scope = {'argparse': argparse}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<preference-args>', 'exec'), scope)
    argv = ['generate_preferences.py', '--input', 'input.jsonl', '--output-dir', 'out',
            '--model', 'model', '--processor', 'model', '--reward', 'reward.py']
    if seed is not None:
        argv += ['--seed', seed]
    monkeypatch.setattr(sys, 'argv', argv)
    if seed is None:
        with pytest.raises(SystemExit) as exc:
            scope['parse_args']()
        assert exc.value.code == 2
    else:
        assert scope['parse_args']().seed == int(seed)
