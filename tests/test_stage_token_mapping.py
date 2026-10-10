"""Dependency-free tests of alignment, including noncanonical tokenization."""
import ast
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'easyr1/verl/trainer'))
from stage_token_mapping import assign_token_stages, decode_with_offsets


def stage_spans(text):
    tree = ast.parse((ROOT / 'easyr1/verl/trainer/ray_trainer.py').read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_sagrpo_stage_char_spans')
    node.decorator_list = []
    namespace = {'re': re}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<stage-parser>', 'exec'), namespace)
    return namespace[node.name](text)


class FragmentTokenizer:
    is_fast = False

    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids, **kwargs):
        return ''.join(self.pieces[i] for i in ids)


class StageTokenMappingTests(unittest.TestCase):
    def test_context_merged_tags_and_content(self):
        tokenizer = FragmentTokenizer(['<LOC>{', 'box', '}</LOC><VIS>Finding', '</VIS><KNO>Relation', '</KNO><CON>yes', '</CON>'])
        text, offsets = decode_with_offsets(tokenizer, list(range(6)))
        self.assertEqual(assign_token_stages(offsets, stage_spans(text)), [1, 1, 2, 3, 4, 0])

    def test_tags_and_outside_tokens_are_neutral(self):
        tokenizer = FragmentTokenizer(['prefix', '<VIS>', 'finding', '</VIS>', 'suffix'])
        text, offsets = decode_with_offsets(tokenizer, list(range(5)))
        self.assertEqual(assign_token_stages(offsets, stage_spans(text)), [0, 0, 2, 0, 0])

    def test_case_whitespace_and_repeated_stages(self):
        text = '<vis> one </vis>\n<VIS>two</VIS><KNO> </KNO>'
        offsets = [(i, i + 1) for i in range(len(text))]
        stages = assign_token_stages(offsets, stage_spans(text))
        self.assertEqual(stages.count(2), 6)
        self.assertNotIn(3, stages)

    def test_empty_and_incomplete_stages(self):
        for text in ['', '<VIS>', '<VIS>unfinished', '<CON></CON>']:
            offsets = [(i, i + 1) for i in range(len(text))]
            self.assertFalse(any(assign_token_stages(offsets, stage_spans(text))))

    def test_nested_model_output_does_not_crash_mapping(self):
        text = '<VIS><CON>yes</CON></VIS>'
        offsets = [(i, i + 1) for i in range(len(text))]
        stages = assign_token_stages(offsets, stage_spans(text))
        self.assertNotIn(2, stages)
        self.assertEqual(stages.count(4), 3)

    def test_noncanonical_ids_use_actual_prefixes(self):
        class Noncanonical(FragmentTokenizer):
            is_fast = True

            def __call__(self, text, **kwargs):
                return {'input_ids': [999], 'offset_mapping': [(0, len(text))]}

        tokenizer = Noncanonical(['<VIS>', 'finding', '</VIS>'])
        text, offsets = decode_with_offsets(tokenizer, [0, 1, 2])
        self.assertEqual(assign_token_stages(offsets, stage_spans(text)), [0, 2, 0])

    def test_partial_utf8_bytes_share_content_span(self):
        class ByteTokenizer:
            is_fast = False

            def decode(self, ids, **kwargs):
                return bytes(ids).decode('utf-8', errors='replace')

        ids = list('<VIS>\u80ba</VIS>'.encode('utf-8'))
        text, offsets = decode_with_offsets(ByteTokenizer(), ids)
        self.assertEqual(assign_token_stages(offsets, stage_spans(text)).count(2), 3)

    def test_alignment_failures_are_explicit(self):
        class Unstable(FragmentTokenizer):
            def decode(self, ids, **kwargs):
                return 'wrong' if len(ids) == 1 else '<VIS>finding</VIS>'

        with self.assertRaisesRegex(ValueError, 'prefix-stable'):
            decode_with_offsets(Unstable([]), [0, 1])
        with self.assertRaisesRegex(ValueError, 'no aligned tokens'):
            assign_token_stages([], stage_spans('<VIS>finding</VIS>'))


if __name__ == '__main__':
    unittest.main()
