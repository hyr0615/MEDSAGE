"""Regression coverage for negated, ambiguous and contradictory answers."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rewards'))
from accuracy_reward import accuracy_reward, explicit_mc_choice
from combined_reward import compute_score
from stage_final_correctness import stage_c_correctness


class AnswerRewardTests(unittest.TestCase):
    def test_retractions_and_answer_level_denials(self):
        for choice in ['Yes', 'No', 'B']:
            for tail in ['This answer is false.', 'That answer was incorrect.',
                         'My answer is not true.', "This answer isn't correct.",
                         'That is false.', 'I retract my answer.',
                         'I withdraw that selection.', 'I revise my choice.']:
                with self.subTest(choice=choice, tail=tail):
                    self.assertEqual(accuracy_reward(choice + '. ' + tail, choice), 0.0)

    def test_mc_rejects_explicit_reassignments(self):
        for first in 'ABCD':
            for other in 'ABCD':
                if other == first:
                    continue
                for tail in [f'The correct option is {other}.', f'The answer: ({other}).',
                             f'The choice should be {other}.', f'Actually {other}.',
                             f'I select option {other}.', f'Correction: {other.lower()}.']:
                    with self.subTest(first=first, tail=tail):
                        self.assertEqual(accuracy_reward(first + '. ' + tail, first), 0.0)

    def test_answer_guards_preserve_unambiguous_explanations(self):
        for pred, ref in [('Yes. This answer is correct.', 'yes'),
                          ('No. False positives are possible.', 'no'),
                          ('B. A false positive result.', 'B'),
                          ('B. The correct option is B.', 'B'),
                          ('B. I select option B.', 'B'),
                          ('B. Actually a lung lesion.', 'B'),
                          ('B. I select a region in the lung.', 'B'),
                          ('B. A lung lesion.', 'B')]:
            self.assertEqual(accuracy_reward(pred, ref), 1.0, pred)

    def test_contradiction_reduces_full_reward_and_stage_c_credit(self):
        for answer, bad in [('B', 'B. The correct option is C.'),
                            ('yes', 'Yes. This answer is false.')]:
            gt = {'answer': answer, 'full_LVKC': True, 'reference_boxes': [[0, 0, 10, 10]]}
            template = ('<LOC>{"bbox_xyxy_pixel":[[0,0,10,10]]}</LOC>'
                        '<VIS>Finding.</VIS><KNO>Relation.</KNO><CON>{}</CON>')
            good, wrong = compute_score([{'response': template.replace('{}', text), 'ground_truth': gt}
                                         for text in [answer, bad]])
            self.assertAlmostEqual(good['overall'], 1.0)
            self.assertEqual(wrong['accuracy'], 0.0)
            self.assertEqual(wrong['stage_c_correctness'], 0.0)
            self.assertAlmostEqual(wrong['overall'], 0.4)

    def test_numbers_and_polarity_symbols_remain_distinct(self):
        for pred, ref in [('1.5', '15'), ('ER+', 'ER-'), ('1/2', '12'), ('-1', '1'), ('pneumonia?', 'pneumonia')]:
            with self.subTest(pred=pred, ref=ref):
                self.assertEqual(accuracy_reward(pred, ref), 0.0)
        for answer in ['1.5', 'ER+', 'ER-', 'Brain tumor']:
            self.assertEqual(accuracy_reward(answer + '.', answer), 1.0)

    def test_explicit_self_contradiction_does_not_score(self):
        for pred in ['Yes, that answer is incorrect.', 'Yes, not correct.', 'No, perhaps.', 'a. b.']:
            self.assertEqual(accuracy_reward(pred, 'A' if pred.startswith('a') else pred.split(',')[0].lower()), 0.0)

    def test_mc_explanatory_article_is_not_a_second_choice(self):
        for pred in ['B. A lung lesion.', 'B. A diagnosis.', '(b) a finding.']:
            self.assertEqual(accuracy_reward(pred, 'B'), 1.0, pred)
        for pred in ['b. a.', 'B. option a', 'B. A is correct', '(B) (A)', 'B. and a']:
            self.assertEqual(accuracy_reward(pred, 'B'), 0.0, pred)

    def test_malformed_and_nonterminal_stages_do_not_become_answers(self):
        for pred in ['<CON>pneumonia', 'pneumonia</CON>', '<CON>pneumonia</CON></CON>',
                     '<VIS>pneumonia</VIS>', '<CON><VIS>pneumonia</VIS></CON>',
                     '<CON>pneumonia</CON> but not pneumonia']:
            self.assertEqual(accuracy_reward(pred, 'pneumonia'), 0.0, pred)

    def test_dict_ground_truth_and_invalid_reference_types(self):
        self.assertEqual(accuracy_reward('yes', {'answer': 'yes'}), 1.0)
        for answer in [None, [], {'text': 'yes'}, 1]:
            self.assertEqual(accuracy_reward('yes', {'answer': answer}), 0.0)

    def test_binary_rejects_negation_and_ambiguity(self):
        for pred in ['not yes', 'yes or no', 'Yes. No.', 'maybe yes', 'Yes is not correct', 'Yes, perhaps.']:
            with self.subTest(pred=pred):
                self.assertEqual(accuracy_reward(pred, 'yes'), 0.0)

    def test_binary_accepts_explicit_choices(self):
        for pred, ref in [('yes', 'yes'), ('Yes, a finding is present.', 'yes'),
                          ('No. No lesion is visible.', 'no'), ('The answer is yes.', 'yes')]:
            self.assertEqual(accuracy_reward(pred, ref), 1.0)
        self.assertEqual(accuracy_reward('no', 'yes'), 0.0)

    def test_open_answers_do_not_use_lexical_overlap(self):
        for pred in ['no pneumonia', 'not pneumonia', 'pneumonia is absent',
                     'pneumonia or edema', 'pneumonia but no pneumonia', 'possible pneumonia']:
            with self.subTest(pred=pred):
                self.assertEqual(accuracy_reward(pred, 'pneumonia'), 0.0)
        self.assertEqual(accuracy_reward('pneumonia', 'no pneumonia'), 0.0)
        self.assertEqual(accuracy_reward('No pneumonia.', 'no pneumonia'), 1.0)
        self.assertEqual(accuracy_reward('The answer is pneumonia.', 'pneumonia'), 1.0)

    def test_mc_rejects_alternatives_and_negation(self):
        for pred in ['not A', 'A or B', 'A. B.', 'A, not A', 'Answer: B', 'A. or b']:
            with self.subTest(pred=pred):
                self.assertEqual(accuracy_reward(pred, 'A'), 0.0)

    def test_mc_accepts_one_explicit_option(self):
        for pred in ['A', '(A)', '(A) Pneumonia', 'A. Pneumonia', 'Option A', 'Answer: A', 'a']:
            self.assertEqual(explicit_mc_choice(pred), 'A', pred)
            self.assertEqual(accuracy_reward(pred, 'A'), 1.0, pred)
        self.assertIsNone(explicit_mc_choice('A lung mass is visible.'))

    def test_stage_c_and_trajectory_agree(self):
        for ref, pred in [('A', 'A or B'), ('yes', 'not yes'), ('pneumonia', 'no pneumonia'), ('A', 'A')]:
            gt = json.dumps({'answer': ref, 'full_LVKC': True, 'task_type': 'full_LVKC'})
            response = '<CON>' + pred + '</CON>'
            self.assertEqual(accuracy_reward(response, gt), stage_c_correctness(response, gt))

    def test_duplicate_or_missing_conclusions_are_not_answers(self):
        gt = json.dumps({'answer': 'yes', 'full_LVKC': True})
        for pred in ['<VIS>yes</VIS>', '<CON>yes</CON><CON>no</CON>']:
            self.assertEqual(accuracy_reward(pred, gt), 0.0)

    def test_shared_rl_dpo_reward_ranks_contradiction_below_correct(self):
        gt = json.dumps({'answer': 'pneumonia', 'full_LVKC': True, 'reference_boxes': [[0, 0, 10, 10]]})
        template = '<LOC>{"bbox_xyxy_pixel":[[0,0,10,10]]}</LOC><VIS>Finding.</VIS><KNO>Relation.</KNO><CON>{}</CON>'
        good, bad = compute_score([{'response': template.replace('{}', answer), 'ground_truth': gt}
                                   for answer in ['pneumonia', 'no pneumonia']])
        self.assertEqual(good['accuracy'], 1.0)
        self.assertEqual(bad['accuracy'], 0.0)
        self.assertGreater(good['overall'], bad['overall'])


if __name__ == '__main__':
    unittest.main()
