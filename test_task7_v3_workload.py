import unittest
from task7_v3_workload import selected_questions, candidate_row, validate_review, create_schedule


def make_q(n, split='selection', dependency='standalone'):
    return {'question_id':f'q{n}', 'split':split, 'context_dependency':dependency,
        'conversation_history':[], 'course_unit':'trees', 'question_type':'why_question',
        'user_utterance':f'What is example {n}?', 'canonical_question':f'example {n}',
        'reference_answer':f'Correct reference {n}', 'reference_key_points':['a'],
        'retrieval':{'condition':'relevant','source':{'filename':'course.pdf','pages':[1], 'evidence_summary':'example'}}}

class Task7V3Tests(unittest.TestCase):
    def setUp(self):self.questions=[make_q(i) for i in range(3)]
    def test_split_and_context_filter(self):
        qs=self.questions+[make_q(4,split='calibration'),make_q(5,dependency='followup')]
        self.assertEqual(len(selected_questions(qs)),3)
    def test_derived_schedule_balanced_and_unique(self):
        rows=create_schedule(selected_questions(self.questions))
        self.assertEqual(len(rows),24)
        self.assertEqual(len({r['trial_id'] for r in rows}),24)
        self.assertEqual(len({r['block_id'] for r in rows}),8)
        self.assertEqual(sum(r['condition']=='off' for r in rows),12)
        self.assertEqual(sum(r['condition']=='on' for r in rows),12)
    def test_review_fails_closed(self):
        rows=[candidate_row(q) for q in self.questions]
        self.assertEqual(len(validate_review(self.questions,rows)),9)
    def test_review_with_provenance_passes(self):
        rows=[dict(candidate_row(q),review_status='VERIFIED',reviewer='Reviewer A',review_notes='Checked source slide 1 against the answer.') for q in self.questions]
        self.assertEqual(validate_review(self.questions,rows),[])
    def test_frozen_change_rejected(self):
        rows=[dict(candidate_row(q),review_status='VERIFIED',reviewer='Reviewer A',review_notes='Checked source slide 1 against the answer.') for q in self.questions]
        rows[0]['reference_answer']='changed'
        self.assertTrue(any('frozen-source mismatch' in x for x in validate_review(self.questions,rows)))
    def test_duplicate_utterance_rejected(self):
        self.questions[1]['user_utterance']=self.questions[0]['user_utterance']
        with self.assertRaises(ValueError):selected_questions(self.questions)

if __name__=='__main__':unittest.main()
