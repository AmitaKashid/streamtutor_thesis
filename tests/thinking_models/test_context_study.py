import json
import tempfile
import unittest
from pathlib import Path
from scripts.context_diagnostics.context_study import load_inputs, schedule


class StudyDesignTest(unittest.TestCase):
    def test_draft_cannot_run_without_review_and_design_is_paired(self):
        drafts = Path('data/context_study/draft_cases.jsonl').read_text().splitlines()
        cases = [json.loads(line) for line in drafts]
        manifest = dict(model_filename='model.gguf', model_sha256='a'*64, gpu='A30',
                        server_version='x', server_configuration='x', load_description='idle', reviewer='R')
        with tempfile.TemporaryDirectory() as directory:
            cases_path, meta_path = Path(directory)/'cases.jsonl', Path(directory)/'meta.json'
            meta_path.write_text(json.dumps(manifest))
            cases_path.write_text('\n'.join(json.dumps(c) for c in cases))
            with self.assertRaisesRegex(ValueError, 'reviewer'):
                load_inputs(cases_path, meta_path)
            for case in cases: case['reviewed_by'] = 'R'
            cases_path.write_text('\n'.join(json.dumps(c) for c in cases))
            loaded, _, digest = load_inputs(cases_path, meta_path)
            self.assertEqual(len(digest), 64)
            tasks = schedule(loaded, 3, 42)
            self.assertEqual(len(tasks), 4*4*2*3)
            self.assertEqual(len({(c['question_id'], n, cache, rep) for c,n,cache,rep in tasks}), len(tasks))

if __name__ == '__main__': unittest.main()
