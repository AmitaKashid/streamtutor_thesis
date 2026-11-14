"""Tests for paired pilot selection, prompt wiring, and output capture."""
import json
import unittest
from unittest.mock import patch
from scripts.context_diagnostics import quality_pilot as pilot
from test_context_measurements import load_memory_class
from prompts.registry import compose_prompt


class Response:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def raise_for_status(self): pass
    def iter_lines(self, decode_unicode=False):
        events = [
            {'choices': [{'delta': {'content': 'First '}, 'finish_reason': None}]},
            {'choices': [{'delta': {'content': 'answer.'}, 'finish_reason': None}]},
            {'choices': [{'delta': {}, 'finish_reason': 'stop'}],
             'timings': {'cache_n': 0, 'prompt_n': 50, 'prompt_ms': 120}},
        ]
        for event in events: yield ('data: ' + json.dumps(event)).encode()
        yield b'data: [DONE]'


class QualityPilotTests(unittest.TestCase):
    def test_selected_cases_have_distinct_valid_conditions(self):
        cases = pilot.selected_cases(pilot.DEFAULT_DATA)
        self.assertEqual([len(c['conversation_history']) // 2 for c in cases], [0, 0, 1, 1, 1, 2, 2, 2])
        schedule = pilot.build_schedule(cases, 42)
        self.assertEqual(len(schedule), 17)
        self.assertEqual(len({(c['question_id'], n) for c, n in schedule}), 17)
        self.assertEqual([(c['question_id'], n) for c, n in schedule],
                         [(c['question_id'], n) for c, n in pilot.build_schedule(cases, 42)])

    def test_prompt_matches_real_memory_context_format(self):
        case = next(c for c in pilot.selected_cases(pilot.DEFAULT_DATA) if len(c['conversation_history']) == 4)
        for limit in (0, 1, 2):
            memory = load_memory_class()()
            for user, assistant in zip(case['conversation_history'][::2], case['conversation_history'][1::2]):
                memory.add_turn(user['content'], assistant['content'])
            turns = memory.snapshot()[-limit:] if limit else []
            history, _ = memory.format_history_snapshot(turns)
            query = memory.build_contextual_user_query(case['user_utterance'], history=history, earlier_omitted=limit < 2)
            expected = compose_prompt(style=pilot.TEACHING_STYLE, user_query=query, context=case['retrieval']['context'])
            prompt, sent = pilot.make_prompt(case, limit)
            self.assertEqual(prompt, expected)
            self.assertEqual(sent, limit)

    def test_all_selected_prompts_are_preflighted_with_supported_style(self):
        cases = pilot.selected_cases(pilot.DEFAULT_DATA)
        self.assertTrue(any(case['teaching_style'] == 'step_by_step' for case in cases))
        prepared = pilot.prepare_schedule(cases, 42)
        self.assertEqual(len(prepared), 17)
        for case, limit, prompt, actual in prepared:
            self.assertTrue(prompt)
            self.assertEqual(actual, limit)

    def test_preflight_rejects_a_bad_case_before_any_request(self):
        cases = pilot.selected_cases(pilot.DEFAULT_DATA)
        broken = [dict(case) for case in cases]
        del broken[-1]['user_utterance']
        with patch.object(pilot.requests, 'post') as post:
            with self.assertRaises(KeyError):
                pilot.prepare_schedule(broken, 42)
            post.assert_not_called()

    def test_stream_captures_text_completion_and_server_metrics(self):
        with patch.object(pilot.requests, 'post', return_value=Response()) as post:
            result = pilot.generate('http://example', 'prompt')
        self.assertEqual(result['answer'], 'First answer.')
        self.assertTrue(result['stream_done'])
        self.assertEqual(result['finish_reason'], 'stop')
        self.assertEqual(result['serverPrefillSeconds'], 0.12)
        self.assertFalse(post.call_args.kwargs['json']['cache_prompt'])
        self.assertFalse(post.call_args.kwargs['json']['chat_template_kwargs']['enable_thinking'])

if __name__ == '__main__': unittest.main()
