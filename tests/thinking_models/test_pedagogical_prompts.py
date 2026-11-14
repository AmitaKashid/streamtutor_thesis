"""Contract checks for the production teaching prompt composition."""
import json
import unittest
from pathlib import Path

from prompts import adaptive, collaborative, inquiry, socratic
from prompts.common import PROMPT_SET_VERSION, SHARED_POLICY, VOICE_POLICY
from prompts.registry import compose_prompt, get_style_prompt
from app.conversation_context import contextual_user_query


class PedagogicalPromptTests(unittest.TestCase):
    def test_four_modes_share_policy_but_have_distinct_methods(self):
        modules = (socratic, inquiry, adaptive, collaborative)
        self.assertEqual(PROMPT_SET_VERSION, 'pedagogy-v1.0.0')
        self.assertEqual(len({module.METHOD for module in modules}), 4)
        for module in modules:
            with self.subTest(module=module.__name__):
                self.assertEqual(module.SYSTEM_PROMPT.count(SHARED_POLICY), 1)
                self.assertEqual(module.SYSTEM_PROMPT.count(VOICE_POLICY), 1)
                self.assertIn(module.METHOD, module.SYSTEM_PROMPT)
                self.assertNotIn('[STATE:', module.SYSTEM_PROMPT)
                self.assertNotIn('[END]', module.SYSTEM_PROMPT)

    def test_question_history_and_rag_survive_composition(self):
        history = 'Turn 1\nUser: My dataset has 73 positive records.\nAssistant: Understood.'
        query = contextual_user_query('How many positives?', history, earlier_omitted=True)
        for style in ('socratic', 'inquiry', 'adaptive', 'collaborative'):
            with self.subTest(style=style):
                prompt = compose_prompt(style=style, user_query=query, context='A decision tree splits data.')
                self.assertIn(history, prompt)
                self.assertIn('How many positives?', prompt)
                self.assertIn('earlier messages may be missing', prompt)
                self.assertIn('[CONTEXT]\nA decision tree splits data.', prompt)
                self.assertTrue(prompt.index('[SYSTEM]') < prompt.index('[CONTEXT]') < prompt.index('[USER]'))

    def test_no_rag_does_not_create_empty_context_or_profile(self):
        p = compose_prompt(style='adaptive', user_query='Please explain entropy', context=None)
        self.assertNotIn('[CONTEXT]', p)
        self.assertIn('When no course material is supplied', p)
        self.assertNotIn('{level}', p)
        self.assertNotIn('{mastery}', p)
        self.assertIn('[USER]\nPlease explain entropy', p)

    def test_baseline_retains_original_prompt_layout(self):
        from scripts.measure_teaching_prompt_tokens import BASELINE, CONTEXT, QUESTION, compose_baseline
        baseline = json.loads(BASELINE.read_text(encoding='utf-8'))
        for style in ('socratic', 'inquiry', 'adaptive', 'collaborative'):
            instruction = baseline['style_prompts'][style]
            expected = f'[SYSTEM]\n{instruction.strip()}\n\n[CONTEXT]\n{CONTEXT}\n\n[USER]\n{QUESTION}\n\n[ASSISTANT]\n'
            self.assertEqual(compose_baseline(instruction), expected)

    def test_invalid_style_remains_explicit(self):
        with self.assertRaises(ValueError):
            get_style_prompt('unknown')

if __name__ == '__main__':
    unittest.main()
