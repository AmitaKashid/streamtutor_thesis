"""Measure baseline and candidate prompts using a llama-server tokenizer.

Makes metadata, template and tokenization requests only; no model generation.
Results use a fixed synthetic question and course snippet, with thinking off.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prompts.common import PROMPT_SET_VERSION
from prompts.registry import compose_prompt, get_style_prompt
from scripts.context_diagnostics.prefix_probe import tokenized_prompt

BASELINE = ROOT / 'data/prompt_evaluation/baseline_v0.json'
QUESTION = 'What is a decision tree?'
CONTEXT = '[Synthetic course snippet] A decision tree uses tests on features to partition data into branches.'
STYLES = ('socratic', 'inquiry', 'adaptive', 'collaborative')


def compose_baseline(text: str) -> str:
    return '\n\n'.join(('[SYSTEM]\n' + text.strip(), '[CONTEXT]\n' + CONTEXT,
                         '[USER]\n' + QUESTION, '[ASSISTANT]\n'))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True, help='llama-server root URL, for example http://127.0.0.1:8008')
    parser.add_argument('--output', help='New JSON output file; never overwritten')
    args = parser.parse_args()
    try:
        baseline = json.loads(BASELINE.read_text(encoding='utf-8'))
        response = requests.get(args.url.rstrip('/') + '/v1/models', timeout=(5, 10))
        response.raise_for_status()
        model_data = response.json().get('data') or []
        model_ids = [r.get('id') for r in model_data if isinstance(r, dict)]
        if not model_ids or any(not isinstance(model_id, str) for model_id in model_ids):
            raise ValueError('Endpoint did not return a valid model identity')
        rows = []
        for style in STYLES:
            for version, instruction, prompt in (
                (baseline['version'], baseline['style_prompts'][style], compose_baseline(baseline['style_prompts'][style])),
                (PROMPT_SET_VERSION, get_style_prompt(style), compose_prompt(style=style, user_query=QUESTION, context=CONTEXT)),
            ):
                tokens = tokenized_prompt(args.url, prompt)
                row = {'style': style, 'version': version, 'instruction_words': len(instruction.split()),
                       'instruction_chars': len(instruction), 'full_prompt_tokens': len(tokens),
                       'prompt_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest()}
                rows.append(row)
                print(f"{style:13} {version:18} words={row['instruction_words']:3} "
                      f"chars={row['instruction_chars']:4} full_prompt_tokens={len(tokens)}")
        output = {'schema': 'teaching-prompt-token-count-v1', 'model_ids_reported': model_ids,
                  'question': QUESTION, 'rag_context': CONTEXT, 'history': [],
                  'thinking_enabled': False, 'tokenization_method': 'apply-template then tokenize; add_special=true, parse_special=true',
                  'rows': rows}
        if args.output:
            with open(args.output, 'x', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2); f.write('\n')
            print(f'Saved {args.output}')
        print('Verify token counts against a generation request; provider, template, model and settings may differ.')
        return 0
    except (requests.RequestException, ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        print(f'Measurement failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
