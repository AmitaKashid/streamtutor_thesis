"""Probe llama-server prompt processing and prefix reuse with synthetic dialogue.

This is an opt-in engineering probe, not an answer-quality benchmark. It sends
requests to the specified existing server. Keep other traffic off that server
for a controlled measurement and run it inside an authorised allocation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.inference_metrics import server_timing_metrics
from prompts.registry import compose_prompt
from app.conversation_context import contextual_user_query

SYSTEM = ('You are an AI tutor for data mining. Help the student understand '
          'clearly, using provided course context when available.')
CONTEXT = ('[synthetic slide | Clustering]\nClustering is an unsupervised method '
           'for finding groups of similar observations. DBSCAN groups dense '
           'regions and can label sparse observations as noise.')
TURNS = [
    ('What is data mining?', 'It extracts useful patterns from data.'),
    ('What is unsupervised learning?', 'It finds patterns without labeled targets.'),
    ('What are data points?', 'They are individual observations in a dataset.'),
    ('What are features?', 'They are properties used to describe an observation.'),
    ('What is similarity?', 'It describes how alike two observations are.'),
    ('What is density?', 'It describes how closely points are packed together.'),
]
QUESTION = 'What is DBSCAN? Answer briefly.'
CHANGED_QUESTION = 'What does DBSCAN label as noise? Answer briefly.'


def make_prompt(turns, question=QUESTION):
    if turns:
        lines = []
        for i, (user, answer) in enumerate(turns, 1):
            lines.extend([f'Turn {i}', f'User: {user}', f'Assistant: {answer}', ''])
        history = '\n'.join(lines).strip()
        question = contextual_user_query(question, history)
    return compose_prompt(style='socratic', user_query=question, context=CONTEXT)


def request(url, prompt, *, cache_prompt):
    payload = {'messages': [{'role': 'system', 'content': SYSTEM},
                            {'role': 'user', 'content': prompt}],
               'stream': True, 'max_tokens': 128, 'temperature': 0,
               'cache_prompt': cache_prompt,
               'chat_template_kwargs': {'enable_thinking': False},
               'reasoning_format': 'deepseek'}
    started = time.perf_counter()
    first_token = None
    finish = None
    metrics = {}
    unexpected_reasoning = False
    saw_done = False
    with requests.post(url.rstrip('/') + '/v1/chat/completions', json=payload,
                       stream=True, timeout=(5, 300)) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=False):
            if not line or not line.startswith(b'data:'):
                continue
            data = line[5:].strip()
            if data == b'[DONE]':
                saw_done = True
                break
            event = json.loads(data.decode('utf-8'))
            if event.get('error'):
                raise RuntimeError('The server returned a streaming error')
            metrics.update(server_timing_metrics(event.get('timings')))
            for choice in event.get('choices') or []:
                delta = choice.get('delta') or {}
                if delta.get('reasoning_content'):
                    unexpected_reasoning = True
                if first_token is None and (delta.get('content') or delta.get('reasoning_content')):
                    first_token = time.perf_counter() - started
                finish = choice.get('finish_reason') or finish
    return {'wall_seconds': time.perf_counter() - started, 'first_output_seconds': first_token,
            'finish_reason': finish, 'stream_done': saw_done,
            'unexpected_reasoning': unexpected_reasoning, **metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True, help='Local tunnel to one llama-server, e.g. http://127.0.0.1:8008')
    parser.add_argument('--model-label', required=True, help='Model label; verify endpoint identity separately')
    parser.add_argument('--output', required=True, help='New JSONL output path; existing files are not overwritten')
    parser.add_argument('--trials', type=int, default=1, help='Repetitions per condition (default: 1)')
    parser.add_argument('--seed', type=int, default=42, help='Random order seed')
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('--trials must be at least 1')
    schedule = [(turns, trial) for trial in range(args.trials) for turns in (0, 1, 3, 5)]
    random.Random(args.seed).shuffle(schedule)
    print('Synthetic probe: 4–6 calls per condition; results are not a quality evaluation.', flush=True)
    failures = []
    missing_timings = []
    incomplete = []
    with open(args.output, 'x', encoding='utf-8') as out:
        def run(turns, trial, phase, conversation, question, cache):
            prompt = make_prompt(conversation, question)
            row = {'schema': 'context-prefill-probe-v1', 'model_label': args.model_label,
                   'url': args.url, 'history_turns': turns, 'trial': trial,
                   'phase': phase, 'cache_prompt': cache, 'seed': args.seed,
                   'prompt_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                   'prompt_chars': len(prompt), 'rag': 'fixed_synthetic',
                   'thinking_enabled': False, 'temperature': 0, 'max_output_tokens': 128}
            try:
                row.update(request(args.url, prompt, cache_prompt=cache))
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                row.update({'error_type': type(exc).__name__, 'stream_done': False})
            if not row.get('stream_done'):
                failures.append(phase)
            elif row.get('finish_reason') != 'stop' or row.get('unexpected_reasoning'):
                incomplete.append(phase)
            if phase not in ('prime', 'prime_sliding') and 'serverPrefillSeconds' not in row:
                missing_timings.append(phase)
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
            out.flush()
            print(f"turns={turns} trial={trial} {phase}: "
                  f"prefill={row.get('serverPrefillSeconds')}s "
                  f"cached={row.get('serverCachedTokens')} "
                  f"decode={row.get('serverDecodeSeconds')}s "
                  f"finish={row.get('finish_reason', row.get('error_type', 'missing'))}", flush=True)
            return row

        for n, trial in schedule:
            history = TURNS[:n]
            run(n, trial, 'cold', history, QUESTION, False)
            prime = run(n, trial, 'prime', history, QUESTION, True)
            if not prime.get('stream_done'):
                continue
            run(n, trial, 'warm_identical', history, QUESTION, True)
            run(n, trial, 'warm_new_question', history, CHANGED_QUESTION, True)
            if n:
                # Reprime the original and compare a sliding history window.
                run(n, trial, 'prime_sliding', history, QUESTION, True)
                run(n, trial, 'warm_sliding_history', TURNS[1:n + 1], QUESTION, True)
    print(f'Saved {args.output}. Inspect finish_reason and missing timing fields before analysis.')
    if incomplete:
        print(f'Warning: {len(incomplete)} streams were truncated or returned unexpected reasoning.')
    if failures or missing_timings:
        print(f'Probe incomplete: failed streams={len(failures)}, measured rows missing timing={len(missing_timings)}.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
