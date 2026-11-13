"""Compare llama-server token prefixes with reported cache reuse on synthetic prompts.

Run on an idle, already-authorized server. This changes its prompt cache; it
neither clears caches nor changes server configuration. Not a quality benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.inference_metrics import server_timing_metrics
from scripts.context_diagnostics.probe import SYSTEM, TURNS, make_prompt


def shared_prefix(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def messages(prompt):
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}]


def post_json(url, path, payload):
    response = requests.post(url.rstrip('/') + path, json=payload, timeout=(5, 60))
    response.raise_for_status()
    return response.json()


def tokenized_prompt(url, prompt):
    rendered = post_json(url, '/apply-template', {
        'messages': messages(prompt), 'chat_template_kwargs': {'enable_thinking': False},
        'reasoning_format': 'deepseek'})['prompt']
    data = post_json(url, '/tokenize', {'content': rendered, 'add_special': True, 'parse_special': True})
    tokens = data['tokens']
    if not isinstance(tokens, list) or not all(type(t) is int for t in tokens):
        raise ValueError('Unexpected /tokenize response')
    return tokens


def generate(url, prompt, slot):
    payload = {'messages': messages(prompt), 'stream': True, 'max_tokens': 16,
               'temperature': 0, 'cache_prompt': True,
               'chat_template_kwargs': {'enable_thinking': False}, 'reasoning_format': 'deepseek'}
    if slot is not None:
        payload['id_slot'] = slot
    metrics, finish, done, observed_slot = {}, None, False, None
    with requests.post(url.rstrip('/') + '/v1/chat/completions', json=payload,
                       stream=True, timeout=(5, 300)) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=False):
            if not line or not line.startswith(b'data:'):
                continue
            data = line[5:].strip()
            if data == b'[DONE]':
                done = True
                break
            event = json.loads(data.decode('utf-8'))
            if event.get('error'):
                raise RuntimeError('Server returned an error during generation')
            metrics.update(server_timing_metrics(event.get('timings')))
            if type(event.get('id_slot')) is int:
                observed_slot = event['id_slot']
            for choice in event.get('choices') or []:
                finish = choice.get('finish_reason') or finish
    if not done or finish not in ('stop', 'length'):
        raise RuntimeError('Incomplete stream or missing finish reason')
    if 'serverCachedTokens' not in metrics or 'serverPromptTokens' not in metrics:
        raise RuntimeError('Server did not return prompt/cache token counts')
    return {'finish_reason': finish, 'observed_slot': observed_slot, **metrics}


def run_condition(url, n, slot):
    original = make_prompt(TURNS[:n])
    shifted = make_prompt(TURNS[1:n + 1])
    a, b = tokenized_prompt(url, original), tokenized_prompt(url, shifted)
    prefix = shared_prefix(a, b)
    prime_same = generate(url, original, slot)
    same = generate(url, original, slot)
    prime_shift = generate(url, original, slot)
    sliding = generate(url, shifted, slot)
    return {'schema': 'prefix-cache-probe-v1', 'history_turns': n,
            'requested_slot': slot, 'original_sha256': hashlib.sha256(original.encode()).hexdigest(),
            'shifted_sha256': hashlib.sha256(shifted.encode()).hexdigest(),
            'tokenized_original': len(a), 'tokenized_shifted': len(b),
            'shared_prefix_tokens': prefix,
            'prime_identical': prime_same, 'identical': same,
            'prime_shifted': prime_shift, 'shifted': sliding,
            'token_count_matches_server': (len(a) == prime_shift['serverPromptTokens'] and
                                           len(b) == sliding['serverPromptTokens'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--output', required=True, help='New output file; never overwrites')
    parser.add_argument('--slot', type=int, help='Optional requested llama-server slot ID')
    args = parser.parse_args()
    if args.slot is not None and args.slot < 0:
        parser.error('--slot must be nonnegative')
    print('Synthetic cache diagnostic: 12 short generations. Keep other model traffic idle.', flush=True)
    try:
        with open(args.output, 'x', encoding='utf-8') as out:
            for n in (1, 3, 5):
                row = run_condition(args.url, n, args.slot)
                out.write(json.dumps(row) + '\n'); out.flush()
                print(f"turns={n} shared={row['shared_prefix_tokens']} "
                      f"identical_cached={row['identical']['serverCachedTokens']} "
                      f"shifted_cached={row['shifted']['serverCachedTokens']} "
                      f"token_counts_match={row['token_count_matches_server']}", flush=True)
    except (requests.RequestException, KeyError, ValueError, RuntimeError) as exc:
        print(f'Diagnostic incomplete: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    print(f'Saved {args.output}. Slot request and template/tokenization settings require verification.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
