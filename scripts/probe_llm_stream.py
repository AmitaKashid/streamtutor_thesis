"""Observe streaming from a running tutor model endpoint.

This sends a real generation request per selected model. It is an engineering
probe, not a benchmark or a guarantee about browser/device latency.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm.llmcli.providers.remote_http import RemoteHTTPProvider
from llm.llmcli.providers.thinking_http import ThinkingHTTPProvider
from scripts.start_all_models import verify_model_identity

# These are the default local tunnel URLs from frontend_api.REMOTE_MODEL_URLS.
MODELS = {
    'llama3_8b_q4': ('http://127.0.0.1:8000', 'legacy'),
    'gemma4_26b_q3': ('http://127.0.0.1:8007', 'llama-server'),
    'qwen3_5_9b_q6': ('http://127.0.0.1:8008', 'llama-server'),
    'gemma4_e4b_thinking': ('http://127.0.0.1:8005', 'llama-server'),
    'bonsai2_9b_thinking': ('http://127.0.0.1:8006', 'llama-server'),
}
DEFAULT_QUESTION = 'Explain decision trees in three short sentences.'


def probe(model, url, protocol, *, question, thinking=False):
    if protocol == 'llama-server':
        response = requests.get(url.rstrip('/') + '/v1/models', timeout=5)
        response.raise_for_status()
        identity = verify_model_identity(model, response.json())
        print(f'  verified_model_file: {identity}', flush=True)
        provider = ThinkingHTTPProvider(url)
        stream = provider.stream_with_cancel(question, None, thinking_enabled=thinking)
    else:
        print('  model_identity: not verified by the legacy endpoint', flush=True)
        provider = RemoteHTTPProvider(url)
        stream = provider.stream(question)

    started = time.perf_counter()
    first = last = None
    count = chars = 0
    for piece in stream:
        now = time.perf_counter() - started
        if first is None:
            first = now
        last = now
        count += 1
        chars += len(piece)
        # ASCII JSON also works on Windows consoles without UTF-8 enabled.
        print(f'  [{now:8.3f}s] {json.dumps(piece, ensure_ascii=True)}', flush=True)
    finished = time.perf_counter() - started
    outcome = 'observed-multiple-deltas' if count > 1 else 'inconclusive-single-delta' if count else 'no-answer-deltas'
    print(f'  outcome: {outcome}', flush=True)
    print(f'  answer_deltas: {count} | answer_characters: {chars}', flush=True)
    if first is not None:
        print(f'  first_answer_delta_s: {first:.3f} | last_answer_delta_s: {last:.3f} | '
              f'completed_s: {finished:.3f}', flush=True)
    return outcome


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=[*MODELS, 'all'])
    parser.add_argument('--url', help='Override the endpoint URL for one model')
    parser.add_argument('--question', default=DEFAULT_QUESTION)
    parser.add_argument('--thinking', action='store_true', help='Request thinking for llama-server models')
    args = parser.parse_args(argv)
    if args.url and args.model == 'all':
        parser.error('--url can only be used with one model')
    if not args.question.strip():
        parser.error('--question must not be empty')

    selected = MODELS if args.model == 'all' else {args.model: MODELS[args.model]}
    outcomes = {}
    print('Live model probe: each selected model will generate one answer.', flush=True)
    print('Times are client-observed; multiple deltas do not rule out proxy buffering.', flush=True)
    for model, (default_url, protocol) in selected.items():
        url = (args.url or default_url).rstrip('/')
        thinking = bool(args.thinking and protocol == 'llama-server')
        print(f'\n[{model}] endpoint={url} protocol={protocol} thinking={thinking}', flush=True)
        try:
            outcomes[model] = probe(model, url, protocol, question=args.question, thinking=thinking)
        except Exception as exc:
            outcomes[model] = 'error'
            print(f'  error: {type(exc).__name__}: {exc}', flush=True)
    print('\nSummary:', flush=True)
    for model, outcome in outcomes.items():
        print(f'  {model}: {outcome}', flush=True)
    return 0 if all(value == 'observed-multiple-deltas' for value in outcomes.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
