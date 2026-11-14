"""Run a paired, human-reviewed context study against one llama-server endpoint.

Inputs and run metadata are saved with every response. Cache-on conditions are
immediately primed with an identical request; reported cache hits must still be
checked. Quality is deliberately left for blinded human review.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.context_diagnostics.quality_pilot import generate, make_prompt, TEACHING_STYLE
from scripts.context_diagnostics.verify_eval_endpoint import check_endpoint

REQUIRED_COUNTERS = ('serverPromptTokens', 'serverCachedTokens', 'serverPrefillSeconds',
                     'serverOutputTokens', 'serverDecodeSeconds')


def load_inputs(cases_path, manifest_path):
    cases_raw = Path(cases_path).read_bytes()
    cases = [json.loads(line) for line in cases_raw.decode('utf-8').splitlines() if line.strip()]
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    for key in ('model_filename', 'model_sha256', 'gpu', 'server_version', 'server_configuration', 'load_description', 'reviewer'):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError(f'Manifest requires {key}')
    if len(manifest['model_sha256']) != 64 or any(c not in '0123456789abcdefABCDEF' for c in manifest['model_sha256']):
        raise ValueError('Manifest model_sha256 must be a SHA-256 hex digest')
    ids = set(); categories = set()
    for case in cases:
        qid = case.get('question_id')
        distance = case.get('required_history_distance')
        if not isinstance(qid, str) or not qid or qid in ids or distance not in (0, 1, 3, 5):
            raise ValueError('Cases require unique IDs and a history distance of 0, 1, 3 or 5')
        ids.add(qid); categories.add(distance)
        history = case.get('conversation_history')
        if not isinstance(history, list) or len(history) != 10:
            raise ValueError(f'{qid}: provide five prior user/assistant turns')
        if any(not isinstance(m, dict) or m.get('role') != ('user' if i % 2 == 0 else 'assistant')
               or not isinstance(m.get('content'), str) or not m['content'].strip() for i, m in enumerate(history)):
            raise ValueError(f'{qid}: invalid conversation roles or text')
        if not case.get('reviewed_by') or not case.get('user_utterance') or not case.get('reference_key_points'):
            raise ValueError(f'{qid}: reviewer, question and reference are required')
        if not isinstance(case.get('retrieval'), dict) or not isinstance(case['retrieval'].get('context'), str):
            raise ValueError(f'{qid}: fixed retrieval context is required')
    if categories != {0, 1, 3, 5}:
        raise ValueError('Include reviewed standalone cases and cases requiring turns 1, 3, and 5 back')
    return cases, manifest, hashlib.sha256(cases_raw).hexdigest()


def schedule(cases, repetitions, seed):
    tasks = [(case, limit, cache, trial) for case in cases for limit in (0, 1, 3, 5)
             for cache in (False, True) for trial in range(repetitions)]
    random.Random(seed).shuffle(tasks)
    return tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ('url', 'cases', 'manifest', 'output'): ap.add_argument('--' + arg, required=True)
    ap.add_argument('--repetitions', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-output-tokens', type=int, default=1024)
    args = ap.parse_args()
    if args.repetitions < 1 or args.max_output_tokens < 1: ap.error('repetitions and output budget must be positive')
    try:
        cases, manifest, cases_hash = load_inputs(args.cases, args.manifest)
        models = check_endpoint(args.url, manifest['model_filename'])
        tasks = schedule(cases, args.repetitions, args.seed)
        failures = 0
        with open(args.output, 'x', encoding='utf-8') as out:
            for index, (case, limit, cache, trial) in enumerate(tasks, 1):
                prompt, actual = make_prompt(case, limit)
                row = {'schema': 'context-study-v1', 'case': case, 'case_sha256': cases_hash,
                       'manifest': manifest, 'observed_model_ids': models, 'history_limit': limit,
                       'history_turns_sent': actual, 'trial': trial, 'seed': args.seed,
                       'thinking_requested': False, 'teaching_style': TEACHING_STYLE, 'temperature': 0,
                       'cache_prompt_requested': cache, 'max_output_tokens': args.max_output_tokens,
                       'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
                try:
                    if cache:
                        prime = generate(args.url, prompt, cache_prompt=True, max_tokens=args.max_output_tokens)
                        row['prime_complete'] = bool(prime.get('stream_done') and prime.get('finish_reason') == 'stop'
                                                     and not prime.get('unexpected_reasoning') and prime.get('answer', '').strip())
                    row.update(generate(args.url, prompt, cache_prompt=cache, max_tokens=args.max_output_tokens))
                except Exception as exc:
                    row['error_type'] = type(exc).__name__
                row['complete'] = bool(row.get('stream_done') and row.get('finish_reason') == 'stop'
                                       and not row.get('unexpected_reasoning') and row.get('answer', '').strip())
                row['metrics_complete'] = (all(row.get(k) is not None for k in REQUIRED_COUNTERS)
                                           and row.get('first_answer_seconds') is not None)
                failures += not row['complete'] or not row['metrics_complete'] or (cache and not row.get('prime_complete'))
                out.write(json.dumps(row, ensure_ascii=False) + '\n'); out.flush()
                print(f"{index}/{len(tasks)} {case['question_id']} history={limit} cache={cache} complete={row['complete']} metrics={row['metrics_complete']}", flush=True)
        print(f'Saved {args.output}; unusable={failures}. Review answers and cache counts before analysis.')
        return 1 if failures else 0
    except Exception as exc:
        print(f'Study stopped: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1

if __name__ == '__main__': raise SystemExit(main())
