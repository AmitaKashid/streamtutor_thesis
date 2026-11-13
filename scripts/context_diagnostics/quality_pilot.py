"""Paired context-history pilot using the existing judge selection dataset.

Exploratory only: curated evidence is fixed, model answers are saved for human
review, and this does not measure the live RAG retrieval pipeline.
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
from scripts.context_diagnostics.probe import SYSTEM

DEFAULT_DATA = ROOT / 'data/judge_evaluation/streamtutor_judge_dataset_v3/questions.jsonl'


def selected_cases(path):
    """Select two standalone and three each with one and two prior turns."""
    groups = {0: [], 1: [], 2: []}
    with open(path, encoding='utf-8') as source:
        for line in source:
            item = json.loads(line)
            hist = item['conversation_history']
            if (item['split'] != 'selection' or item['retrieval']['condition'] != 'relevant'
                    or item['context_dependency'] == 'visual_deictic' or len(hist) % 2):
                continue
            n = len(hist) // 2
            if n not in groups or any(hist[i]['role'] != ('user' if i % 2 == 0 else 'assistant') for i in range(len(hist))):
                continue
            if n == 0 and item['context_dependency'] != 'standalone':
                continue
            if n > 0 and 'follow_up' not in item['context_dependency'] and item['context_dependency'] != 'pronoun_and_history':
                continue
            groups[n].append(item)
    counts = {0: 2, 1: 3, 2: 3}
    if any(len(groups[n]) < counts[n] for n in groups):
        raise ValueError('Not enough eligible selection examples in the dataset')
    return [case for n in groups for case in groups[n][:counts[n]]]


def make_prompt(case, history_limit):
    history = case['conversation_history']
    turns = list(zip(history[::2], history[1::2]))
    selected = turns[-history_limit:] if history_limit else []
    question = case['user_utterance'].strip()
    if selected:
        blocks = [f"Turn {i}\nUser: {u['content'].strip()}\nAssistant: {a['content'].strip()}"
                  for i, (u, a) in enumerate(selected, 1)]
        question = ('Previous conversation between the user and assistant:\n' + '\n\n'.join(blocks)
                    + '\n\nCurrent user question:\n' + question
                    + '\n\nUse the previous conversation as context. Answer the current user question directly.')
    prompt = compose_prompt(style=case['teaching_style'], user_query=question,
                            context=case['retrieval']['context'])
    return prompt, len(selected)


def generate(url, prompt):
    body = {'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}],
            'stream': True, 'max_tokens': 1024, 'temperature': 0, 'cache_prompt': False,
            'chat_template_kwargs': {'enable_thinking': False}, 'reasoning_format': 'deepseek'}
    started = time.perf_counter(); first = None; pieces = []; done = False
    finish = None; metrics = {}; reasoning_seen = False
    with requests.post(url.rstrip('/') + '/v1/chat/completions', json=body,
                       stream=True, timeout=(5, 300)) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=False):
            if not line or not line.startswith(b'data:'):
                continue
            data = line[5:].strip()
            if data == b'[DONE]':
                done = True; break
            event = json.loads(data.decode('utf-8'))
            if event.get('error'):
                raise RuntimeError('Generation stream returned an error')
            metrics.update(server_timing_metrics(event.get('timings')))
            for choice in event.get('choices') or []:
                delta = choice.get('delta') or {}
                reasoning_seen |= bool(delta.get('reasoning_content'))
                content = delta.get('content') or ''
                if content:
                    if first is None and content.strip(): first = time.perf_counter() - started
                    pieces.append(content)
                finish = choice.get('finish_reason') or finish
    return {'answer': ''.join(pieces), 'first_answer_seconds': first,
            'wall_seconds': time.perf_counter() - started, 'finish_reason': finish,
            'stream_done': done, 'unexpected_reasoning': reasoning_seen, **metrics}


def build_schedule(cases, seed):
    # Only conditions with distinct included histories; no invented 3/5-turn data.
    schedule = [(case, n) for case in cases for n in range(len(case['conversation_history']) // 2 + 1)]
    random.Random(seed).shuffle(schedule)
    return schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--model-label', required=True, help='User-supplied label, not verified identity')
    parser.add_argument('--output', required=True, help='New JSONL file; will not overwrite')
    parser.add_argument('--dataset', default=str(DEFAULT_DATA))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    try:
        cases = selected_cases(args.dataset)
        raw = Path(args.dataset).read_bytes()
        dataset_hash = hashlib.sha256(raw).hexdigest()
        schedule = build_schedule(cases, args.seed)
        failures = 0
        with open(args.output, 'x', encoding='utf-8') as out:
            print(f'Exploratory pilot: {len(cases)} examples, {len(schedule)} requests. Fixed curated evidence; cache requested off.', flush=True)
            for case, limit in schedule:
                prompt, actual = make_prompt(case, limit)
                row = {'schema': 'context-quality-pilot-v1', 'question_id': case['question_id'],
                       'conversation_id': case['conversation_id'], 'context_dependency': case['context_dependency'],
                       'history_limit': limit, 'history_turns_sent': actual, 'model_label': args.model_label,
                       'dataset_sha256': dataset_hash, 'seed': args.seed, 'rag_mode': 'fixed_curated_evidence',
                       'thinking_enabled': False, 'temperature': 0, 'cache_prompt_requested': False,
                       'max_output_tokens': 1024, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                       'question': case['user_utterance'], 'full_history': case['conversation_history'],
                       'evidence': case['retrieval']['context'], 'reference_key_points': case['reference_key_points']}
                try:
                    row.update(generate(args.url, prompt))
                except (requests.RequestException, ValueError, RuntimeError) as exc:
                    row.update({'error_type': type(exc).__name__, 'stream_done': False})
                valid = (row.get('stream_done') and row.get('finish_reason') == 'stop'
                         and not row.get('unexpected_reasoning') and bool(row.get('answer', '').strip()))
                row['complete'] = bool(valid)
                required_metrics = ('serverPromptTokens', 'serverCachedTokens', 'serverPrefillSeconds',
                                    'serverOutputTokens', 'serverDecodeSeconds')
                row['metrics_complete'] = all(k in row for k in required_metrics)
                failures += not valid or not row['metrics_complete']
                out.write(json.dumps(row, ensure_ascii=False) + '\n'); out.flush()
                print(f"{case['question_id']} history={actual} finish={row.get('finish_reason', row.get('error_type'))} complete={bool(valid)} metrics={row['metrics_complete']}", flush=True)
        print(f'Saved {args.output}; incomplete={failures}. Human review required.')
        return 1 if failures else 0
    except (OSError, KeyError, ValueError) as exc:
        print(f'Pilot could not continue: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
