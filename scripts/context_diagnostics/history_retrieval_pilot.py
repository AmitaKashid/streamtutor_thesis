"""Exploratory counterfactual history test using the application's live retriever.

Two constructed dialogues differ only in an early student-provided value. Their
identical current question is retrieved once, so every condition sees identical
RAG text. These are synthetic cases, not a human-validated thesis benchmark.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import re
import sys
import time
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.context_diagnostics.quality_pilot import generate, make_prompt, TEACHING_STYLE

QUESTION = 'What were the two class percentages I gave in my first message?'


def build_case(yes, no):
    turns = [
        (f'For my decision-tree node example, {yes}% are yes and {no}% are no.',
         'I will remember those class percentages for your example.'),
        ('What is a decision-tree node?', 'It represents a subset of the training data.'),
        ('What does node impurity measure?', 'It describes how mixed the class labels are in a node.'),
        ('What is a pure node?', 'A pure node contains examples from only one class.'),
        ('How does the Gini measure describe mixing?', 'It uses one minus the sum of squared class proportions.'),
    ]
    return {'question_id': f'synthetic-gini-{yes}-{no}', 'conversation_id': f'synthetic-{yes}-{no}',
            'context_dependency': 'first_turn_recall', 'teaching_style': TEACHING_STYLE,
            'user_utterance': QUESTION, 'reference_key_points': [f'yes={yes}%', f'no={no}%'],
            'conversation_history': [message for u, a in turns for message in
                                     ({'role': 'user', 'content': u}, {'role': 'assistant', 'content': a})]}


def make_schedule(context, seed):
    cases = [build_case(75, 25), build_case(60, 40)]
    for case in cases:
        case['retrieval'] = {'context': context}
    # Same question, four history windows per counterfactual case.
    work = [(case, n, *make_prompt(case, n)) for case in cases for n in (0, 1, 3, 5)]
    random.Random(seed).shuffle(work)
    return work


def has_explicit_reference(context, percentages):
    return any(re.search(r'(?<!\d)' + str(p) + r'\s*(?:%|percent\b)', context, re.I)
               for p in percentages)



def generate_legacy(url, prompt):
    """Read the app's NDJSON stream; it has no verified finish or server timings."""
    start = time.perf_counter()
    first = None
    text_parts = []
    done = False
    finish_reason = None
    native_metrics = {}
    with requests.post(url.rstrip('/') + '/stream', json={'prompt': prompt},
                       stream=True, timeout=(5, 300)) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=False):
            if not line:
                continue
            event = json.loads(line.decode('utf-8'))
            if event.get('error'):
                raise RuntimeError('Legacy stream returned an error')
            if event.get('done'):
                done = True
                finish_reason = event.get('finish_reason')
                if isinstance(event.get('metrics'), dict):
                    native_metrics = event['metrics']
                break
            chunk = event.get('delta') or ''
            if not isinstance(chunk, str):
                raise ValueError('Invalid legacy stream delta')
            if chunk:
                if first is None and chunk.strip():
                    first = time.perf_counter() - start
                text_parts.append(chunk)
    return {'answer': ''.join(text_parts), 'first_answer_seconds': first,
            'wall_seconds': time.perf_counter() - start, 'stream_done': done,
            'finish_reason': finish_reason, 'unexpected_reasoning': None,
            'completion_verified': finish_reason == 'stop', **native_metrics}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--model-label', required=True, help='User-supplied; verify server identity separately')
    parser.add_argument('--output', required=True, help='New JSONL file; never overwrites')
    parser.add_argument('--rag-config', default='configs/retrieval.yaml')
    parser.add_argument('--top-k', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--protocol', choices=('llama-server', 'legacy'), default='llama-server',
                        help='legacy: app /stream; no reliable finish reason or server timing metrics')
    args = parser.parse_args()
    if args.top_k < 1: parser.error('--top-k must be positive')
    try:
        # Lazy import allows tests without model downloads or GPU dependencies.
        from rag.rag_retrieve import RagRetriever
        retriever = RagRetriever(args.rag_config)
        hits = retriever.retrieve(QUESTION, top_k=args.top_k)
        context = retriever.format_context(hits)
        if not hits or not context.strip():
            raise ValueError('Retrieval returned no formatted evidence; check the index and configuration')
        schedule = make_schedule(context, args.seed)
        meta = [{'slide_id': h.slide_id, 'source_pdf': h.source_pdf,
                 'slide_number': str(h.slide_number), 'score': h.score} for h in hits]
        leak = has_explicit_reference(context, [75, 25, 60, 40])
        failures = 0
        with open(args.output, 'x', encoding='utf-8') as out:
            print(f'8 synthetic requests, live retrieval once, protocol={args.protocol}, top_k={args.top_k}, explicit_reference_in_rag={leak}', flush=True)
            for case, limit, prompt, actual in schedule:
                row = {'schema': 'history-retrieval-pilot-v1', 'question_id': case['question_id'],
                       'question': QUESTION, 'full_history': case['conversation_history'],
                       'history_limit': limit, 'history_turns_sent': actual,
                       'reference_key_points': case['reference_key_points'],
                       'synthetic': True, 'source_validation': 'pending_human_review',
                       'model_label': args.model_label, 'teaching_style': TEACHING_STYLE,
                       'protocol': args.protocol,
                       'thinking_enabled': False if args.protocol == 'llama-server' else None,
                       'cache_prompt_requested': False if args.protocol == 'llama-server' else None,
                       'temperature': 0 if args.protocol == 'llama-server' else None,
                       'max_output_tokens': 1024 if args.protocol == 'llama-server' else None, 'seed': args.seed,
                       'rag_config': args.rag_config, 'retrieval_query': QUESTION,
                       'top_k_requested': args.top_k, 'retrieved_hits': meta, 'rag_context': context,
                       'rag_sha256': hashlib.sha256(context.encode()).hexdigest(),
                       'config_sha256': hashlib.sha256(Path(args.rag_config).read_bytes()).hexdigest(),
                       'explicit_reference_in_rag': leak,
                       'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
                try:
                    row.update(generate_legacy(args.url, prompt) if args.protocol == 'legacy'
                               else generate(args.url, prompt))
                except Exception as exc:
                    # Preserve failures without logging the request or sensitive response text.
                    row.update({'error_type': type(exc).__name__, 'stream_done': False})
                required = ('serverPromptTokens', 'serverCachedTokens', 'serverPrefillSeconds',
                            'serverOutputTokens', 'serverDecodeSeconds')
                row['metrics_complete'] = all(k in row for k in required)
                row['complete'] = bool(row.get('stream_done') and row.get('finish_reason') == 'stop'
                                       and row.get('unexpected_reasoning') is not True and row.get('answer', '').strip())
                row['completion_verified'] = bool(row['complete'])
                # Legacy done can be emitted after server errors, so report only transport evidence.
                row['transport_usable'] = bool(row.get('stream_done') and row.get('answer', '').strip())
                failures += (not row['complete'] or not row['metrics_complete']) if args.protocol == 'llama-server' else not row['transport_usable']
                out.write(json.dumps(row, ensure_ascii=False) + '\n'); out.flush()
                print(f"{case['question_id']} history={actual} finish={row.get('finish_reason', row.get('error_type'))} complete={row['complete']} metrics={row['metrics_complete']}", flush=True)
        print(f'Saved {args.output}; unusable={failures}. Manual evidence and answer review required.')
        if args.protocol == 'legacy':
            print('Legacy: exact prefill/decode and token usage may be unavailable; inspect finish_reason and native metrics.')
        return 1 if failures else 0
    except (ImportError, OSError, KeyError, ValueError) as exc:
        print(f'Pilot could not continue: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1

if __name__ == '__main__': raise SystemExit(main())
