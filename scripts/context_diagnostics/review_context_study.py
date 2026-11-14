"""Make a blinded answer-review CSV from a context-study JSONL file."""
import argparse
import csv
import hashlib
import json
import random


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True, help='New CSV file; never overwrites')
    ap.add_argument('--seed', type=int, default=17)
    args = ap.parse_args()
    with open(args.input, encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    items = []
    for n, row in enumerate(rows):
        case = row['case']; limit = row['history_limit']
        history = case['conversation_history'][-2*limit:] if limit else []
        # Stable opaque ID joins to the original row number, without revealing condition.
        review_id = hashlib.sha256(f"{args.seed}:{n}".encode()).hexdigest()[:16]
        items.append({'review_id': review_id, 'question': case['user_utterance'],
                      'history_sent': json.dumps(history, ensure_ascii=False),
                      'rag_context': case['retrieval']['context'],
                      'reference': json.dumps(case['reference_key_points'], ensure_ascii=False),
                      'answer': row.get('answer', ''), 'complete': row.get('complete'),
                      'correct_recall': '', 'appropriate_clarification': '',
                      'false_denial': '', 'notes': ''})
    random.Random(args.seed).shuffle(items)
    if not items: raise ValueError('Input contains no rows')
    with open(args.output, 'x', encoding='utf-8-sig', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(items[0]))
        writer.writeheader(); writer.writerows(items)
    print(f'Wrote {len(items)} rows. Review each quality field as yes/no/not-applicable; incomplete rows are not quality scores.')

if __name__ == '__main__': main()
