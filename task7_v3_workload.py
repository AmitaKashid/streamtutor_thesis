"""Task 7: reproducible v3 selection-split workload, review and execution plan.

No GPU operations, judge calls, browser automation, or changes to frozen v3 files.
Run `prepare`, review generated CSV against PDFs, then `finalize`.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

DATASET = Path('data/judge_evaluation/streamtutor_judge_dataset_v3/questions.jsonl')
DEFAULT_ROOT = Path('judge_outputs/task7')
REVIEW_COLUMNS = ('question_id', 'split', 'course_unit', 'question_type', 'exact_utterance', 'canonical_question', 'reference_answer', 'reference_key_points', 'pdf_filename', 'pdf_pages', 'evidence_summary', 'review_status', 'reviewer', 'review_notes', 'source_sha256')
SCHEDULE_COLUMNS = ('block_id', 'block_order', 'condition', 'trial_id', 'question_id', 'question', 'replicate', 'status')


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def load_questions(path):
    data = []
    with path.open(encoding='utf-8-sig') as handle:
        for n, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f'Expected JSON object on line {n}')
                data.append(row)
    ids = [q.get('question_id') for q in data]
    if None in ids or len(ids) != len(set(ids)):
        raise ValueError('Missing or duplicate question_id')
    return data


def eligible(q):
    r = q.get('retrieval') or {}
    src = r.get('source') or {}
    return (q.get('split') == 'selection' and q.get('context_dependency') == 'standalone'
            and not q.get('conversation_history') and r.get('condition') == 'relevant'
            and bool(src.get('filename')) and bool(src.get('pages'))
            and bool((q.get('reference_answer') or '').strip())
            and bool((q.get('user_utterance') or '').strip()))


def selected_questions(questions):
    selected = sorted((q for q in questions if eligible(q)), key=lambda x: x['question_id'])
    texts = [q['user_utterance'] for q in selected]
    if len(texts) != len(set(texts)):
        raise ValueError('Duplicate exact utterances; exact-string runtime reference lookup ambiguous')
    if not selected:
        raise ValueError('No eligible standalone selection-split questions')
    return selected


def candidate_row(q):
    source = q['retrieval']['source']
    return dict(question_id=q['question_id'], split=q['split'], course_unit=q.get('course_unit', ''),
        question_type=q.get('question_type', ''), exact_utterance=q['user_utterance'],
        canonical_question=q.get('canonical_question', ''), reference_answer=q['reference_answer'],
        reference_key_points=' | '.join(q.get('reference_key_points') or []),
        pdf_filename=source['filename'], pdf_pages=','.join(map(str, source['pages'])),
        evidence_summary=source.get('evidence_summary', ''), review_status='UNREVIEWED',
        reviewer='', review_notes='', source_sha256=sha(q))


def read_review(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not set(REVIEW_COLUMNS).issubset(reader.fieldnames or []):
            raise ValueError('Review CSV schema mismatch')
        return list(reader)


def validate_review(questions, rows):
    expected = {q['question_id']: candidate_row(q) for q in selected_questions(questions)}
    if len(rows) != len(expected):
        raise ValueError(f'Expected {len(expected)} review rows, got {len(rows)}')
    seen = set(); failures=[]
    protected = [c for c in REVIEW_COLUMNS if c not in ('review_status','reviewer','review_notes')]
    for row in rows:
        key = row.get('question_id')
        if key in seen or key not in expected:
            failures.append(f'Duplicate or unknown question_id: {key}'); continue
        seen.add(key)
        for col in protected:
            if str(row.get(col,'')) != str(expected[key].get(col,'')):
                failures.append(f'{key}: frozen-source mismatch in {col}')
        if row.get('review_status','').strip() != 'VERIFIED':
            failures.append(f'{key}: review_status must be VERIFIED')
        if len(row.get('reviewer','').strip()) < 2:
            failures.append(f'{key}: reviewer missing')
        if len(row.get('review_notes','').strip()) < 12:
            failures.append(f'{key}: review_notes missing or too short (record actual slide verification)')
    for key in expected.keys()-seen:
        failures.append(f'{key}: missing review row')
    return failures


def create_schedule(selected, repeats=4):
    """8 sequential blocks: OFF/ON then ON/OFF (ABBA per round), 11 prompts per block.
    Condition grouping avoids restarting FastAPI for each question. Each trial
    still requires its own clean browser conversation. Block counterbalancing
    controls gross order but is not fully randomized; see protocol limitations.
    """
    if repeats != 4:
        raise ValueError('Frozen design has four repetitions per condition')
    blocks = ['off', 'on', 'on', 'off', 'on', 'off', 'off', 'on']
    rows=[]
    counts=Counter()
    for idx, condition in enumerate(blocks):
        block_id=f'b{idx+1:02d}_{condition}'
        prompts = selected if idx%2==0 else list(reversed(selected))
        for position, q in enumerate(prompts, 1):
            qid=q['question_id']; counts[(qid, condition)]+=1
            rows.append(dict(block_id=block_id, block_order=position, condition=condition,
                trial_id=f't7_{qid}_{condition}_{counts[(qid,condition)]:02d}',
                question_id=qid, question=q['user_utterance'],
                replicate=counts[(qid,condition)], status='PLANNED_NOT_EXECUTED'))
    assert len(rows)==len(selected)*8
    assert all(counts[(q['question_id'], c)]==4 for q in selected for c in ('off','on'))
    return rows


def write_json(path, obj):
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def write_csv(path, columns, rows):
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare','check','finalize'))
    p.add_argument('--dataset',type=Path,default=DATASET)
    p.add_argument('--outdir',type=Path,default=DEFAULT_ROOT)
    args=p.parse_args(argv)
    questions=load_questions(args.dataset)
    chosen=selected_questions(questions)
    review=args.outdir/'v3_selection_reference_review.csv'
    if args.action=='prepare':
        write_csv(review,REVIEW_COLUMNS,[candidate_row(q) for q in chosen])
        write_json(args.outdir/'v3_selection_workload_candidate.json',dict(status='UNREVIEWED',
            frozen_questions_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            question_count=len(chosen),questions=[dict(question_id=q['question_id'],
            utterance=q['user_utterance'],source=q['retrieval']['source']) for q in chosen]))
        print(f'Prepared {len(chosen)} standalone questions from selection split')
        print('Source units:',dict(Counter(q.get('course_unit') for q in chosen)))
        print('Review PDF citations and reference content in:',review)
        print('Status: UNREVIEWED. No runtime registry or final schedule exported.')
        return 0
    rows=read_review(review)
    failures=validate_review(questions,rows)
    if failures:
        print(f'BLOCKED: {len(failures)} review problems')
        for failure in failures[:30]:print(' -',failure)
        return 2
    print(f'PASS: {len(chosen)} reviewed references match exact frozen dataset fields')
    if args.action=='check':return 0
    # Publish mapping only after manual review and frozen-source reconciliation.
    registry={q['user_utterance']:q['reference_answer'] for q in chosen}
    metadata={q['user_utterance']:dict(question_id=q['question_id'],source=q['retrieval']['source'],
        source_sha256=sha(q), reviewer=next(x['reviewer'] for x in rows if x['question_id']==q['question_id']),
        review_notes=next(x['review_notes'] for x in rows if x['question_id']==q['question_id'])) for q in chosen}
    write_json(args.outdir/'runtime_verified_references.json',registry)
    write_json(args.outdir/'runtime_reference_provenance.json',metadata)
    schedule=create_schedule(chosen)
    write_csv(args.outdir/'runtime_schedule.csv',SCHEDULE_COLUMNS,schedule)
    print(f'Produced {len(schedule)} planned trials in 8 condition blocks (4 OFF, 4 ON).')
    print('IMPORTANT: exact runtime adapter registry path/configuration must be verified before ON execution.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
