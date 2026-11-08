"""Run: python -m unittest test_task6_benchmark -v"""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import task6_benchmark as bench
from contextlib import closing

class MetricsParserTests(unittest.TestCase):
    def setUp(self):
        self.d=tempfile.TemporaryDirectory()
        self.root=Path(self.d.name)

    def tearDown(self):
        self.d.cleanup()

    def test_missing_files_do_not_fabricate_measurements(self):
        self.assertEqual(bench.collect_logs(self.root/'missing'),[])
        self.assertEqual(bench.collect_events(self.root/'missing'),([],[]))
        self.assertEqual(bench.collect_database(self.root/'missing',0,9),([],[]))
        self.assertIsNone(bench.percentile([],95))

    def test_parser_keeps_backend_and_browser_latency_separate(self):
        log=self.root/'backend.log'
        log.write_text('[TEXT] What is entropy?\n[LLM METRICS]\n'
            'LLM TTFT (prompt -> first token) : 0.177 s\n'
            '[PIPELINE TOTALS]\nEnd-to-end total : 2.187 s\n', encoding='utf-8')
        turns=bench.collect_logs(log)
        events=self.root/'events.jsonl'
        records=[{'session_id':'s','turn_id':'t','event':event,'mono_ns':ns} for event,ns in [
            ('tutor_turn_started',1000000000),('audio_chunk',1200000000),
            ('audio_done',3000000000),('browser_audio_finished',7000000000)]]
        events.write_text('\n'.join(json.dumps(r) for r in records)+'\n',encoding='utf-8')
        turns_events,invalid=bench.collect_events(events)
        self.assertEqual(invalid,0)
        self.assertEqual(turns[0]['backend_total_s'],2.187)
        self.assertEqual(turns_events[0]['browser_ack_s'],6)
        self.assertEqual(turns_events[0]['browser_drain_after_done_s'],4)
        self.assertNotEqual(turns[0]['backend_total_s'], turns_events[0]['browser_ack_s'])

    def test_detect_duplicate_and_premature_ack(self):
        f=self.root/'e.jsonl'
        names=[('tutor_turn_started',10),('browser_audio_finished',20),('audio_done',30),
               ('browser_audio_finished',40)]
        f.write_text('\n'.join(json.dumps(dict(session_id='s',turn_id='t',event=e,mono_ns=n)) for e,n in names))
        r,_=bench.collect_events(f)
        self.assertTrue(r[0]['ack_before_audio_done'])
        self.assertEqual(r[0]['ack_count'],2)

    def test_query_database_filters_by_run_start_and_end(self):
        db=self.root/'db.sqlite'
        with closing(sqlite3.connect(db)) as conn, conn:
            conn.execute('CREATE TABLE evaluations (id TEXT, status TEXT, reason TEXT,score INTEGER,attempts INTEGER,created_at REAL,updated_at REAL,payload TEXT)')
            conn.execute('CREATE TABLE attempts (id TEXT, number INTEGER,elapsed_s REAL,status TEXT,reason TEXT)')
            payload=json.dumps({'turn_id':'T','evaluation_dimension':'correctness','outcome':'delivered',
                                'question_text':'What is entropy?', 'judge_model_id':'fare_20b_q4'})
            for item,when in [('old',5),('included',15),('new',25)]:
                conn.execute('INSERT INTO evaluations VALUES(?,?,?,?,?,?,?,?)',
                             (item,'valid',None,4,1,when,when+1,payload))
                conn.execute('INSERT INTO attempts VALUES(?,?,?,?,?)',(item,1,.5,'valid',None))
        rows, attempts=bench.collect_database(db,10,20)
        self.assertEqual([x['id'] for x in rows], ['included'])
        self.assertEqual([x['id'] for x in attempts], ['included'])


if __name__=='__main__':unittest.main()
