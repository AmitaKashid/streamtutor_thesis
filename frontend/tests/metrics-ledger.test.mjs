import test from 'node:test';
import assert from 'node:assert/strict';
import { createMetricsLedger } from '../app/metrics-ledger.mjs';

test('early metrics survive turn start and late metrics update the same completed row', () => {
  const ledger = createMetricsLedger();
  ledger.update('one', { stt: 1.5, promptChars: 5000 });
  assert.deepEqual(ledger.start('one', 'llama'), { stt: 1.5, promptChars: 5000, model: 'llama' });
  ledger.update('one', { ttft: .4 });
  ledger.finish('one');
  ledger.start('two', 'qwen');
  ledger.update('two', { ttft: .7 });
  ledger.update('one', { llmAnswerTime: 3.2 });
  ledger.finish('two');
  assert.deepEqual(ledger.history().map(row => row.metrics), [
    { stt: 1.5, promptChars: 5000, model: 'llama', ttft: .4, llmAnswerTime: 3.2 },
    { model: 'qwen', ttft: .7 },
  ]);
});

test('duplicate and unknown completion events do not fabricate rows', () => {
  const ledger = createMetricsLedger();
  ledger.finish('unknown');
  ledger.start('one', null);
  ledger.finish('one'); ledger.finish('one');
  assert.equal(ledger.history().length, 1);
});

test('STT zero placeholder does not erase a measured value', () => {
  const ledger = createMetricsLedger();
  ledger.start('one', null);
  ledger.update('one', { stt: 1.2 });
  assert.equal(ledger.update('one', { stt: 0 }).stt, 1.2);
  assert.equal(ledger.update('two', { stt: 0 }).stt, undefined);
});
