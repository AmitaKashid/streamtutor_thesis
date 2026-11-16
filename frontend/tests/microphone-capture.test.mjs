import test from 'node:test';
import assert from 'node:assert/strict';
import { openEchoCancelledMicrophone } from '../app/microphone-capture.mjs';

test('requires AEC without changing noise suppression or gain settings', async () => {
  const calls = [], stream = {};
  const devices = { getUserMedia: async options => { calls.push(options); return stream; } };
  assert.equal(await openEchoCancelledMicrophone(devices), stream);
  assert.deepEqual(calls[0].audio, { echoCancellation: { exact: true },
    noiseSuppression: true, autoGainControl: true, channelCount: 1 });
  assert.equal(calls.length, 1);
});

test('falls back only when echo cancellation itself cannot be required', async () => {
  const calls = [], stream = {};
  const devices = { getUserMedia: async options => {
    calls.push(options);
    if (calls.length === 1) throw Object.assign(new Error('unsupported'), {
      name: 'OverconstrainedError', constraint: 'echoCancellation',
    });
    return stream;
  }};
  assert.equal(await openEchoCancelledMicrophone(devices), stream);
  assert.equal(calls[1].audio.echoCancellation, true);
});

test('does not hide permission errors or unrelated constraint failures', async () => {
  for (const error of [
    Object.assign(new Error('denied'), {name: 'NotAllowedError'}),
    Object.assign(new Error('channel'), {name: 'OverconstrainedError', constraint: 'channelCount'}),
  ]) {
    let calls = 0;
    await assert.rejects(openEchoCancelledMicrophone({getUserMedia: async () => { calls++; throw error; }}));
    assert.equal(calls, 1);
  }
});
