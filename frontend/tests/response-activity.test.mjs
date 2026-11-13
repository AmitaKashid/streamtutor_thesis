import test from 'node:test';
import assert from 'node:assert/strict';
import { idleResponse, responseActivity as step, responseIsActive as active } from '../app/response-activity.mjs';

test('text submission stays active until generation completes', () => {
  let s = step(idleResponse(), { type: 'submit' });
  assert.equal(active(s), true);
  s = step(s, { type: 'start', audio: false });
  assert.equal(active(s), true);
  s = step(s, { type: 'generation_done' });
  assert.equal(active(s), false);
});

test('audio response stays active after generation until browser playback drains', () => {
  let s = step(step(idleResponse(), { type: 'submit' }), { type: 'start', audio: true });
  s = step(s, { type: 'generation_done' });
  assert.equal(active(s), true);
  s = step(s, { type: 'audio_done' });
  assert.equal(active(s), false);
});

test('generation remains active if audio finishes first', () => {
  let s = step(idleResponse(), { type: 'start', audio: true });
  s = step(s, { type: 'audio_done' });
  assert.equal(active(s), true);
  s = step(s, { type: 'generation_done' });
  assert.equal(active(s), false);
});

test('stop remains pending until interrupted generation finishes', () => {
  let s = step(idleResponse(), { type: 'submit' });
  s = step(s, { type: 'stop' });
  assert.equal(active(s), true);
  assert.equal(s.cancelling, true);
  s = step(s, { type: 'start', audio: true });
  assert.equal(s.audioPending, false);
  s = step(s, { type: 'generation_done' });
  assert.equal(active(s), false);
  assert.equal(s.cancelling, false);
});

test('stopping queued audio after generation ends completes immediately', () => {
  let s = step(idleResponse(), { type: 'start', audio: true });
  s = step(s, { type: 'generation_done' });
  s = step(s, { type: 'stop' });
  assert.equal(active(s), false);
});

test('disconnect or error clears pending activity', () => {
  const s = step(step(idleResponse(), { type: 'start', audio: true }), { type: 'reset' });
  assert.deepEqual(s, idleResponse());
});
