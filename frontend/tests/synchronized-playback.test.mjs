import test from 'node:test';
import assert from 'node:assert/strict';
import { createSynchronizedPlayback } from '../app/synchronized-playback.mjs';

function fixture({ suspended = false, failStart = false } = {}) {
  const events = [];
  const sources = [];
  let resolveResume;
  let rejectResume;
  const resumePromise = new Promise((resolve, reject) => {
    resolveResume = resolve;
    rejectResume = reject;
  });
  const context = {
    state: suspended ? 'suspended' : 'running',
    destination: {},
    resume: () => resumePromise,
    createBufferSource() {
      const source = {
        buffer: null, onended: null, stopped: false,
        connect() {},
        start() {
          if (failStart) throw new Error('start failed');
          events.push(['start', this.buffer]);
        },
        stop() { this.stopped = true; },
        end() { this.onended?.(); },
      };
      sources.push(source);
      return source;
    },
  };
  const playback = createSynchronizedPlayback({
    getContext: () => context,
    onText: (id, text) => events.push(['text', id, text]),
    onDrained: (id) => events.push(['drained', id]),
    onError: (id, error) => events.push(['error', id, error.message]),
  });
  return { playback, context, events, sources, resolveResume, rejectResume };
}

const buffer = (name) => ({ name });

test('reveals each segment on playback start and drains only after all audio', () => {
  const { playback, events, sources } = fixture();
  const first = buffer('first');
  const second = buffer('second');
  playback.start('t1');
  playback.enqueueAudio('t1', first, 'First sentence. ');
  playback.enqueueText('t1', '😊 ');
  playback.enqueueAudio('t1', second, 'Second sentence. ');
  playback.finish('t1');
  assert.deepEqual(events, [['start', first], ['text', 't1', 'First sentence. ']]);
  sources[0].end();
  assert.deepEqual(events.slice(2), [
    ['text', 't1', '😊 '], ['start', second], ['text', 't1', 'Second sentence. '],
  ]);
  sources[1].end();
  assert.deepEqual(events.at(-1), ['drained', 't1']);
  sources[1].end();
  assert.equal(events.filter((event) => event[0] === 'drained').length, 1);
});

test('audio without a new text marker does not repeat the segment', () => {
  const { playback, events, sources } = fixture();
  playback.start('t1');
  playback.enqueueAudio('t1', buffer('a'), 'Hello. ');
  playback.enqueueAudio('t1', buffer('b'));
  sources[0].end();
  assert.deepEqual(events.filter((event) => event[0] === 'text'), [['text', 't1', 'Hello. ']]);
});

test('cancellation drops queued text and ignores stale audio callbacks', () => {
  const { playback, events, sources } = fixture();
  playback.start('old');
  playback.enqueueAudio('old', buffer('a'), 'Heard. ');
  const lateEnd = sources[0].onended;
  playback.enqueueAudio('old', buffer('b'), 'Never heard. ');
  playback.finish('old');
  playback.clear();
  assert.equal(sources[0].stopped, true);
  playback.start('new');
  playback.enqueueAudio('new', buffer('c'), 'New turn. ');
  lateEnd();
  playback.enqueueAudio('old', buffer('stale'), 'Stale. ');
  assert.deepEqual(events.filter((event) => event[0] === 'text'), [
    ['text', 'old', 'Heard. '], ['text', 'new', 'New turn. '],
  ]);
  assert.equal(events.some((event) => event[0] === 'drained'), false);
});

test('does not reveal speech while the audio context is suspended', async () => {
  const { playback, context, events, resolveResume } = fixture({ suspended: true });
  playback.start('t1');
  playback.enqueueAudio('t1', buffer('a'), 'Spoken text. ');
  assert.deepEqual(events, []);
  context.state = 'running';
  resolveResume();
  await Promise.resolve();
  assert.deepEqual(events.map((event) => event[0]), ['start', 'text']);
});

test('a late resume from a cancelled turn cannot start its audio', async () => {
  const { playback, context, events, resolveResume } = fixture({ suspended: true });
  playback.start('old');
  playback.enqueueAudio('old', buffer('old'), 'Old text. ');
  playback.clear();
  playback.start('new');
  context.state = 'running';
  playback.enqueueAudio('new', buffer('new'), 'New text. ');
  resolveResume();
  await Promise.resolve();
  assert.deepEqual(events.filter((event) => event[0] === 'text'), [['text', 'new', 'New text. ']]);
});

test('playback errors do not reveal text or report delivery', async () => {
  const startFailure = fixture({ failStart: true });
  startFailure.playback.start('t1');
  startFailure.playback.enqueueAudio('t1', buffer('a'), 'Hidden. ');
  assert.deepEqual(startFailure.events, [['error', 't1', 'start failed']]);

  const resumeFailure = fixture({ suspended: true });
  resumeFailure.playback.start('t2');
  resumeFailure.playback.enqueueAudio('t2', buffer('b'), 'Also hidden. ');
  resumeFailure.rejectResume(new Error('permission denied'));
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(resumeFailure.events, [['error', 't2', 'permission denied']]);
});

test('a text-only segment waits behind audio and precedes completion', () => {
  const { playback, events, sources } = fixture();
  playback.start('t1');
  playback.enqueueAudio('t1', buffer('a'), 'Last sentence. ');
  playback.enqueueText('t1', '😊 ');
  playback.finish('t1');
  sources[0].end();
  assert.deepEqual(events.slice(-2), [['text', 't1', '😊 '], ['drained', 't1']]);
});
