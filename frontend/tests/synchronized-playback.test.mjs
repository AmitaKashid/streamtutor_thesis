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
    currentTime: 0,
    destination: {},
    resume: () => resumePromise,
    createBufferSource() {
      const source = {
        buffer: null, onended: null, stopped: false,
        connect() {},
        start(at) {
          if (failStart) throw new Error('start failed');
          this.startedAt = at;
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
    onUnderrun: (id, seconds) => events.push(['underrun', id, seconds]),
    onPauseError: (id, error) => events.push(['pause-error', id, error.message]),
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

test('reveals each aligned word according to the audio clock', async () => {
  const { playback, context, events, sources } = fixture();
  playback.start('t1');
  playback.enqueueAudio('t1', buffer('a'), 'Hello world! ', [
    { text: 'Hello', start: .2, end: .4 },
    { text: ' world! ', start: .7, end: 1.1 },
  ]);
  playback.finish('t1');
  assert.equal(events.filter((event) => event[0] === 'text').length, 0);
  context.currentTime = .25;
  await new Promise((resolve) => setTimeout(resolve, 40));
  assert.deepEqual(events.filter((event) => event[0] === 'text'), [['text', 't1', 'Hello']]);
  context.currentTime = .75;
  await new Promise((resolve) => setTimeout(resolve, 40));
  assert.deepEqual(events.filter((event) => event[0] === 'text'), [
    ['text', 't1', 'Hello'], ['text', 't1', ' world! '],
  ]);
  sources[0].end();
  assert.deepEqual(events.at(-1), ['drained', 't1']);
});

test('stop prevents scheduled words from a previous turn from appearing', async () => {
  const { playback, context, events } = fixture();
  playback.start('old');
  playback.enqueueAudio('old', buffer('a'), 'Hello world', [
    { text: 'Hello', start: 0, end: .2 },
    { text: ' world', start: .7, end: .9 },
  ]);
  playback.clear();
  context.currentTime = 2;
  await new Promise((resolve) => setTimeout(resolve, 40));
  assert.deepEqual(events.filter((event) => event[0] === 'text'), [['text', 'old', 'Hello']]);
  assert.equal(events.some((event) => event[0] === 'drained'), false);
});

test('prefers the output-device audio clock when available', async () => {
  const { playback, context, events, sources } = fixture();
  let outputTime = 0;
  context.getOutputTimestamp = () => ({ contextTime: outputTime });
  playback.start('t1');
  playback.enqueueAudio('t1', buffer('a'), 'Hello', [{ text: 'Hello', start: .2, end: .4 }]);
  context.currentTime = 1;
  await new Promise((resolve) => setTimeout(resolve, 40));
  assert.equal(events.filter((event) => event[0] === 'text').length, 0);
  outputTime = .3;
  await new Promise((resolve) => setTimeout(resolve, 40));
  assert.deepEqual(events.filter((event) => event[0] === 'text'), [['text', 't1', 'Hello']]);
  sources[0].end();
});

test('keeps the last word pending after rendering until the output clock reaches it', async () => {
  const { playback, context, events, sources } = fixture();
  let outputTime = 0;
  context.getOutputTimestamp = () => ({ contextTime: outputTime });
  playback.start('t1');
  playback.enqueueAudio('t1', buffer('a'), 'Hello', [{ text: 'Hello', start: .5, end: .9 }]);
  playback.finish('t1');
  context.currentTime = 1;
  sources[0].end();
  assert.equal(events.some((event) => event[0] === 'text' || event[0] === 'drained'), false);
  outputTime = .6;
  await new Promise((resolve) => setTimeout(resolve, 40));
  assert.deepEqual(events.slice(-2), [['text', 't1', 'Hello'], ['drained', 't1']]);
});

test('reports completion only when the output clock reaches the end of audio', async () => {
  const { playback, context, events, sources } = fixture();
  let outputTime = 0;
  context.getOutputTimestamp = () => ({ contextTime: outputTime });
  playback.start('t1');
  playback.enqueueAudio('t1', { name: 'a', duration: 1 }, 'Hello', [
    { text: 'Hello', start: .2, end: .8 },
  ]);
  playback.finish('t1');
  context.currentTime = 1;
  outputTime = .5;
  sources[0].end();
  await new Promise((resolve) => setTimeout(resolve, 40));
  assert.deepEqual(events.filter((event) => event[0] === 'text'), [['text', 't1', 'Hello']]);
  assert.equal(events.some((event) => event[0] === 'drained'), false);
  outputTime = 1;
  await new Promise((resolve) => setTimeout(resolve, 40));
  assert.deepEqual(events.at(-1), ['drained', 't1']);
});

test('keeps non-spoken text behind the previous output until it is heard', async () => {
  const { playback, context, events, sources } = fixture();
  let outputTime = 0;
  context.getOutputTimestamp = () => ({ contextTime: outputTime });
  playback.start('t1');
  playback.enqueueAudio('t1', { name: 'a', duration: 1 }, 'Hello', [
    { text: 'Hello', start: .2, end: .8 },
  ]);
  playback.enqueueText('t1', '😊 ');
  playback.finish('t1');
  outputTime = .5;
  context.currentTime = 1;
  sources[0].end();
  await new Promise((resolve) => setTimeout(resolve, 40));
  assert.deepEqual(events.filter((event) => event[0] === 'text'), [['text', 't1', 'Hello']]);
  outputTime = 1;
  await new Promise((resolve) => setTimeout(resolve, 40));
  assert.deepEqual(events.slice(-2), [['text', 't1', '😊 '], ['drained', 't1']]);
});

test('schedules a ready next timed segment without waiting for an end callback', async () => {
  const { playback, context, events, sources } = fixture();
  playback.start('t1');
  playback.enqueueAudio('t1', { name: 'one', duration: 1 }, 'Hello ', [
    { text: 'Hello ', start: 0, end: .8 },
  ]);
  playback.enqueueAudio('t1', { name: 'two', duration: 1 }, 'world', [
    { text: 'world', start: .1, end: .7 },
  ]);
  assert.equal(sources.length, 2);
  assert.equal(sources[1].startedAt, 1);
  assert.deepEqual(events.filter((event) => event[0] === 'text'), [['text', 't1', 'Hello ']]);
  context.currentTime = 1.2;
  await new Promise((resolve) => setTimeout(resolve, 40));
  assert.deepEqual(events.filter((event) => event[0] === 'text'), [['text', 't1', 'Hello '], ['text', 't1', 'world']]);
  playback.clear();
  assert.equal(sources[0].stopped, true);
  assert.equal(sources[1].stopped, true);
});

test('schedules every ready timed segment without waiting for main-thread end callbacks', () => {
  const { playback, sources } = fixture();
  playback.start('t1');
  for (const text of ['one', 'two', 'three']) {
    playback.enqueueAudio('t1', { name: text, duration: 1 }, text,
      [{ text, start: 0, end: .8 }]);
  }
  assert.equal(sources.length, 3);
  assert.deepEqual(sources.map((s) => s.startedAt), [0, 1, 2]);
  playback.clear();
  assert.ok(sources.every((s) => s.stopped));
});

test('schedules untimed audio ahead and delays its text until its audio starts', async () => {
  const { playback, context, events, sources } = fixture();
  playback.start('t1');
  playback.enqueueAudio('t1', { duration: 1 }, 'First ', [{ text: 'First ', start: 0, end: .8 }]);
  playback.enqueueAudio('t1', { duration: 1 }, 'Second');
  assert.equal(sources.length, 2);
  assert.equal(sources[1].startedAt, 1);
  assert.deepEqual(events.filter((e) => e[0] === 'text'), [['text', 't1', 'First ']]);
  context.currentTime = 1.1;
  await new Promise((resolve) => setTimeout(resolve, 40));
  assert.deepEqual(events.filter((e) => e[0] === 'text'), [['text', 't1', 'First '], ['text', 't1', 'Second']]);
  playback.clear();
});

test('text-only markers do not block scheduling later audio or reveal early', async () => {
  const { playback, context, events, sources } = fixture();
  playback.start('t1');
  playback.enqueueAudio('t1', { duration: 1 }, 'First ', [{ text: 'First ', start: 0, end: .8 }]);
  playback.enqueueText('t1', '😊 ');
  playback.enqueueAudio('t1', { duration: 1 }, 'Second', [{ text: 'Second', start: .1, end: .8 }]);
  assert.equal(sources.length, 2);
  assert.equal(sources[1].startedAt, 1);
  assert.deepEqual(events.filter((e) => e[0] === 'text'), [['text', 't1', 'First ']]);
  context.currentTime = 1.2;
  await new Promise((resolve) => setTimeout(resolve, 40));
  assert.equal(events.filter((e) => e[0] === 'text').map((e) => e[2]).join(''), 'First 😊 Second');
  playback.clear();
});

test('limits scheduled nodes and refills from the queue after a node ends', () => {
  const { playback, context, sources } = fixture();
  playback.start('t1');
  for (let i = 0; i < 18; i += 1) {
    playback.enqueueAudio('t1', { duration: 1 }, `${i} `, [{ text: `${i} `, start: 0, end: .8 }]);
  }
  assert.equal(sources.length, 17);
  context.currentTime = 1;
  sources[0].end();
  assert.equal(sources.length, 18);
  assert.equal(sources[17].startedAt, 17);
  playback.clear();
});

test('completion remains correct when later end callbacks arrive first', async () => {
  const { playback, context, events, sources } = fixture();
  playback.start('t1');
  for (const text of ['one ', 'two ', 'three']) {
    playback.enqueueAudio('t1', { duration: 1 }, text, [{ text, start: 0, end: .8 }]);
  }
  playback.finish('t1');
  context.currentTime = 2;
  sources[1].end();
  sources[0].end();
  assert.equal(events.some((e) => e[0] === 'drained'), false);
  context.currentTime = 3;
  sources[2].end();
  await new Promise((resolve) => setTimeout(resolve, 40));
  assert.equal(events.filter((e) => e[0] === 'drained').length, 1);
  assert.equal(events.filter((e) => e[0] === 'text').map((e) => e[2]).join(''), 'one two three');
  playback.clear();
});

test('records a late segment as an underrun rather than fabricating continuous audio', () => {
  const { playback, context, events, sources } = fixture();
  playback.start('t1');
  playback.enqueueAudio('t1', { name: 'one', duration: 1 }, 'First', [
    { text: 'First', start: 0, end: .8 },
  ]);
  context.currentTime = 1;
  sources[0].end();
  context.currentTime = 1.5;
  playback.enqueueAudio('t1', { name: 'two', duration: 1 }, 'Second', [
    { text: 'Second', start: 0, end: .8 },
  ]);
  assert.deepEqual(events.filter((event) => event[0] === 'underrun'), [['underrun', 't1', .5]]);
  playback.clear();
});

test('rejected interruption resumes the same queued answer and word timing', async () => {
  const { playback, context, events, sources } = fixture();
  context.suspend = () => { context.state = 'suspended'; return Promise.resolve(); };
  context.resume = () => { context.state = 'running'; return Promise.resolve(); };
  playback.start('t1');
  playback.enqueueAudio('t1', { duration: 1 }, 'First ', [{ text: 'First ', start: 0, end: .8 }]);
  playback.pause('t1');
  await Promise.resolve();
  playback.enqueueAudio('t1', { duration: 1 }, 'second', [{ text: 'second', start: .1, end: .8 }]);
  playback.finish('t1');
  assert.equal(context.state, 'suspended');
  assert.equal(sources.length, 1);
  assert.equal(sources[0].stopped, false);
  assert.equal(events.some((event) => event[0] === 'drained'), false);
  playback.resume('t1');
  await Promise.resolve();
  assert.equal(context.state, 'running');
  assert.equal(sources.length, 2);
  assert.equal(sources[1].startedAt, 1);
  context.currentTime = 1.2;
  await new Promise((resolve) => setTimeout(resolve, 40));
  assert.deepEqual(events.filter((event) => event[0] === 'text'), [['text', 't1', 'First '], ['text', 't1', 'second']]);
  playback.clear();
});

test('resume requested while suspend is pending is not lost', async () => {
  const { playback, context, sources } = fixture();
  let completeSuspend;
  context.suspend = () => new Promise((resolve) => {
    completeSuspend = () => { context.state = 'suspended'; resolve(); };
  });
  context.resume = () => { context.state = 'running'; return Promise.resolve(); };
  playback.start('t1');
  playback.enqueueAudio('t1', { duration: 1 }, 'Hello', [{ text: 'Hello', start: 0, end: .8 }]);
  playback.pause('t1');
  playback.resume('t1');
  completeSuspend();
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(context.state, 'running');
  assert.equal(sources[0].stopped, false);
  playback.clear();
});

test('a late suspend after cancellation cannot strand a new turn', async () => {
  const { playback, context, sources } = fixture();
  let completeSuspend;
  context.suspend = () => new Promise((resolve) => {
    completeSuspend = () => { context.state = 'suspended'; resolve(); };
  });
  context.resume = () => { context.state = 'running'; return Promise.resolve(); };
  playback.start('old');
  playback.enqueueAudio('old', { duration: 1 }, 'Old', [{ text: 'Old', start: 0, end: .8 }]);
  playback.pause('old');
  playback.clear();
  playback.start('new');
  playback.enqueueAudio('new', { duration: 1 }, 'New', [{ text: 'New', start: 0, end: .8 }]);
  completeSuspend();
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(context.state, 'running');
  assert.equal(sources[0].stopped, true);
  assert.equal(sources[1].stopped, false);
  playback.clear();
});

test('resume advances the output clock after the last source ended during pause', async () => {
  const { playback, context, events, sources } = fixture();
  let outputTime = 0;
  context.getOutputTimestamp = () => ({ contextTime: outputTime });
  context.suspend = () => { context.state = 'suspended'; return Promise.resolve(); };
  context.resume = () => { context.state = 'running'; return Promise.resolve(); };
  playback.start('t1');
  playback.enqueueAudio('t1', { duration: 1 }, 'Hello', [{ text: 'Hello', start: 0, end: .8 }]);
  playback.finish('t1');
  playback.pause('t1');
  sources[0].end();
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(events.some((event) => event[0] === 'drained'), false);
  playback.resume('t1');
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(context.state, 'running');
  outputTime = 1;
  await new Promise((resolve) => setTimeout(resolve, 40));
  assert.equal(events.filter((event) => event[0] === 'drained').length, 1);
  playback.clear();
});

test('failed pause does not discard or falsely finish the answer', async () => {
  const { playback, context, events, sources } = fixture();
  context.suspend = () => Promise.reject(new Error('suspend failed'));
  playback.start('t1');
  playback.enqueueAudio('t1', buffer('a'), 'Answer.');
  playback.pause('t1');
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(sources[0].stopped, false);
  assert.deepEqual(events.at(-1), ['pause-error', 't1', 'suspend failed']);
  assert.equal(events.some((event) => event[0] === 'drained'), false);
  playback.clear();
});
