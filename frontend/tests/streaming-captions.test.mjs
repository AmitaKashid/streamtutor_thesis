import test from 'node:test';
import assert from 'node:assert/strict';
import { createStreamingCaptions } from '../app/streaming-captions.mjs';

function fixture(t) {
  let now = 0;
  const text = [];
  const captions = createStreamingCaptions({ clock: () => now, onText: (s) => text.push(s) });
  t.after(() => captions.clear());
  const advance = (time) => { now = time; captions.tick(); };
  const packet = (offset, at) => captions.scheduled({segmentId:'s',offset,frames:100,rate:1000},at);
  return {captions, text, advance, packet};
}

test('audio can play before timings arrive; late words catch up once without duplication', (t) => {
  const {captions, text, advance, packet} = fixture(t);
  captions.start('s', 'Hello world. ', 'acoustic');
  packet(0, 0); packet(100, .1);
  advance(.08);
  assert.deepEqual(text, []);
  captions.words('s', [{text:'Hello ',start:0,end:.05}]);
  assert.deepEqual(text, ['Hello ']);
  captions.words('s', [{text:'Hello ',start:0,end:.05}]); // duplicate is not a source prefix
  assert.deepEqual(text, ['Hello ']);
  captions.words('s', [{text:'world. ',start:.12,end:.18}]);
  advance(.14);
  assert.deepEqual(text, ['Hello ', 'world. ']);
});

test('network underrun freezes caption progress until the corresponding PCM plays', (t) => {
  const {captions, text, advance, packet} = fixture(t);
  captions.start('s', 'First second. ', 'acoustic');
  captions.words('s', [{text:'First ',start:0,end:.08},{text:'second. ',start:.1,end:.18}]);
  packet(0, 0);
  packet(100, 1); // 900 ms gap; same segment, discontinuous playback clock
  advance(.8);
  assert.deepEqual(text, ['First ']);
  advance(1.01);
  assert.deepEqual(text, ['First ', 'second. ']);
});

test('failed or slow alignment reveals the remaining text at the actual segment end', (t) => {
  const {captions, text, advance, packet} = fixture(t);
  captions.start('s', 'The complete sentence. ', 'acoustic');
  packet(0, 0); packet(100, 1);
  captions.end('s', 200, 1000);
  advance(1.05);
  assert.deepEqual(text, []);
  advance(1.1);
  assert.deepEqual(text, ['The complete sentence. ']);
  captions.words('s', [{text:'The ',start:0,end:.02}]);
  assert.deepEqual(text, ['The complete sentence. ']);
});

test('fallback segment caption waits for actual scheduled playback, not network arrival', (t) => {
  const {captions, text, advance, packet} = fixture(t);
  captions.start('s', 'This is a segment. ', 'segment');
  packet(0, 2);
  advance(1.9);
  assert.deepEqual(text, []);
  advance(2);
  assert.deepEqual(text, ['This is a segment. ']);
});

test('pause and cancellation discard scheduled and late captions', (t) => {
  const {captions, text, advance, packet} = fixture(t);
  captions.start('s', 'Hello world. ', 'acoustic');
  packet(0, 0); packet(100, .1);
  captions.words('s', [{text:'Hello ',start:0,end:.05},{text:'world. ',start:.12,end:.18}]);
  advance(.04);
  captions.pause(); advance(.15);
  assert.deepEqual(text, ['Hello ']);
  captions.resume();
  assert.deepEqual(text, ['Hello ', 'world. ']);
  captions.clear();
  captions.words('s', [{text:'ignored',start:0,end:0}]);
  advance(10);
  assert.equal(text.join(''), 'Hello world. ');
});

test('invalid updates cannot change displayed text and broken sample continuity is rejected', (t) => {
  const {captions, text, advance, packet} = fixture(t);
  captions.start('s', 'Valid text. ', 'acoustic'); packet(0, 0);
  captions.words('s', [{text:'Valid ',start:0,end:.03},{text:'invented',start:.04,end:.05}]);
  advance(.08);
  assert.deepEqual(text, []);
  assert.throws(() => packet(150, .1), /discontinuous/);
});
