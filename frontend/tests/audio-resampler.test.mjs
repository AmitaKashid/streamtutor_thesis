import test from 'node:test';
import assert from 'node:assert/strict';
import { createAudioResampler } from '../app/audio-resampler.mjs';

test('preserves audio duration across many short microphone blocks', () => {
  for (const rate of [44100, 48000, 16000]) {
    const resampler = createAudioResampler(rate);
    let outputLength = 0;
    for (let i = 0; i < 1000; i++) outputLength += resampler.push(new Float32Array(128).fill(.5)).length;
    assert.ok(Math.abs(outputLength - Math.floor(128000 * 16000 / rate)) <= 1);
  }
});

test('output does not depend on worklet block boundaries', () => {
  const input = Float32Array.from({ length: 1024 }, (_, i) => Math.sin(i / 12));
  const single = createAudioResampler(44100).push(input);
  const split = createAudioResampler(44100);
  const chunks = [];
  for (let i = 0; i < input.length; i += 37) chunks.push(...split.push(input.slice(i, i + 37)));
  assert.deepEqual(chunks, [...single]);
});
