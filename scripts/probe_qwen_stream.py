"""Measure the existing local Qwen server, independently of the browser/LLM."""
from __future__ import annotations

import argparse
import base64
import json
import statistics
import time

import websocket


DEFAULT_TEXT = (
    'A decision tree predicts an outcome by splitting examples into smaller groups. '
    'Each split asks a question about one feature.'
)


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def measure(url, text, speaker, language, chunk_size, timeout):
    ws = websocket.create_connection(url, timeout=timeout)
    try:
        start = time.perf_counter()
        ws.send(json.dumps({'type': 'synthesize', 'text': text, 'speaker': speaker,
                            'language': language, 'chunk_size': chunk_size}))
        first = None
        packet_times = []
        total_bytes = 0
        sample_rate = None
        while True:
            raw = ws.recv()
            if not raw:
                raise RuntimeError('Server disconnected before audio_done')
            msg = json.loads(raw)
            if msg.get('type') == 'error':
                raise RuntimeError(msg.get('message', 'Qwen server error'))
            if msg.get('type') == 'audio_chunk':
                now = time.perf_counter()
                audio = base64.b64decode(msg['pcm16_base64'], validate=True)
                if not audio or len(audio) % 2:
                    raise RuntimeError('Invalid PCM16 audio from server')
                rate = int(msg.get('sample_rate') or 24000)
                if sample_rate is not None and rate != sample_rate:
                    raise RuntimeError('Sample rate changed during a segment')
                sample_rate = rate
                first = first or now
                packet_times.append(now)
                total_bytes += len(audio)
            if msg.get('type') == 'audio_done':
                end = time.perf_counter()
                if first is None or sample_rate is None:
                    raise RuntimeError('Server completed without audio')
                duration = total_bytes / (2 * sample_rate)  # mono signed PCM16
                wall = end - start
                gaps = [b - a for a, b in zip(packet_times, packet_times[1:])]
                return {'ttfa_s': first - start, 'total_s': wall, 'audio_s': duration,
                        'audio_per_wall': duration / wall, 'packets': len(packet_times),
                        'max_packet_gap_s': max(gaps, default=0.0)}
    finally:
        ws.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='ws://127.0.0.1:8094/stream')
    parser.add_argument('--text', default=DEFAULT_TEXT)
    parser.add_argument('--speaker', default='Aiden')
    parser.add_argument('--language', default='en')
    parser.add_argument('--chunk-size', type=int, default=4)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=180)
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0 or args.chunk_size < 1:
        parser.error('runs and chunk-size must be positive; warmup must be non-negative')
    for index in range(args.warmup):
        print(f'Warm-up {index + 1}/{args.warmup}', flush=True)
        measure(args.url, args.text, args.speaker, args.language, args.chunk_size, args.timeout)
    rows = []
    for index in range(args.runs):
        row = measure(args.url, args.text, args.speaker, args.language, args.chunk_size, args.timeout)
        rows.append(row)
        print(f"Run {index + 1:02d}: TTFA={row['ttfa_s']:.3f}s  "
              f"total={row['total_s']:.3f}s  audio={row['audio_s']:.3f}s  "
              f"audio/wall={row['audio_per_wall']:.2f}x  packets={row['packets']}  "
              f"max_gap={row['max_packet_gap_s']:.3f}s", flush=True)
    ttfa = [row['ttfa_s'] for row in rows]
    print(f"\nServer-only TTFA: p50={statistics.median(ttfa):.3f}s "
          f"p95={percentile(ttfa, .95):.3f}s; runs={len(rows)}")
    print('audio/wall > 1 means the server generated audio faster than its playback duration.')
    print('These numbers exclude microphone, STT, LLM, StreamTutor alignment, browser and speaker.')


if __name__ == '__main__':
    main()
