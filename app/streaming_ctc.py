"""CPU acoustic tracking of a known English TTS transcript.

This is prefix alignment, not speech recognition: only characters from the
synthesis text can be displayed. No word-rate/duration guesses are made.
Wav2Vec2 is non-causal; overlapping windows provide context, and the last
160 ms is withheld from timing decisions. PCM never waits for this worker.
These are heuristic acoustic estimates, not calibrated or exact word timings.
"""
from __future__ import annotations

import json
import os
import re
import threading
import unicodedata
from pathlib import Path
from math import gcd

import numpy as np
from scipy.signal import resample_poly

MODEL_REVISION = 'a19f851b3d42865797e410752b4c570c871e4825'
MODEL_SHA256 = 'cd5040c147381580ed73258143dd8e0c28e800a09e74ee42ee2b3e8cb4d760a3'
MODEL_DIR = Path(__file__).resolve().parents[1] / 'runtime' / 'word-alignment'
SAMPLE_RATE = 16000
FRAME_SAMPLES = 320


def transcript_tokens(text, vocab):
    """Keep source spans exact while normalizing the acoustic target only."""
    from num2words import num2words
    ids, words, pending = [], [], ''
    for match in re.finditer(r'\S+\s*', text):
        display = pending + match.group()
        normalized = unicodedata.normalize('NFKD', match.group().replace('’', "'"))
        normalized = normalized.encode('ascii', 'ignore').decode()
        def number(m):
            try:
                return num2words(m.group().replace(',', ''), lang='en')
            except (ValueError, OverflowError, NotImplementedError):
                raise ValueError('Unsupported number in speech alignment')
        normalized = re.sub(r'\d[\d,]*(?:\.\d+)?', number, normalized)
        normalized = re.sub(r"[^A-Z']+", '|', normalized.upper()).strip("|'")
        if not normalized:
            if any(c.isalnum() for c in match.group()):
                raise ValueError('The streaming aligner supports English text only')
            if words:
                words[-1]['text'] += display
                pending = ''
            else:
                pending = display
            continue
        if ids:
            ids.append(vocab['|'])
        first = len(ids)
        ids.extend(vocab[c] for c in normalized)
        words.append({'text': display, 'first': first, 'last': len(ids) - 1})
        pending = ''
    if pending and words:
        words[-1]['text'] += pending
    if not words or len(ids) > 1200:
        raise ValueError('Speech alignment text is empty or too long')
    if ''.join(w['text'] for w in words) != text:
        raise ValueError('Speech alignment must preserve the source text')
    return np.array(ids, dtype=np.int32), words


def prefix_path(log_probs, targets, minimum_token=-1):
    """CTC Viterbi with an OPEN endpoint: unspoken suffixes are not forced in.

    Repeated labels require a blank. Unlike whole-utterance forced alignment,
    this never demands that the final token has already occurred.
    """
    labels = np.zeros(2 * len(targets) + 1, dtype=np.int32)
    labels[1::2] = targets
    scores = np.full(len(labels), -np.inf, dtype=np.float64)
    scores[0] = 0
    skip = np.zeros(len(labels), dtype=bool)
    skip[2:] = (labels[2:] != 0) & (labels[2:] != labels[:-2])
    backs = np.empty((len(log_probs), len(labels)), dtype=np.uint8)
    for frame, row in enumerate(log_probs):
        one = np.r_[-np.inf, scores[:-1]]
        two = np.r_[-np.inf, -np.inf, scores[:-2]]
        two[~skip] = -np.inf
        candidates = np.stack((scores, one, two))
        backs[frame] = candidates.argmax(axis=0)
        scores = candidates.max(axis=0) + row[labels]
    # Never retract a previously published prefix. A partial path may be less
    # probable later, but already displayed words must not move or repeat.
    floor = max(0, 2 * minimum_token + 1)
    state = floor + int(np.argmax(scores[floor:]))
    if not np.isfinite(scores[state]):
        return []
    trace = np.empty(len(log_probs), dtype=np.int32)
    for frame in range(len(log_probs) - 1, -1, -1):
        trace[frame] = state
        state -= int(backs[frame, state])
    spans = []
    for token_index, label in enumerate(targets):
        frames = np.flatnonzero(trace == 2 * token_index + 1)
        if not len(frames):
            break
        spans.append((int(frames[0]), int(frames[-1]) + 1,
                      float(np.exp(log_probs[frames, label]).max())))
    return spans


class CpuAcousticModel:
    def __init__(self, model_dir=MODEL_DIR):
        import onnxruntime as ort
        model_dir = Path(model_dir)
        self.vocab = json.loads((model_dir / 'vocab.json').read_text(encoding='utf-8'))
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(4, int(os.getenv('LLMTUTOR_ALIGNMENT_THREADS', '2'))))
        options.inter_op_num_threads = 1
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        self.session = ort.InferenceSession(str(model_dir / 'model_quantized.onnx'),
                                            sess_options=options, providers=['CPUExecutionProvider'])
        self.lock = threading.Lock()
        self.infer(np.zeros(8000, dtype=np.float32))  # warm before accepting turns

    def infer(self, audio):
        samples = np.asarray(audio, dtype=np.float32)
        samples = (samples - samples.mean()) / np.sqrt(samples.var() + 1e-7)
        with self.lock:
            logits = self.session.run(None, {'input_values': samples[None, :]})[0][0]
        shifted = logits - logits.max(axis=-1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))

    def start(self, text):
        return StreamingCtcAlignment(self, text)


class StreamingCtcAlignment:
    """Process coalesced cumulative PCM snapshots; emit only new source words."""
    def __init__(self, model, text):
        self.model = model
        self.text = text
        self.targets, self.words = transcript_tokens(text, model.vocab)
        self.emissions = np.empty((0, len(model.vocab)), dtype=np.float32)
        self.next_word = 0
        self.last_start = 0.0
        self.minimum_token = -1

    def update(self, audio, sample_rate, final=False, cancelled=None):
        if cancelled is not None and cancelled():
            return []
        samples = np.asarray(audio, dtype=np.float32)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        if samples.ndim != 1 or sample_rate <= 0 or not np.isfinite(samples).all():
            raise ValueError('Invalid streaming alignment audio')
        if len(samples) / sample_rate > 45:
            raise ValueError('Streaming alignment segment exceeds 45 seconds')
        if sample_rate != SAMPLE_RATE:
            divisor = gcd(int(sample_rate), SAMPLE_RATE)
            samples = resample_poly(samples, SAMPLE_RATE // divisor, int(sample_rate) // divisor).astype(np.float32)
        if len(samples) < (800 if final else 7680):
            return []
        # Fill every missing frame even if the alignment worker was coalesced
        # for several seconds. Bounded inference windows keep CPU cost linear.
        wanted = max(0, (len(samples) - 400) // FRAME_SAMPLES + 1 - (0 if final else 8))
        while len(self.emissions) < wanted:
            if cancelled is not None and cancelled():
                return []
            # ~1.6 s context, with up to .64 s newly committed acoustic frames.
            end_frame = min(wanted, len(self.emissions) + 32)
            audio_end = min(len(samples), (end_frame + (0 if final and end_frame == wanted else 8)) * FRAME_SAMPLES + 400)
            start = max(0, audio_end - 25600) // FRAME_SAMPLES * FRAME_SAMPLES
            begin_frame = start // FRAME_SAMPLES
            if begin_frame > len(self.emissions):
                raise ValueError('Gap in acoustic alignment frames')
            new = self.model.infer(samples[start:audio_end])
            count = min(end_frame - begin_frame, len(new))
            if begin_frame + count <= len(self.emissions):
                break
            # Refresh overlapping emissions with the new right context. Freezing
            # a weak window-edge prediction can otherwise stall all later words.
            # Published source words remain monotonic even if this path changes.
            self.emissions = np.concatenate((self.emissions[:begin_frame], new[:count]))
        if not len(self.emissions):
            return []
        spans = prefix_path(self.emissions, self.targets, self.minimum_token)
        result = []
        while self.next_word < len(self.words):
            word = self.words[self.next_word]
            # Require at least two character targets (one for one-letter
            # words). The evidence thresholds are heuristics, not confidence
            # probabilities; uncertain words remain for the browser fallback.
            required = min(word['last'], word['first'] + 1)
            if required >= len(spans):
                break
            observed_end = min(word['last'], len(spans) - 1)
            evidence = [s[2] for s in spans[word['first']:observed_end + 1]]
            if evidence[0] < .08 or max(evidence) < .5 or np.mean(evidence) < .35:
                break
            onset = spans[word['first']][0] * .02
            end = spans[min(word['last'], len(spans) - 1)][1] * .02
            if onset < self.last_start:
                break
            result.append({'text': word['text'], 'start': onset, 'end': end})
            self.last_start = onset
            self.minimum_token = required
            self.next_word += 1
        return result
