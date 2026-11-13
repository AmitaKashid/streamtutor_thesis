"""Normalize optional llama-server timings without inventing missing values."""
from __future__ import annotations

import math


def server_timing_metrics(timings: object) -> dict:
    if not isinstance(timings, dict):
        return {}

    def count(name):
        value = timings.get(name)
        return value if type(value) is int and value >= 0 else None

    def seconds(name):
        value = timings.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return value / 1000 if math.isfinite(value) and value >= 0 else None

    def nonnegative_number(name):
        value = timings.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return value if math.isfinite(value) and value >= 0 else None

    processed = count('prompt_n')
    cached = count('cache_n')
    output = count('predicted_n')
    prefill = seconds('prompt_ms')
    decode = seconds('predicted_ms')
    decode_rate = nonnegative_number('predicted_per_second')
    metrics = {}
    if processed is not None and cached is not None:
        metrics['serverPromptTokens'] = processed + cached
        metrics['serverCachedTokens'] = cached
        metrics['serverProcessedPromptTokens'] = processed
    if output is not None:
        metrics['serverOutputTokens'] = output
    if prefill is not None:
        metrics['serverPrefillSeconds'] = prefill
    if decode is not None:
        metrics['serverDecodeSeconds'] = decode
    if decode_rate is not None and decode is not None and decode > 0:
        metrics['serverDecodeTokensPerSecond'] = decode_rate
    return metrics
