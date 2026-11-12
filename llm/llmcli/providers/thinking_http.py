"""Streaming client for a llama-server configured for separated reasoning."""
import json
import time
import requests
from app.thinking_stream import ThinkingStream, ThinkingStreamError, ReasoningNotObserved
from .base import GenerationResult


class ThinkingHTTPProvider:
    def __init__(self, base_url, read_timeout=900):
        self.base_url = base_url.rstrip('/')
        self.read_timeout = max(read_timeout, 900)

    def healthcheck(self):
        try:
            response = requests.get(self.base_url + '/health', timeout=3)
            response.raise_for_status()
            return response.json().get('status') == 'ok'
        except (requests.RequestException, ValueError):
            return False

    def stream(self, prompt):
        # Retry once only when no answer has been released. Some model runs
        # omit reasoning despite the requested template; never forward them.
        for attempt in range(2):
            try:
                yield from self._stream_once(prompt)
                return
            except ReasoningNotObserved:
                if attempt:
                    raise

    def _stream_once(self, prompt):
        body = {'messages': [{'role': 'system', 'content': 'You are a concise AI tutor for data mining. Use provided course context when available. Answer clearly and briefly.'}, {'role': 'user', 'content': prompt}], 'stream': True, 'max_tokens': 8192, 'temperature': 1.0, 'top_p': 0.95, 'chat_template_kwargs': {'enable_thinking': True}, 'reasoning_format': 'deepseek'}
        parser = ThinkingStream()
        with requests.post(self.base_url + '/v1/chat/completions', json=body, stream=True, timeout=(3, self.read_timeout)) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    parser.end()
                    return
                try:
                    value = parser.feed(json.loads(data))
                except (ValueError, TypeError, KeyError) as exc:
                    raise ThinkingStreamError('Invalid model-server stream') from exc
                if value:
                    yield value
            raise ThinkingStreamError('Model server closed stream without DONE')

    def generate(self, prompt):
        start = time.perf_counter()
        text = ''.join(self.stream(prompt))
        return GenerationResult(text=text, total_seconds=time.perf_counter()-start, prompt_tokens=0, new_tokens=0, tokens_per_second=0.0)
