"""Streaming client for a llama-server configured for separated reasoning."""
import json
import os
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
                print('[THINKING RETRY] no separated reasoning; retrying once', flush=True)

    def _stream_once(self, prompt):
        body = {'messages': [{'role': 'system', 'content': 'You are a concise AI tutor for data mining. Use provided course context when available. Answer clearly and briefly.'}, {'role': 'user', 'content': prompt}], 'stream': True, 'max_tokens': 8192, 'temperature': 1.0, 'top_p': 0.95, 'chat_template_kwargs': {'enable_thinking': True}, 'reasoning_format': 'deepseek'}
        budget = os.environ.get('LLMTUTOR_THINKING_BUDGET', '').strip()
        if budget:
            if not budget.isdigit() or int(budget) < 1:
                raise ValueError('LLMTUTOR_THINKING_BUDGET must be a positive integer')
            body['reasoning_budget_tokens'] = int(budget)
        parser = ThinkingStream()
        started = time.perf_counter()
        first_reasoning = None
        first_content = None
        reasoning_chars = 0
        with requests.post(self.base_url + '/v1/chat/completions', json=body, stream=True, timeout=(3, self.read_timeout)) as response:
            response.raise_for_status()
            # SSE is UTF-8. Decode bytes explicitly; requests otherwise may
            # fall back to ISO-8859-1 when the server omits a charset.
            for raw_line in response.iter_lines(decode_unicode=False):
                try:
                    line = raw_line.decode('utf-8')
                except UnicodeDecodeError as exc:
                    raise ThinkingStreamError('Invalid UTF-8 from model server') from exc
                if not line or not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    parser.end()
                    def seconds(value):
                        return 'N/A' if value is None else f'{value:.3f}s'
                    print(f'[THINKING TIMING] first_reasoning={seconds(first_reasoning)} '
                          f'first_answer={seconds(first_content)} '
                          f'reasoning_chars={reasoning_chars} '
                          f'budget={budget or "server-default"} '
                          f'total={time.perf_counter() - started:.3f}s', flush=True)
                    return
                try:
                    event = json.loads(data)
                    choices = event.get('choices') or []
                    if choices:
                        delta = choices[0].get('delta') or {}
                        reasoning = delta.get('reasoning_content') or ''
                        content = delta.get('content') or ''
                        if isinstance(reasoning, str) and reasoning:
                            reasoning_chars += len(reasoning)
                            if first_reasoning is None:
                                first_reasoning = time.perf_counter() - started
                        if isinstance(content, str) and content and first_content is None:
                            first_content = time.perf_counter() - started
                    value = parser.feed(event)
                except (ValueError, TypeError, KeyError) as exc:
                    raise ThinkingStreamError('Invalid model-server stream') from exc
                if value:
                    yield value
            raise ThinkingStreamError('Model server closed stream without DONE')

    def generate(self, prompt):
        start = time.perf_counter()
        text = ''.join(self.stream(prompt))
        return GenerationResult(text=text, total_seconds=time.perf_counter()-start, prompt_tokens=0, new_tokens=0, tokens_per_second=0.0)
