from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional
import json
import time

import requests

from .base import LLMProvider, GenerationResult


@dataclass
class RemoteHTTPProvider(LLMProvider):
    base_url: str
    connect_timeout: float = 3.0
    read_timeout: float = 120.0
    system_prompt: Optional[str] = None  # kept for interface compatibility

    def _url(self, path: str) -> str:
        return self.base_url.rstrip("/") + path

    def healthcheck(self) -> bool:
        try:
            resp = requests.get(
                self._url("/health"),
                timeout=(self.connect_timeout, self.connect_timeout),
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("status") == "ok"
        except Exception:
            return False

    def generate(self, user_text: str) -> GenerationResult:
        """
        Non-streaming API for batch mode or debugging.
        Expects server response JSON:
        {
          "text": "...",
          "prompt_tokens": 123,
          "new_tokens": 45,
          "total_seconds": 1.23
        }
        """
        t0 = time.perf_counter()

        resp = requests.post(
            self._url("/generate"),
            json={"prompt": user_text},
            timeout=(self.connect_timeout, self.read_timeout),
        )
        resp.raise_for_status()
        data = resp.json()

        total_seconds = float(data.get("total_seconds", time.perf_counter() - t0))
        prompt_tokens = int(data.get("prompt_tokens", 0))
        new_tokens = int(data.get("new_tokens", 0))
        text = (data.get("text") or "").strip()

        raw_value = data.get("raw_text")
        raw_text = str(raw_value) if raw_value is not None else None
        protocol_value = data.get("response_protocol")
        response_protocol = (
            str(protocol_value) if protocol_value is not None else None
        )
        error_value = data.get("protocol_error")
        protocol_error = str(error_value) if error_value is not None else None

        tps = (new_tokens / total_seconds) if total_seconds > 0 and new_tokens > 0 else 0.0

        return GenerationResult(
            text=text,
            total_seconds=total_seconds,
            prompt_tokens=prompt_tokens,
            new_tokens=new_tokens,
            tokens_per_second=tps,
            raw_text=raw_text,
            response_protocol=response_protocol,
            protocol_error=protocol_error,
        )

    def stream(self, user_text: str) -> Iterator[str]:
        yield from self.stream_with_metrics(user_text)

    def stream_with_metrics(self, user_text: str, on_server_metrics=None) -> Iterator[str]:
        """Consume NDJSON; old servers remain supported but cannot verify success."""
        with requests.post(self._url("/stream"), json={"prompt": user_text},
                           timeout=(self.connect_timeout, self.read_timeout), stream=True) as resp:
            resp.raise_for_status()
            done = False
            for line in resp.iter_lines(decode_unicode=False):
                if not line:
                    continue
                data = json.loads(line.decode('utf-8'))
                if data.get('error'):
                    raise RuntimeError('Remote model generation failed')
                if data.get('done'):
                    done = True
                    reason = data.get('finish_reason')
                    if reason and reason != 'stop':
                        raise RuntimeError(f'Remote model generation ended: {reason}')
                    if reason == 'stop' and on_server_metrics and isinstance(data.get('metrics'), dict):
                        on_server_metrics(data['metrics'])
                    break
                delta = data.get('delta')
                if delta:
                    yield delta
            if not done:
                raise RuntimeError('Remote model stream ended without completion')
