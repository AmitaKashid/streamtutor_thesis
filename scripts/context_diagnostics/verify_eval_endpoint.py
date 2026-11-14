"""Fail closed unless a healthy llama-server reports the intended GGUF filename."""
from __future__ import annotations
import argparse
import sys
import requests

EXPECTED = 'Meta-Llama-3-8B-Instruct.Q4_K_M.gguf'


def check_endpoint(url, expected=EXPECTED):
    root = url.rstrip('/')
    health = requests.get(root + '/health', timeout=(5, 10))
    health.raise_for_status()
    if health.json().get('status') != 'ok':
        raise ValueError('Endpoint is not ready')
    response = requests.get(root + '/v1/models', timeout=(5, 10))
    response.raise_for_status()
    payload = response.json()
    rows = payload.get('data')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Model list is empty or invalid')
    ids = [item.get('id') for item in rows if isinstance(item, dict) and isinstance(item.get('id'), str)]
    names = [item.replace('\\', '/').rsplit('/', 1)[-1] for item in ids]
    if len(ids) != len(rows) or any(name.casefold() != expected.casefold() for name in names):
        raise ValueError(f'Model ID mismatch: expected {expected!r}, observed {ids!r}')
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    args = parser.parse_args()
    try:
        ids = check_endpoint(args.url)
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        print(f'Endpoint verification failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    print(f'Endpoint healthy; reported model IDs: {ids}')
    print('Filename identity only; verify model hash and hardware separately for formal evaluation.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
