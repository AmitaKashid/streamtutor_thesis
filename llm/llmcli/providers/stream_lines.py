"""Read streaming HTTP lines without waiting for a fixed-size network buffer."""
from __future__ import annotations

from collections.abc import Iterator


def iter_stream_lines(response) -> Iterator[bytes]:
    """Yield byte lines promptly, including across UTF-8 and CRLF boundaries.

    urllib3's read1 returns currently available bytes rather than waiting to
    fill the requested size. Fall back to Requests' one-byte reads when an
    adapter does not expose read1; line decoding remains the caller's job.
    """
    raw = getattr(response, 'raw', None)
    read1 = getattr(raw, 'read1', None)
    if not callable(read1):
        yield from response.iter_lines(chunk_size=1, decode_unicode=False)
        return

    pending = bytearray()
    while True:
        chunk = read1(8192, decode_content=True)
        if not chunk:
            break
        pending.extend(chunk)
        while True:
            lf = pending.find(b'\n')
            cr = pending.find(b'\r')
            if lf < 0 and cr < 0:
                break
            index = lf if cr < 0 else cr if lf < 0 else min(lf, cr)
            line = bytes(pending[:index])
            delimiter_len = 2 if pending[index:index + 2] == b'\r\n' else 1
            del pending[:index + delimiter_len]
            yield line
    if pending:
        yield bytes(pending)
