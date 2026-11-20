"""Coordinate clearing session history with queued and active responses."""


def clear_context_if_idle(memory, pipeline_lock, request_lock, pending_requests, mic_enabled, audio_processing, browser_audio_active=None):
    """Return False if audio capture or any pipeline may still add a turn."""
    with request_lock:
        if (mic_enabled.is_set() or audio_processing.is_set() or pending_requests or
                (browser_audio_active is not None and browser_audio_active.is_set()) or
                not pipeline_lock.acquire(blocking=False)):
            return False
        try:
            memory.clear()
        finally:
            pipeline_lock.release()
    return True
