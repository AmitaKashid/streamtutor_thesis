"""Track the assistant text actually released to a browser, per turn.

Generation, audio transmission and audible playback finish at different
times. Never put generated-but-unpresented text into conversational memory.
"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)


class ConversationDelivery:
    def __init__(self, memory):
        self.memory = memory
        self.lock = threading.RLock()
        self.turns = {}

    def start(self, turn_id, user_text, remember):
        with self.lock:
            self.turns[turn_id] = {
                'user': user_text, 'remember': remember,
                'expected': '', 'displayed': '',
            }

    def expect(self, turn_id, text):
        if not isinstance(text, str) or not text:
            return
        with self.lock:
            turn = self.turns.get(turn_id)
            if turn is not None:
                turn['expected'] += text

    def progress(self, turn_id, text):
        if not isinstance(text, str) or not text:
            return False
        with self.lock:
            turn = self.turns.get(turn_id)
            if turn is None:
                return False
            candidate = turn['displayed'] + text
            # Accept only a prefix of text the backend actually sent for this
            # turn. Late/duplicated/malformed browser events cannot add claims.
            if not turn['expected'].startswith(candidate):
                log.warning('Ignored invalid or out-of-order caption progress for %s', turn_id)
                return False
            turn['displayed'] = candidate
            return True

    def visible_chunks(self, turn_id):
        with self.lock:
            turn = self.turns.get(turn_id)
            # Preserve exact whitespace: finish() validates this snapshot as a
            # prefix, including any leading or trailing whitespace.
            text = turn['displayed'] if turn else ''
            return [text] if text.strip() else []

    def finish(self, turn_id, delivered, visible_at_stop=None):
        """Commit once, including only the caption prefix released to the UI."""
        with self.lock:
            turn = self.turns.pop(turn_id, None)
            if turn is None or not turn['remember']:
                return
            shown = turn['displayed']
            if visible_at_stop is not None:
                # Snapshot from microphone onset must be a genuine prefix.
                candidate = visible_at_stop[0] if visible_at_stop else ''
                if shown.startswith(candidate):
                    shown = candidate
            shown = shown.strip()
            if not shown:
                answer = '[The assistant reply ended before any text was displayed.]'
            elif not delivered or shown != turn['expected'].strip():
                answer = '[The assistant reply was interrupted or only partially displayed. Text displayed:]\n' + shown
            else:
                answer = shown
            self.memory.add_turn(user_text=turn['user'], assistant_text=answer)

    def clear(self):
        with self.lock:
            self.turns.clear()
