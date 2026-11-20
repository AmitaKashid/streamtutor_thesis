/**
 * Keeps text segments in the same ordered queue as their browser audio.
 * Timed words are revealed according to the Web Audio playback clock.
 *
 * @param {{
 *   getContext: () => AudioContext,
 *   onText: (turnId: string, text: string) => void,
 *   onDrained: (turnId: string) => void,
 *   onError: (turnId: string, error: unknown) => void,
 *   onUnderrun?: (turnId: string, seconds: number) => void,
 *   onPauseError?: (turnId: string, error: unknown) => void,
 * }} callbacks
 */
export function createSynchronizedPlayback({ getContext, onText, onDrained, onError, onUnderrun, onPauseError }) {
  /** @type {Array<{buffer: AudioBuffer | null, text: string, words?: Array<{text: string, start: number, end: number}>}>} */
  let queue = [];
  /** @type {AudioBufferSourceNode | null} */
  let source = null;
  let sourceEnd = 0;
  /** @type {Array<{node: AudioBufferSourceNode, end: number, ended: boolean}>} */
  let scheduled = [];
  // Bound the number of future Web Audio nodes without waiting for callbacks
  // after every segment. Audio already received remains in the ordinary queue.
  const maxScheduledNodes = 16;
  /** @type {string | null} */
  let turnId = null;
  let epoch = 0;
  let resuming = false;
  let paused = false;
  let transitionId = 0;
  let ended = false;
  let drained = false;
  /** @type {Array<{text: string, at: number}>} */
  let pendingWords = [];
  let pendingOutputUntil = 0;
  let lastScheduledEnd = 0;
  /** @type {ReturnType<typeof setTimeout> | null} */
  let textTimer = null;

  function clear() {
    epoch += 1;
    if (textTimer !== null) clearTimeout(textTimer);
    textTimer = null;
    pendingWords = [];
    pendingOutputUntil = 0;
    lastScheduledEnd = 0;
    queue = [];
    transitionId += 1;
    resuming = false;
    paused = false;
    ended = false;
    drained = false;
    const oldSource = source;
    source = null;
    const oldScheduled = scheduled;
    scheduled = [];
    sourceEnd = 0;
    if (oldSource) {
      oldSource.onended = null;
      try { oldSource.stop(); } catch { /* already stopped */ }
    }
    for (const item of oldScheduled) {
      item.node.onended = null;
      try { item.node.stop(); } catch { /* already stopped */ }
    }
    turnId = null;
  }

  /** @param {AudioContext} ctx */
  function outputClock(ctx) {
    try {
      const stamp = typeof ctx.getOutputTimestamp === 'function'
        ? ctx.getOutputTimestamp().contextTime : ctx.currentTime;
      return Number.isFinite(stamp) ? stamp : ctx.currentTime;
    } catch {
      return ctx.currentTime;
    }
  }

  /** @param {number} startedAt @param {number} duration @param {string} currentTurn */
  function trackSegmentTiming(startedAt, duration, currentTurn) {
    if (lastScheduledEnd > 0 && startedAt - lastScheduledEnd > 0.05) {
      onUnderrun?.(currentTurn, startedAt - lastScheduledEnd);
    }
    if (Number.isFinite(duration)) lastScheduledEnd = startedAt + duration;
  }

  /** @param {AudioContext} ctx @param {string} currentTurn @param {number} currentEpoch */
  function revealDueWords(ctx, currentTurn, currentEpoch) {
    if (epoch !== currentEpoch || turnId !== currentTurn) return;
    textTimer = null;
    if (paused) return;
    const outputTime = outputClock(ctx);
    let text = '';
    while (pendingWords.length && pendingWords[0].at <= outputTime) {
      text += pendingWords.shift()?.text || '';
    }
    if (text) onText(currentTurn, text);
    if (pendingWords.length) {
      textTimer = setTimeout(() => revealDueWords(ctx, currentTurn, currentEpoch), 16);
    } else if (!source) {
      pump();
    }
  }

  /** @param {AudioContext} ctx @param {string} currentTurn @param {number} currentEpoch */
  function scheduleAhead(ctx, currentTurn, currentEpoch) {
    if (!source || !sourceEnd) return;
    let tailEnd = scheduled.length ? scheduled[scheduled.length - 1].end : sourceEnd;
    while (queue.length && scheduled.length < maxScheduledNodes) {
      const item = queue[0];
      if (!item.buffer) {
        queue.shift();
        if (item.text) pendingWords.push({ text: item.text, at: tailEnd });
        continue;
      }
      if (!Number.isFinite(item.buffer.duration)) break;
      queue.shift();
      const next = ctx.createBufferSource();
      next.buffer = item.buffer;
      next.connect(ctx.destination);
      const startedAt = Math.max(ctx.currentTime, tailEnd);
      const end = startedAt + item.buffer.duration;
      next.onended = () => handleEnd(next, ctx, currentTurn, currentEpoch);
      next.start(startedAt);
      trackSegmentTiming(startedAt, item.buffer.duration, currentTurn);
      scheduled.push({ node: next, end, ended: false });
      pendingOutputUntil = Math.max(pendingOutputUntil, end);
      if (item.words?.length) {
        pendingWords.push(...item.words.map((word) => ({ text: word.text, at: startedAt + word.start })));
      } else if (item.text) {
        pendingWords.push({ text: item.text, at: startedAt });
      }
      tailEnd = end;
    }
    if (pendingWords.length && textTimer === null) revealDueWords(ctx, currentTurn, currentEpoch);
  }

  /** @param {AudioBufferSourceNode} node @param {AudioContext} ctx @param {string} currentTurn @param {number} currentEpoch */
  function handleEnd(node, ctx, currentTurn, currentEpoch) {
    if (epoch !== currentEpoch || turnId !== currentTurn) return;
    if (source !== node) {
      const future = scheduled.find((item) => item.node === node);
      if (future) future.ended = true;
      return;
    }
    source = null;
    while (scheduled.length && !source) {
      const next = scheduled.shift();
      if (next && !next.ended) {
        source = next.node;
        sourceEnd = next.end;
      }
    }
    if (source) {
      try { scheduleAhead(ctx, currentTurn, currentEpoch); }
      catch (error) { onError(currentTurn, error); }
    } else {
      sourceEnd = 0;
      pump();
    }
  }

  /** @param {AudioContext} ctx @param {string} currentTurn @param {number} currentEpoch */
  function requestResume(ctx, currentTurn, currentEpoch) {
    const transition = ++transitionId;
    resuming = true;
    void ctx.resume().then(() => {
      if (transition !== transitionId) { if (turnId) pump(); return; }
      resuming = false;
      if (epoch !== currentEpoch || turnId !== currentTurn) {
        if (turnId) pump();
        return;
      }
      if (ctx.state !== 'running') throw new Error('Audio context is not running');
      pump();
    }).catch((error) => {
      if (transition !== transitionId) { if (turnId) pump(); return; }
      resuming = false;
      if (epoch !== currentEpoch || turnId !== currentTurn) {
        if (turnId) pump();
        return;
      }
      onError(currentTurn, error);
    });
  }

  function pump() {
    if (!turnId || resuming) return;
    const currentTurn = turnId;
    const currentEpoch = epoch;
    if (paused) {
      try {
        const ctx = getContext();
        if (ctx.state === 'running') {
          const transition = ++transitionId;
          resuming = true;
          void ctx.suspend().then(() => {
            if (transition !== transitionId) { if (turnId) pump(); return; }
            resuming = false;
            if (turnId) pump();
          }).catch((error) => {
            if (transition !== transitionId) { if (turnId) pump(); return; }
            resuming = false;
            if (epoch === currentEpoch && turnId === currentTurn && paused) {
              paused = false;
              onPauseError?.(currentTurn, error);
            }
            if (turnId) pump();
          });
        }
      } catch (error) {
        paused = false;
        onPauseError?.(currentTurn, error);
        pump();
      }
      return;
    }
    if (source || pendingWords.length || pendingOutputUntil > 0 || queue[0]?.buffer) {
      const ctx = getContext();
      if (ctx.state !== 'running') {
        requestResume(ctx, currentTurn, currentEpoch);
        return;
      }
    }
    if (source) {
      const ctx = getContext();
      if (pendingWords.length && textTimer === null) revealDueWords(ctx, currentTurn, currentEpoch);
      try { scheduleAhead(ctx, currentTurn, currentEpoch); }
      catch (error) { onError(currentTurn, error); }
      return;
    }

    while (queue.length && !queue[0].buffer) {
      if (pendingWords.length) return;
      if (pendingOutputUntil > 0) {
        const ctx = getContext();
        if (pendingOutputUntil > outputClock(ctx)) {
          if (textTimer === null) textTimer = setTimeout(() => revealDueWords(ctx, currentTurn, currentEpoch), 16);
          return;
        }
      }
      const item = queue.shift();
      if (item?.text) onText(currentTurn, item.text);
    }
    if (!queue.length) {
      if (ended && !drained && !pendingWords.length) {
        if (pendingOutputUntil > 0) {
          const ctx = getContext();
          if (pendingOutputUntil > outputClock(ctx)) {
            if (textTimer === null) textTimer = setTimeout(() => revealDueWords(ctx, currentTurn, currentEpoch), 16);
            return;
          }
        }
        drained = true;
        onDrained(currentTurn);
      }
      return;
    }

    try {
      const ctx = getContext();
      if (ctx.state !== 'running') {
        requestResume(ctx, currentTurn, currentEpoch);
        return;
      }

      const item = queue.shift();
      if (!item?.buffer) return;
      const next = ctx.createBufferSource();
      next.buffer = item.buffer;
      next.connect(ctx.destination);
      source = next;
      const startedAt = ctx.currentTime;
      const words = item.words || [];
      sourceEnd = Number.isFinite(item.buffer.duration) ? startedAt + item.buffer.duration : 0;
      next.onended = () => handleEnd(next, ctx, currentTurn, currentEpoch);
      next.start(startedAt);
      trackSegmentTiming(startedAt, item.buffer.duration, currentTurn);
      if (Number.isFinite(item.buffer.duration)) {
        pendingOutputUntil = Math.max(pendingOutputUntil, startedAt + item.buffer.duration);
      }
      if (words.length) {
        pendingWords.push(...words.map((word) => ({ text: word.text, at: startedAt + word.start })));
        if (textTimer === null) revealDueWords(ctx, currentTurn, currentEpoch);
      }
      else if (item.text) {
        pendingWords.push({ text: item.text, at: startedAt });
        if (textTimer === null) revealDueWords(ctx, currentTurn, currentEpoch);
      }
      scheduleAhead(ctx, currentTurn, currentEpoch);
    } catch (error) {
      onError(currentTurn, error);
    }
  }

  return {
    /** @param {string} id */
    start(id) { clear(); turnId = id; },
    /** @param {string} id @param {AudioBuffer} buffer @param {string} [text] @param {Array<{text: string, start: number, end: number}>} [words] */
    enqueueAudio(id, buffer, text = '', words = []) {
      if (id !== turnId || ended) return;
      queue.push({ buffer, text, words });
      pump();
    },
    /** @param {string} id @param {string} text */
    enqueueText(id, text) {
      if (id !== turnId || ended) return;
      queue.push({ buffer: null, text });
      pump();
    },
    /** @param {string} id */
    finish(id) {
      if (id !== turnId || ended) return;
      ended = true;
      pump();
    },
    /** @param {string} id */
    pause(id) {
      if (id !== turnId || drained || paused) return;
      paused = true;
      if (textTimer !== null) clearTimeout(textTimer);
      textTimer = null;
      pump();
    },
    /** @param {string} id */
    resume(id) {
      if (id !== turnId || drained || !paused) return;
      paused = false;
      pump();
    },
    clear,
  };
}
