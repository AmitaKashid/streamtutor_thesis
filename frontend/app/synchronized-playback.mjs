/**
 * Keeps text segments in the same ordered queue as their browser audio.
 * Timed words are revealed according to the Web Audio playback clock.
 *
 * @param {{
 *   getContext: () => AudioContext,
 *   onText: (turnId: string, text: string) => void,
 *   onDrained: (turnId: string) => void,
 *   onError: (turnId: string, error: unknown) => void,
 * }} callbacks
 */
export function createSynchronizedPlayback({ getContext, onText, onDrained, onError }) {
  /** @type {Array<{buffer: AudioBuffer | null, text: string, words?: Array<{text: string, start: number, end: number}>}>} */
  let queue = [];
  /** @type {AudioBufferSourceNode | null} */
  let source = null;
  /** @type {string | null} */
  let turnId = null;
  let epoch = 0;
  let resuming = false;
  let ended = false;
  let drained = false;
  /** @type {Array<{text: string, at: number}>} */
  let pendingWords = [];
  let pendingOutputUntil = 0;
  /** @type {ReturnType<typeof setTimeout> | null} */
  let textTimer = null;

  function clear() {
    epoch += 1;
    if (textTimer !== null) clearTimeout(textTimer);
    textTimer = null;
    pendingWords = [];
    pendingOutputUntil = 0;
    queue = [];
    resuming = false;
    ended = false;
    drained = false;
    const oldSource = source;
    source = null;
    if (oldSource) {
      oldSource.onended = null;
      try { oldSource.stop(); } catch { /* already stopped */ }
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

  /** @param {AudioContext} ctx @param {string} currentTurn @param {number} currentEpoch */
  function revealDueWords(ctx, currentTurn, currentEpoch) {
    if (epoch !== currentEpoch || turnId !== currentTurn) return;
    textTimer = null;
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

  function pump() {
    if (!turnId || source || resuming) return;
    const currentTurn = turnId;
    const currentEpoch = epoch;

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
        resuming = true;
        void ctx.resume().then(() => {
          if (epoch !== currentEpoch || turnId !== currentTurn) return;
          resuming = false;
          if (ctx.state !== 'running') throw new Error('Audio context is not running');
          pump();
        }).catch((error) => {
          if (epoch !== currentEpoch || turnId !== currentTurn) return;
          resuming = false;
          onError(currentTurn, error);
        });
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
      next.onended = () => {
        if (epoch !== currentEpoch || source !== next) return;
        source = null;
        pump();
      };
      next.start();
      if (Number.isFinite(item.buffer.duration)) {
        pendingOutputUntil = Math.max(pendingOutputUntil, startedAt + item.buffer.duration);
      }
      if (words.length) {
        pendingWords.push(...words.map((word) => ({ text: word.text, at: startedAt + word.start })));
        if (textTimer === null) revealDueWords(ctx, currentTurn, currentEpoch);
      }
      else if (item.text) onText(currentTurn, item.text);
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
    clear,
  };
}
