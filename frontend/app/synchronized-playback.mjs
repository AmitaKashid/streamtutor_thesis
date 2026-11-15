/**
 * Keeps text segments in the same ordered queue as their browser audio.
 * Text is revealed when its first AudioBufferSourceNode is started; engines do
 * not supply word timestamps, so this is segment-level, not word-level timing.
 *
 * @param {{
 *   getContext: () => AudioContext,
 *   onText: (turnId: string, text: string) => void,
 *   onDrained: (turnId: string) => void,
 *   onError: (turnId: string, error: unknown) => void,
 * }} callbacks
 */
export function createSynchronizedPlayback({ getContext, onText, onDrained, onError }) {
  /** @type {Array<{buffer: AudioBuffer | null, text: string}>} */
  let queue = [];
  /** @type {AudioBufferSourceNode | null} */
  let source = null;
  /** @type {string | null} */
  let turnId = null;
  let epoch = 0;
  let resuming = false;
  let ended = false;
  let drained = false;

  function clear() {
    epoch += 1;
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

  function pump() {
    if (!turnId || source || resuming) return;
    const currentTurn = turnId;
    const currentEpoch = epoch;

    while (queue.length && !queue[0].buffer) {
      const item = queue.shift();
      if (item?.text) onText(currentTurn, item.text);
    }
    if (!queue.length) {
      if (ended && !drained) {
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
      next.onended = () => {
        if (epoch !== currentEpoch || source !== next) return;
        source = null;
        pump();
      };
      next.start();
      if (item.text) onText(currentTurn, item.text);
    } catch (error) {
      onError(currentTurn, error);
    }
  }

  return {
    /** @param {string} id */
    start(id) { clear(); turnId = id; },
    /** @param {string} id @param {AudioBuffer} buffer @param {string} [text] */
    enqueueAudio(id, buffer, text = '') {
      if (id !== turnId || ended) return;
      queue.push({ buffer, text });
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
