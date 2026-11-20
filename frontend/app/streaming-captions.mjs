/**
 * Map acoustic timestamps to ACTUAL scheduled PCM, including playback gaps.
 * No duration/word-count guessing; late timings catch up, never rewind text.
 * @param {{clock: () => number, onText: (text: string) => void}} callbacks
 */
export function createStreamingCaptions({ clock, onText }) {
  const segments = new Map();
  let timer = null;
  let paused = false;

  function clear() {
    if (timer !== null) clearTimeout(timer);
    timer = null;
    segments.clear();
    paused = false;
  }

  function tick() {
    if (timer !== null) clearTimeout(timer);
    timer = null;
    if (paused) return;
    const now = clock();
    let waiting = false;
    for (const segment of segments.values()) {
      if (segment.done) continue;
      waiting = true;
      let played = 0;
      let audible = false;
      for (const span of segment.spans) {
        if (now < span.at) break;
        audible = true;
        played = span.offset + Math.min(span.frames, Math.max(0, (now - span.at) * span.rate));
      }
      if (!audible) break;
      let prefix = segment.shown;
      if (segment.mode === 'segment') {
        prefix = segment.text.length;
      } else {
        for (const word of segment.words) {
          if (word.start * segment.rate >= played) break;
          prefix = Math.max(prefix, word.until);
        }
      }
      // Uncertain/missing/slow alignment never stalls audio or loses text.
      // Reveal the remaining suffix only after the segment has been heard.
      const finished = segment.total !== null && played >= segment.total - .01;
      if (finished) prefix = segment.text.length;
      if (prefix > segment.shown) {
        onText(segment.text.slice(segment.shown, prefix));
        segment.shown = prefix;
      }
      if (finished) {
        segment.done = true;
        segment.spans = [];
        segment.words = [];
        waiting = false;
      } else break; // Later segments must not overtake an unfinished caption.
    }
    if (waiting) timer = setTimeout(tick, 16);
  }

  return {
    /** @param {string} id @param {string} text @param {'acoustic'|'segment'} mode */
    start(id, text, mode) {
      if (segments.has(id)) return;
      segments.set(id, { text, mode, words: [], spans: [], shown: 0, matched: 0,
        rate: 0, scheduled: 0, total: null, done: false });
    },
    /** @param {{segmentId: string, offset: number, frames: number, rate: number}} packet @param {number} at */
    scheduled(packet, at) {
      const segment = segments.get(packet.segmentId);
      if (!segment || segment.done) throw new Error('Missing live speech segment');
      if (!Number.isSafeInteger(packet.offset) || packet.offset !== segment.scheduled ||
          !Number.isSafeInteger(packet.frames) || packet.frames <= 0 ||
          !Number.isFinite(packet.rate) || packet.rate <= 0 || !Number.isFinite(at) ||
          (segment.rate && segment.rate !== packet.rate) ||
          (segment.total !== null && packet.offset + packet.frames > segment.total)) {
        throw new Error('Invalid or discontinuous live PCM');
      }
      segment.rate = packet.rate;
      segment.scheduled += packet.frames;
      segment.spans.push({ ...packet, at });
      tick();
    },
    /** @param {string} id @param {number} total @param {number} rate */
    end(id, total, rate) {
      const segment = segments.get(id);
      if (!segment || segment.done) return;
      if (!Number.isSafeInteger(total) || total <= 0 || rate <= 0 ||
          (segment.rate && segment.rate !== rate) || total < segment.scheduled) {
        throw new Error('Invalid live speech end');
      }
      segment.total = total;
      segment.rate = rate;
      tick();
    },
    /** @param {string} id @param {Array<{text:string, start:number, end:number}>} words */
    words(id, words) {
      const segment = segments.get(id);
      if (!segment || segment.done || segment.mode !== 'acoustic') return;
      // Reject an entire invalid update without poisoning the accepted prefix.
      let matched = segment.matched;
      let lastStart = segment.words.at(-1)?.start ?? 0;
      const accepted = [];
      for (const word of words) {
        if (!word.text || !segment.text.startsWith(word.text, matched) ||
            !Number.isFinite(word.start) || !Number.isFinite(word.end) ||
            word.start < lastStart || word.end < word.start ||
            (segment.total !== null && word.end * segment.rate > segment.total + 1)) return;
        matched += word.text.length;
        accepted.push({ ...word, until: matched });
        lastStart = word.start;
      }
      segment.matched = matched;
      segment.words.push(...accepted);
      tick();
    },
    pause() { paused = true; if (timer !== null) clearTimeout(timer); timer = null; },
    resume() { paused = false; tick(); },
    tick,
    clear,
  };
}
