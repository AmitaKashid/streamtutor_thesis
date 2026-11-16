/** Preserve the fractional sample position across AudioWorklet blocks. */
export function createAudioResampler(inputRate, outputRate = 16000) {
  if (!(Number.isFinite(inputRate) && inputRate > 0 && Number.isFinite(outputRate) && outputRate > 0)) {
    throw new Error('Invalid audio sample rate');
  }
  const ratio = inputRate / outputRate;
  let remaining = ratio;
  let sum = 0;
  return {
    /** @param {Float32Array} input */
    push(input) {
      const result = [];
      for (const sample of input) {
        let width = 1;
        while (width > 1e-10) {
          const part = Math.min(width, remaining);
          sum += sample * part;
          remaining -= part;
          width -= part;
          if (remaining < 1e-10) {
            result.push(sum / ratio);
            remaining = ratio;
            sum = 0;
          }
        }
      }
      return Float32Array.from(result);
    },
  };
}
