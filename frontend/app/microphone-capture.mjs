/** Request browser AEC as a requirement where supported, then fall back. */
export async function openEchoCancelledMicrophone(mediaDevices) {
  const common = { noiseSuppression: true, autoGainControl: true, channelCount: 1 };
  let stream;
  try {
    stream = await mediaDevices.getUserMedia({ audio: { ...common, echoCancellation: { exact: true } } });
  } catch (error) {
    if (error?.name !== 'OverconstrainedError' ||
        (error.constraint && error.constraint !== 'echoCancellation')) throw error;
    stream = await mediaDevices.getUserMedia({ audio: { ...common, echoCancellation: true } });
  }
  return stream;
}
