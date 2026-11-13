/** Tracks response generation independently from queued browser playback. */
export const idleResponse = () => ({ generating: false, audioPending: false, cancelling: false });

/** @param {{generating: boolean, audioPending: boolean, cancelling: boolean}} state
 *  @param {{type: string, audio?: boolean}} event */
export function responseActivity(state, event) {
  switch (event.type) {
    case 'submit':
      return { generating: true, audioPending: false, cancelling: false };
    case 'start':
      return { generating: true, audioPending: !state.cancelling && !!event.audio, cancelling: state.cancelling };
    case 'generation_done':
      return { ...state, generating: false, cancelling: false };
    case 'audio_done':
      return { ...state, audioPending: false };
    case 'stop':
      return { ...state, audioPending: false, cancelling: state.generating };
    case 'reset':
      return idleResponse();
    default:
      return state;
  }
}

/** @param {{generating: boolean, audioPending: boolean}} state */
export const responseIsActive = (state) => state.generating || state.audioPending;
