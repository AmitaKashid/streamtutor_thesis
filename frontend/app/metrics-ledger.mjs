/** Keep metrics tied to a turn even if WebSocket events arrive late. */
export function createMetricsLedger() {
  const turns = new Map();
  const completed = [];

  function entry(id) {
    if (!turns.has(id)) turns.set(id, { metrics: {}, started: false, completed: false });
    return turns.get(id);
  }

  return {
    start(id, model) {
      const turn = entry(id);
      turn.started = true;
      if (model != null) turn.metrics = { ...turn.metrics, model };
      return { ...turn.metrics };
    },
    update(id, incoming) {
      const turn = entry(id);
      const values = { ...incoming };
      // A text turn reports STT=0 as a placeholder, not a measurement.
      if (values.stt != null && values.stt <= 0) delete values.stt;
      turn.metrics = { ...turn.metrics, ...values };
      return { ...turn.metrics };
    },
    finish(id) {
      const turn = turns.get(id);
      if (turn?.started && !turn.completed) {
        turn.completed = true;
        completed.push(id);
      }
    },
    history() {
      return completed.map((id, index) => ({
        turn: index + 1, turnId: id, metrics: { ...turns.get(id).metrics },
      }));
    },
  };
}
