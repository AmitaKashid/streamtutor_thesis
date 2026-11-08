# Judge latency runner v3

This executes the comparative workload from
[the v3 execution protocol](judge_execution_protocol_v3.md). It does not update
prior compatibility outcomes or score judge quality.

Offline plan (no judge server or GPU required):

```text
python -m judge.run_latency_benchmark_v3 --model-id prometheus_7b_v2_q4 --run-id prometheus-latency-plan-v3 --dry-run
```

For a measured run, use a new run ID, omit `--dry-run`, and supply
`--base-url` and `--slurm-job-id` for the actual A30 job. Start the server
with `--n-ctx 4096 --max-new-tokens 256 --temperature 0 --top-p 1 --disable-system-prompt`.
The runner does not configure or attest the remote server. Keep the launch
log, model hash, backend version, GPU samples and Slurm allocation alongside
the results. Do not use an A40 run for the A30 deployment claim.

Each run stores the ordered task plan and checksum before sending any requests.
Five disjoint calibration tasks warm up the server (one request each). The
runner then executes the same 30 scalar prompts three times, sequentially.
Any format retries follow the shared scalar evaluator. Raw outputs, retries,
and transport errors remain in `evaluations.jsonl`; `attempts.csv` is the
tabular view. `summary.json` excludes warm-ups from aggregates.

Client evaluation time includes prompt assembly and any retry but excludes
disk writes. Attempt wall time measures the provider call. Server completion
time includes prefill and decoding. The non-streaming endpoint does not
expose TTFT or decode-only throughput; these remain null. A completed run
means the workload finished, not that it passed a deployment gate.

Use a fresh run ID after an interruption. Partial evidence is retained, but
mixing timings from different sessions or Slurm allocations into one benchmark
is not supported.
