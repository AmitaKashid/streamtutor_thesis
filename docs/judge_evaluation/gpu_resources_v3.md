# A30 GPU resource evidence

Run the collector on the **judge compute node, inside the judge's Slurm GPU
allocation**, not from the login node or Windows. Confirm the assigned
physical GPU UUID from the allocation and server device mapping. Do not
assume that CUDA device 0 equals physical GPU index 0. The collector requires
the UUID explicitly; it does not verify Slurm-to-physical-device mapping.

Use the same compute-node clock for the latency runner and sampler. With the
server loaded but idle, first record a short idle snapshot. Then start a fresh
benchmark capture, wait for its first sample, run the latency benchmark, and
stop the capture with Ctrl+C **after** the runner finishes. Keep the server
loaded until the capture ends. Use fresh output filenames for each run.

```bash
python -m judge.sample_gpu_resources_v3 collect --gpu-uuid "$JUDGE_GPU_UUID" --phase loaded-idle --samples 5 --output "$EVIDENCE_DIR/loaded_idle.jsonl"
python -m judge.sample_gpu_resources_v3 collect --gpu-uuid "$JUDGE_GPU_UUID" --phase benchmark --output "$EVIDENCE_DIR/gpu_samples.jsonl"
```

After the benchmark and capture finish:

```bash
python -m judge.sample_gpu_resources_v3 summarize --run-dir "$LATENCY_RUN_DIR" --samples "$EVIDENCE_DIR/gpu_samples.jsonl" --loaded-idle "$EVIDENCE_DIR/loaded_idle.jsonl" --server-log "$JUDGE_SERVER_LOG"
```

The summary records the median loaded-idle memory, sampled workload peak
used memory, minimum free memory, and median and maximum GPU utilization.
It requires samples bracketing the measured window, at least one sample
inside it, and no gap greater than 2 seconds across the bracket. These
checks are sampling quality rules, not new model-selection thresholds.
The protocol's 2048 MiB threshold is reported against the observed samples.
Board-level polling can miss short-lived spikes and cannot attribute usage
to the judge process alone; use a dedicated GPU.

Keep the original server startup log from the same allocation. A full layer
count is evidence of layer placement, but not proof of no unintended CPU
tensor offload. CPU-mapped and pinned-host buffers require review. Missing
or ambiguous evidence stays unverified. This tool does not pass or fail the
overall deployment gate.
