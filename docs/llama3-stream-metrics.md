# Llama 3 streamed model metrics

The Llama 3 tutor uses the Python remote server, not the `llama-server`
endpoint used by the newer thinking models. Its streaming response does not
include the normal non-streaming `usage` payload in the reviewed versions.
The remote server serializes access to the model and temporarily observes the
existing token generator for `llama-cpp-python` 0.3.4 and 0.3.20 only. Other
versions leave the corresponding fields unavailable until reviewed.

For the Python Llama 3 path:

- Prompt tokens count the fully formatted tokens passed to the generator;
  processed prompt tokens are the first evaluation, and cached tokens are the
  difference. This follows the prefix-cache logic in the reviewed versions.
- Prefill is elapsed wall time for the first evaluation plus GPU
  synchronization. This adds one synchronization per completed or interrupted
  inference. It excludes model-request queueing and network transit.
- Output counts non-end-of-generation sampled tokens, including tokens hidden
  by a stop sequence. Decode time sums model generator steps after the first
  sampled token and excludes end-of-generation steps and consumer/network
  waits. The decode rate uses the count of those timed steps. A one-token
  response has no decode rate. These timing boundaries are not necessarily
  identical to the native timings reported by other model servers.
- Counters arrive at a confirmed `stop` terminal event. Interrupted, failed or
  unsupported requests may have missing metrics. Thinking time does not apply
  to Llama 3 unless the server actually emits reasoning timing.

Sources: [llama-cpp-python 0.3.4 implementation](https://github.com/abetlen/llama-cpp-python/blob/v0.3.4/llama_cpp/llama.py),
[llama-cpp-python 0.3.20 implementation](https://github.com/abetlen/llama-cpp-python/blob/v0.3.20/llama_cpp/llama.py), and
[llama.cpp server API](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

Run `scripts/deploy_llama3_stream_metrics.ps1` from the Windows checkout to
copy the patched provider and server to Sofja. It stages both files and stores
the originals in a uniquely named `~/llmtutor/backups/llama3-metrics-*`
directory. The existing launcher does not upload these files by itself. Start
normally with `scripts/start_all_models.py` afterward so that the Slurm job
loads the updated code. Do not use `--reuse-stack` for that first restart.

Check the tunnel using `Invoke-RestMethod http://127.0.0.1:8000/health`.
`stream_metrics_protocol` should be `llamacpp-observed-v1` and
`llama3_metrics_version_supported` should be `true`. If either is missing or
false, keep the actual version and server response for troubleshooting; do not
install or replace GPU packages without checking the cluster environment.
