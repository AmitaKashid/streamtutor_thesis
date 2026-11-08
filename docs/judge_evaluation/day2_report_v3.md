# Day 2 A30 deployment report

Generate an empty evidence report offline:

```text
python -m judge.report_day2_v3 --output-dir judge_outputs/day2-no-evidence
```

For a real report, supply one verified latency evidence archive per
candidate, using a fresh output directory. Include the same-run
`configuration`, `slurm-job`, `server-log`, and `gpu-samples` attachments.

```text
python -m judge.report_day2_v3 --archive run-a.zip --archive run-b.zip --reviews day2_reviews.json --output-dir judge_outputs/day2-reviewed
```

The review file is optional. Without it the report cannot certify the
actual hardware and configuration. Key each review by the ZIP SHA-256
reported by `judge.archive_evidence_v3 verify`. Only include checks you
can substantiate; the strings below are placeholders, not evidence.

```json
{
  "REPLACE_WITH_VERIFIED_ARCHIVE_SHA256": {
    "frozen_a30_configuration": {"value": true, "evidence": "Verified Slurm allocation, artifact, and actual launch configuration"},
    "no_unintended_cpu_tensor_offload": {"value": true, "evidence": "Same-run tensor placement trace"},
    "no_unrecovered_crash_or_hang": {"value": true, "evidence": "Same-run runtime log and exit status"},
    "median_decode_throughput": {"value": 10.0, "evidence": "Source and calculation for decode-only timings"}
  }
}
```

The report distinguishes a verified failed threshold from an unverified
gate. It checks the archive, frozen plan and order, then uses raw evaluation
timing and GPU samples. P95 completion latency means end-to-end evaluation
wall time including any format retry. GPU headroom is the minimum observed
in the sampled window, not a bound on unobserved transients. A model-server
TPS value that includes prefill is not a substitute for decode-only TPS.

The report writes JSON with individual gate outcomes and a Markdown summary
for review. It does not rank judge quality, certify tutor non-interference,
or substitute for the protocol's separate compatibility evidence.
