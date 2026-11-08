# Judge evidence archive

Archive a stopped run using its stage: `scalar`, `pairwise`, `latency`, or
`interference`. Use a fresh output path outside the run directory.
Do not resume the run or let any sampler or logger write the inputs while
the archive is being made.

Example (replace the paths with files from the same run):

```text
python -m judge.archive_evidence_v3 create --stage latency --run-dir judge_outputs/latency_v3/a30-001 --output judge_outputs/evidence/a30-001.zip --attach configuration configuration.txt --attach slurm-job slurm-job.txt --attach server-log server.log --attach gpu-samples gpu-samples.jsonl
python -m judge.archive_evidence_v3 verify judge_outputs/evidence/a30-001.zip
```

The run's original manifest supplies the recorded Git commit and, when
present, the Slurm job ID. For a runner that doesn't record the job ID,
supply `--slurm-job-id` from the allocation record; the archive identifies
this as operator-supplied. A matching job ID does not establish that a
separately supplied log belongs to that job. Check that before attachment.

For `configuration.txt`, record the run settings that were actually used:
model/artifact revision and digest, backend, launch arguments, GPU UUIDs,
and any runtime configuration not already in the manifest. Do not dump
environment variables or attach tokens, keys, or private data. Additional
attachment roles are `loaded-idle`, `runtime-judge`, `analysis-report`,
`agreement-by-dimension`, and `scored-predictions`.

The archive includes only known run artifacts and explicitly named
attachments. It does not include model weights or copy a whole directory.
Every archived file except the checksum inventory itself is covered by
that inventory. Save the reported ZIP SHA-256 separately; passing it as
`--sha256` during verification detects replacement of the entire ZIP.

Dry runs remain dry runs. Use `--allow-incomplete` to preserve a stopped
failed or interrupted run. A passing integrity check does not establish
that the experiment was valid, complete, or passed a deployment gate.
