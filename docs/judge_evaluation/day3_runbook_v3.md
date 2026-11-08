# Day 3: judge quality, robustness, and controlled contrasts

This runbook executes the frozen v3 protocol. Day 3 uses comparable
A40 48 GB allocations for quality evaluation. These runs do not certify
Day 2's A30 deployment gate.

## 1. Choose one candidate and verify its server

| Candidate | Model key |
|---|---|
| GLM-4-32B | `glm4_32b_0414_q4` |
| FARE-20B | `fare_20b_q4` |
| CompassJudger-2-7B | `compassjudger_2_7b_q4` |
| Prometheus-7B v2 | `prometheus_7b_v2_q4` |

Use a clean, committed checkout containing the v3 dataset and runners. Inside
an **existing, authorized A40 allocation**, use the project environment and
set `MODEL_ID` to the candidate key. Do not start the server on a login node.
For example, when the client runs on the same compute node:

```bash
MODEL_ID=prometheus_7b_v2_q4
python app/remote_llm_server.py --model "$MODEL_ID" --host 127.0.0.1 --port 18204 --n-ctx 4096 --max-new-tokens 256 --temperature 0 --top-p 1 --disable-system-prompt
```

For a client on another machine, use the established SSH forward to the
allocated server and set the client URL to the local forwarded port. Use
one loaded candidate at a time and do not expose a benchmark server publicly.
Capture the actual launch arguments, loaded artifact and SHA-256, backend,
GPU UUID, Slurm job ID, and server log. Verify the candidate and frozen
settings from this evidence: a successful health endpoint alone does not
identify the model behind the port. The runner generation flags record
expected settings; they do not reconfigure a running server.

## 2. Run calibration first

In Windows CMD from the thesis repo, set these variables for the candidate
whose server you verified. If running from Sofja instead, use the project
environment's `python` and translate the CMD variable syntax to your shell.

```bat
set "DAY3_MODEL=prometheus_7b_v2_q4"
set "DAY3_URL=http://127.0.0.1:18204"
set "DAY3_SERIES=day3-v3-001"
```

Repeat calibration for all four candidates by changing `DAY3_MODEL` and
`DAY3_URL` as needed.

```bat
.\.venv\Scripts\python.exe -m judge.run_judge_benchmark_v3 --split calibration --base-url "%DAY3_URL%" --model-id "%DAY3_MODEL%" --quantization Q4_K_M --hardware "NVIDIA A40 48 GB" --n-ctx 4096 --max-new-tokens 256 --temperature 0 --top-p 1 --seed 0 --run-id "%DAY3_SERIES%-%DAY3_MODEL%-calibration"
```

Expect **132** planned evaluations per candidate. Review completion,
validation errors, retries, and score coverage. Calibration is for pipeline
verification, not candidate-specific prompt or rubric tuning. Do not start
selection until every candidate has completed comparable calibration.

## 3. Run selection, robustness, and contrasts

```bat
.\.venv\Scripts\python.exe -m judge.run_judge_benchmark_v3 --split selection --base-url "%DAY3_URL%" --model-id "%DAY3_MODEL%" --quantization Q4_K_M --hardware "NVIDIA A40 48 GB" --n-ctx 4096 --max-new-tokens 256 --temperature 0 --top-p 1 --seed 0 --run-id "%DAY3_SERIES%-%DAY3_MODEL%-selection"
.\.venv\Scripts\python.exe -m judge.run_judge_benchmark_v3 --split robustness --base-url "%DAY3_URL%" --model-id "%DAY3_MODEL%" --quantization Q4_K_M --hardware "NVIDIA A40 48 GB" --n-ctx 4096 --max-new-tokens 256 --temperature 0 --top-p 1 --seed 0 --run-id "%DAY3_SERIES%-%DAY3_MODEL%-robustness"
.\.venv\Scripts\python.exe -m judge.run_controlled_contrasts_v3 --base-url "%DAY3_URL%" --model-id "%DAY3_MODEL%" --quantization Q4_K_M --hardware "NVIDIA A40 48 GB" --n-ctx 4096 --max-new-tokens 256 --temperature 0 --top-p 1 --seed 0 --run-id "%DAY3_SERIES%-%DAY3_MODEL%-contrasts"
```

| Per candidate | Planned evaluations |
|---|---:|
| Calibration | 132 |
| Selection | 792 |
| Robustness | 123 |
| Controlled contrasts | 28 (14 contrasts in both orientations) |

Each command is independent: check its exit code and the run's manifest
before moving on. Do not shorten selection with `--max-responses`.

## 4. Analyze and archive

Set `DAY3_SPLIT` to `calibration`, `selection`, or `robustness` for the
completed scalar run you want to analyze:

```bat
set "DAY3_SPLIT=selection"
set "DAY3_RUN=judge_outputs\benchmark_v3\%DAY3_SERIES%-%DAY3_MODEL%-%DAY3_SPLIT%"
.\.venv\Scripts\python.exe -m judge.analyze_judge_benchmark_v3 --run-dir "%DAY3_RUN%" --reference-source provisional_ai --acknowledge-provisional-not-human
```

The provisional reference is AI-derived, not human ground truth. For a
separate human-reference analysis, use `--reference-source human`,
`--human-labels PATH`, `--rater-id ID`, and
`--human-reference-status individual` or `adjudicated` as appropriate.
Use a distinct `--output-dir` so analyses do not overwrite one another.
For contrasts, read the separate run's `summary.json`: report valid paired
coverage and position consistency before provisional preference accuracy.
The scalar analyzer does not analyze contrast results.

Archive each stopped run with the evidence archiver, specifying `--stage
scalar` or `--stage pairwise`, and attach its same-run allocation,
configuration, and server log. If the run manifest lacks a Slurm job ID,
supply `--slurm-job-id` from the allocation record.

## 5. Resume or rehearse safely

- To resume an interrupted live run, use its **original command and exact
  run ID**, adding only `--resume`. Keep the same Git commit, model artifact,
  settings, endpoint, dataset, and output location. Confirm the actual server
  again before resuming; do not run two writers against one directory.
- If any of those change, or you need to rerun already recorded errors, use
  a new run ID and preserve the earlier results. Resume skips existing records;
  it does not replace them.
- To rehearse a command, use a **distinct** run ID ending in `-plan` and add
  `--dry-run`. These runners do not contact the endpoint in dry-run mode, but
  still require a URL argument. Inspect `plan.json`; do not resume a dry run
  as a live run or feed its directory to the analyzer.
