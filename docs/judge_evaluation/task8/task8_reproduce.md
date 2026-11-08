# Task 8 reproducibility checkpoint — Task 7 evidence

## Frozen artifact

- Git branch: `judge-selection-sprint`
- Source revision recorded by operator: `82e5a9898cf1826803594731cc842aba73973738`
- Local archive: `task7_evidence_20261108.zip`
- SHA-256: `47426b73fad70fdc6635159d2e89d33b290219a8a72d975acf74581f8fc0b20f`

The archive includes 8 accepted Task 7 block manifests, results, events, one SQLite judge-spool snapshot, reference provenance and earlier failed infrastructure attempts. Do not treat every record in the SQLite database or failed-attempt folder as a valid Task 7 trial.

## Reproduce from a Windows CMD terminal

Copy `task7_reproduce.py` to the repository root alongside `task7_evidence_20261108.zip`. Run:

```cmd
.\.venv\Scripts\python.exe task7_reproduce.py task7_evidence_20261108.zip --out task7_reproduced
```

The script uses only Python's standard library. It checks the archive hash, verifies exactly 11 distinct trial IDs in each of 8 scheduled blocks, enforces matching event/turn/session IDs, checks ordering of first audio, audio generation and playback acknowledgment, verifies the 44 ON turns each received seven valid dimension scores, and excludes older SQLite entries by turn ID. It writes:

- `task7_reproduced/task7_per_trial_analysis.csv`
- `task7_reproduced/task7_judge_evaluations.csv`
- `task7_reproduced/task7_summary.json`

Expected first-audio medians: OFF approximately 0.603 s; ON approximately 0.600 s. Expected mean question-paired ON−OFF difference: approximately +0.00037 s. Browser-completion medians: OFF approximately 78.317 s; ON approximately 70.570 s. The judge's seven scores became available a median approximately 5.95 s after the browser playback acknowledgement.

The script also computes a **question-cluster percentile bootstrap** (20,000 resamples, seed 20261108). Its confidence interval need not match a previous report using a different bootstrap implementation or random seed.

## Interpretation boundaries

Passing the 88-trial end-to-end typed-input pilot supports operational reliability **within the sampled workload**, not deployment certification, zero runtime overhead, STT quality, human-rated judge accuracy, or A30 suitability. The longer playback-completion times depend strongly on generated speech duration. The ON/OFF comparisons are observational within the frozen block schedule; GPU co-location and block order can confound estimates.

## Follow-on gates for Task 8

- Reconcile the existing judge candidate selection report and screening evidence without replacing their conclusions.
- Verify a clean checkout, including any non-tracked runtime datasets, hardware-specific backends and external inference servers.
- Write the final deployment gate (A40 observed, A30 not independently tested) and human-validation protocol.
- Commit the source script, manifest and instructions; archive the raw ZIP externally with its hash. Avoid committing the SQLite spool without review.
