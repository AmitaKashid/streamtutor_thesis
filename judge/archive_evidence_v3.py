"""Package a stopped judge experiment into a checksummed evidence ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

STAGES = {
    "scalar": ("predictions.jsonl", "summary.json", "summary_by_dimension.csv"),
    "pairwise": ("pairwise_predictions.jsonl", "summary.json"),
    "latency": ("plan.json", "evaluations.jsonl", "attempts.csv", "summary.json"),
    "interference": ("plan.json", "turns.jsonl"),
}
ATTACHMENTS = {
    "server-log": "server.log",
    "slurm-job": "slurm-job.txt",
    "configuration": "configuration.txt",
    "gpu-samples": "gpu-samples.jsonl",
    "loaded-idle": "loaded-idle.jsonl",
    "runtime-judge": "runtime-judge.jsonl",
    "analysis-report": "analysis-report.json",
    "agreement-by-dimension": "agreement-by-dimension.csv",
    "scored-predictions": "scored-predictions.csv",
}
DRY_STATUSES = {"dry_run", "dry_run_complete"}
INCOMPLETE_STATUSES = {"interrupted", "incomplete", "runner_error", "failed_healthcheck"}
INDEX = "evidence.json"
CHECKSUMS = "checksums.sha256"
CHUNK = 1024 * 1024


def _json_bytes(data):
    return (json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def _sha256(source, destination=None):
    digest = hashlib.sha256()
    while chunk := source.read(CHUNK):
        digest.update(chunk)
        if destination is not None:
            destination.write(chunk)
    return digest.hexdigest()


def _require_file(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular, non-symlink file: {path}")


def _copy_member(archive, name, path):
    _require_file(path)
    before = path.stat()
    with path.open("rb") as source, archive.open(name, "w", force_zip64=True) as destination:
        digest = _sha256(source, destination)
    after = path.stat()
    if not os.path.samestat(before, after) or (before.st_size, before.st_mtime_ns) != (
        after.st_size, after.st_mtime_ns,
    ):
        raise ValueError(f"Source changed during archive: {path}")
    return digest


def _required_run_names(stage, status):
    if status == "completed":
        return {"manifest.json", *STAGES[stage]}
    if status in DRY_STATUSES:
        return {"manifest.json", "plan.json"}
    return {"manifest.json"}


def _run_files(run_dir, stage, status):
    names = {"manifest.json", "plan.json", *STAGES[stage]}
    if stage == "latency":
        names.add("gpu_summary.json")
    for name in sorted(_required_run_names(stage, status)):
        _require_file(run_dir / name)
    return [
        (f"run/{name}", run_dir / name)
        for name in sorted(names)
        if (run_dir / name).exists() or (run_dir / name).is_symlink()
    ]


def create_archive(run_dir, stage, output, attachments=(), *, slurm_job_id=None,
                   allow_incomplete=False):
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError("Expected a regular run directory")
    if output.resolve().is_relative_to(run_dir.resolve()):
        raise ValueError("Place the archive outside the run directory")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output}")
    manifest_path = run_dir / "manifest.json"
    _require_file(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("run_id"), str) or not manifest["run_id"].strip():
        raise ValueError("Manifest is missing its run ID")
    status = manifest.get("status")
    if not isinstance(status, str):
        raise ValueError("Manifest is missing its status")
    if status == "running":
        raise ValueError("Stop the run before archiving; the manifest is still marked running")
    if status in INCOMPLETE_STATUSES:
        if not allow_incomplete:
            raise ValueError("Use --allow-incomplete to preserve this stopped, incomplete run")
    elif status not in DRY_STATUSES | {"completed"}:
        raise ValueError(f"Unrecognized run status: {status!r}")
    manifest_job = manifest.get("slurm_job_id")
    if slurm_job_id is not None and manifest_job is not None and str(manifest_job) != slurm_job_id:
        raise ValueError("Supplied Slurm job ID conflicts with the run manifest")

    files = _run_files(run_dir, stage, status)
    roles = set()
    for role, path in attachments:
        if role not in ATTACHMENTS or role in roles:
            raise ValueError(f"Unknown or repeated attachment role: {role}")
        roles.add(role)
        _require_file(path)
        files.append((f"attachments/{ATTACHMENTS[role]}", path))

    job = str(manifest_job) if manifest_job is not None else slurm_job_id
    gaps = []
    if not manifest.get("git_commit"):
        gaps.append("run_git_commit")
    if status not in DRY_STATUSES:
        if not job:
            gaps.append("slurm_job_id")
        for role in ("configuration", "server-log", "slurm-job"):
            if role not in roles:
                gaps.append(role)
        if stage in {"latency", "interference"} and "gpu-samples" not in roles:
            gaps.append("gpu-samples")
    evidence = {
        "archive_format": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "run_id": manifest["run_id"],
        "run_status": status,
        "run_git_commit": manifest.get("git_commit"),
        "slurm_job_id": job,
        "slurm_job_id_source": "run_manifest" if manifest_job is not None else (
            "operator" if slurm_job_id is not None else None
        ),
        "missing_evidence": gaps,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage the ZIP before publishing. Exclusive create also handles a race at output.
    with tempfile.TemporaryFile(dir=output.parent) as staged:
        with ZipFile(staged, "w", compression=ZIP_DEFLATED) as archive:
            hashes = {
                name: _copy_member(archive, name, path)
                for name, path in sorted(files)
            }
            if hashes["run/manifest.json"] != hashlib.sha256(manifest_bytes).hexdigest():
                raise ValueError("Run manifest changed during archiving")
            index = _json_bytes(evidence)
            archive.writestr(INDEX, index)
            hashes[INDEX] = hashlib.sha256(index).hexdigest()
            archive.writestr(CHECKSUMS, "".join(
                f"{digest}  {name}\n" for name, digest in sorted(hashes.items())
            ))
        staged.seek(0)
        with output.open("xb") as destination:
            shutil.copyfileobj(staged, destination)
    return evidence


def verify_archive(path, expected_sha256=None):
    _require_file(path)
    with path.open("rb") as stream:
        archive_hash = _sha256(stream)
        if expected_sha256 is not None and archive_hash != expected_sha256.lower():
            raise ValueError("ZIP SHA-256 does not match the supplied digest")
        stream.seek(0)
        with ZipFile(stream) as archive:
            names = archive.namelist()
            allowed = {
                INDEX, CHECKSUMS, "run/manifest.json", "run/plan.json",
                "run/gpu_summary.json",
                *(f"run/{name}" for files in STAGES.values() for name in files),
                *(f"attachments/{name}" for name in ATTACHMENTS.values()),
            }
            if len(names) != len(set(names)) or set(names) - allowed:
                raise ValueError("ZIP contains unexpected or duplicate member names")
            if not {INDEX, CHECKSUMS, "run/manifest.json"} <= set(names):
                raise ValueError("ZIP is missing required metadata")
            expected = {}
            for line in archive.read(CHECKSUMS).decode("utf-8").splitlines():
                match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
                if not match or match[2] in expected:
                    raise ValueError("Invalid or duplicate checksum entry")
                expected[match[2]] = match[1]
            if set(expected) != set(names) - {CHECKSUMS}:
                raise ValueError("ZIP members do not match the checksum inventory")
            for name, digest in expected.items():
                with archive.open(name) as member:
                    if _sha256(member) != digest:
                        raise ValueError(f"Checksum mismatch: {name}")
            evidence = json.loads(archive.read(INDEX))
            manifest = json.loads(archive.read("run/manifest.json"))
            if (
                not isinstance(evidence, dict)
                or not isinstance(manifest, dict)
                or evidence.get("archive_format") != 1
                or evidence.get("stage") not in STAGES
                or evidence.get("run_id") != manifest.get("run_id")
                or evidence.get("run_status") != manifest.get("status")
                or evidence.get("run_git_commit") != manifest.get("git_commit")
            ):
                raise ValueError("Evidence index does not match the run manifest")
            if not {
                f"run/{name}" for name in _required_run_names(evidence["stage"], evidence["run_status"])
            } <= set(names):
                raise ValueError("Archive is missing a required run artifact")
    return {"verified": True, "archive_sha256": archive_hash, **evidence}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--run-dir", type=Path, required=True)
    create.add_argument("--stage", choices=STAGES, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--attach", nargs=2, action="append", default=[], metavar=("ROLE", "FILE"))
    create.add_argument("--slurm-job-id")
    create.add_argument("--allow-incomplete", action="store_true")
    verify = sub.add_parser("verify")
    verify.add_argument("archive", type=Path)
    verify.add_argument("--sha256")
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            create_archive(
                args.run_dir, args.stage, args.output,
                [(role, Path(path)) for role, path in args.attach],
                slurm_job_id=args.slurm_job_id, allow_incomplete=args.allow_incomplete,
            )
            result = verify_archive(args.output)
        else:
            result = verify_archive(args.archive, args.sha256)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, BadZipFile) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
