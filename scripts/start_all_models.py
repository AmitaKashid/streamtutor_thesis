"""Start the existing StreamTutor stack and connect both thinking models.

Run on Windows: .venv/Scripts/python.exe scripts/start_all_models.py
Use --reuse-stack to leave the existing local stack and legacy jobs alone.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
KEY = Path.home() / '.ssh' / 'llmtutor_ed25519'
USER = 'ovyj81af'
LOGIN = 'sofja.urz.uni-magdeburg.de'

# Slurm jobs and local ports are listed here for the final status report.
SERVICES = (
    ('Llama 3 8B', 'llmtutor_llama', 8000),
    ('Gemma 4 26B', 'llmtutor_gemma', 8002),
    ('Qwen 3.5 9B', 'llmtutor_qwen_llm', 8003),
    ('FARE judge', 'llmtutor_judge', 8004),
    ('Gemma 4 E4B Thinking', 'llmtutor_e4b', 8005),
    ('Bonsai 2 9B Thinking', 'llmtutor_bonsai2', 8006),
)


def ssh(command: str) -> str:
    result = subprocess.run(['ssh', '-i', str(KEY), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
                             f'{USER}@{LOGIN}', command], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def jobs(name: str):
    command = f"squeue -h -u {shlex.quote(USER)} -n {shlex.quote(name)} -o '%i|%T|%N'"
    rows = [row.split('|', 2) for row in ssh(command).splitlines() if row.strip()]
    if any(len(row) != 3 or not row[0].isdigit() for row in rows):
        raise RuntimeError(f'Unexpected Slurm output for {name}: {rows!r}')
    if len(rows) > 1:
        raise RuntimeError(f'Multiple {name} jobs exist; resolve them before starting: {rows!r}')
    return rows


def ensure_job(name, model, port, remote_script):
    current = jobs(name)
    if current:
        print(f'Reusing {name} job {current[0][0]} ({current[0][1]})', flush=True)
        return current[0][0]
    command = (f'mkdir -p ~/llmtutor/logs && cd ~/llmtutor && '
               f'sbatch --parsable --job-name={shlex.quote(name)} '
               f'--output=logs/{shlex.quote(name)}_%j.log '
               f'{shlex.quote(remote_script)} {shlex.quote(model)} {int(port)}')
    jobid = ssh(command).split(';', 1)[0].strip()
    if not jobid.isdigit(): raise RuntimeError(f'Invalid sbatch response: {jobid!r}')
    print(f'Submitted {name}: {jobid}', flush=True)
    return jobid


def wait_node(name, jobid, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = jobs(name)
        if not current or current[0][0] != jobid:
            raise RuntimeError(f'{name} job {jobid} disappeared. Check ~/llmtutor/logs/{name}_{jobid}.log')
        _, state, node = current[0]
        if state == 'RUNNING' and node and all(c.isalnum() or c in '.-' for c in node):
            return node
        print(f'Waiting for {name} job {jobid}: {state} {node}', flush=True)
        time.sleep(15)
    raise TimeoutError(f'{name} job {jobid} did not start in {timeout}s')


def healthy(port):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3) as response:
            return json.load(response).get('status') == 'ok'
    except (urllib.error.URLError, OSError, ValueError):
        return False


# Exact deployed language-model files, not the UI labels or generic /health.
THINKING_MODEL_FILES = {
    'gemma4_e4b_thinking': 'gemma-4-E4B_q4_0-it.gguf',
    'bonsai2_9b_thinking': 'Ternary-Bonsai-2-9B-Q2_0.gguf',
}


def verify_model_identity(model_key, payload):
    """Check llama-server's reported model ID against the selected GGUF."""
    expected = THINKING_MODEL_FILES[model_key]
    if not isinstance(payload, dict) or not isinstance(payload.get('data'), list):
        raise RuntimeError(f'{model_key}: invalid /v1/models response')
    ids = [entry.get('id') for entry in payload['data'] if isinstance(entry, dict)]
    # The direct llama-server normally reports the path passed with -m.
    if not any(isinstance(value, str) and value.replace('\\', '/').rsplit('/', 1)[-1] == expected for value in ids):
        raise RuntimeError(f'{model_key}: expected {expected}, server reports {ids!r}')
    return expected


def remote_models(node, port):
    # Node is validated by wait_node and port is validated in main. The request
    # runs on the login node because compute-node names are not exposed locally.
    url = f'http://{node}:{int(port)}/v1/models'
    raw = ssh('curl --noproxy "*" -fsS --connect-timeout 3 --max-time 5 ' + shlex.quote(url))
    try:
        return json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f'{node}:{port}: invalid /v1/models JSON') from exc


def local_models(port):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=5) as response:
            return json.load(response)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RuntimeError(f'localhost:{port}: could not read /v1/models: {exc}') from exc


def wait_remote_model(name, jobid, node, port, model_key, timeout=300):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        active = jobs(name)
        if not active or active[0][0] != jobid or active[0][1] != 'RUNNING' or active[0][2] != node:
            raise RuntimeError(f'{name} job {jobid} is no longer running on {node}')
        try:
            payload = remote_models(node, port)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            last_error = exc
            time.sleep(5)
            continue
        verify_model_identity(model_key, payload)
        print(f'{name}: verified {model_key} at {node}:{port}', flush=True)
        return
    raise TimeoutError(f'{name}: no model server at {node}:{port}; check the job log and remote port. Last error: {last_error}')


def print_service_summary():
    """Report observed job status and endpoint health, without inferring identity."""
    print('\n=== StreamTutor model and judge status ===', flush=True)
    for label, name, port in SERVICES:
        try:
            rows = jobs(name)
            job_info = f'{rows[0][0]} {rows[0][1]} node={rows[0][2]}' if rows else 'no active job'
        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
            job_info = f'job status unavailable ({exc})'
        endpoint = 'healthy' if healthy(port) else 'unreachable'
        print(f'{label}: {job_info}; localhost:{port} {endpoint}', flush=True)
    print('Health confirms endpoint response only; it does not verify model identity.', flush=True)


def port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(('127.0.0.1', port))
            return True
        except OSError:
            return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-stack', action='store_true', help='Do not run the existing full-stack startup script')
    parser.add_argument('--gemma-remote-port', type=int, default=8005)
    parser.add_argument('--bonsai-remote-port', type=int, default=18086)
    parser.add_argument('--wait-seconds', type=int, default=1800)
    args = parser.parse_args()
    if os.name != 'nt': raise RuntimeError('Run this launcher on Windows.')
    if not KEY.is_file(): raise RuntimeError(f'SSH key missing: {KEY}')
    if not all(1024 <= p <= 65535 for p in (args.gemma_remote_port, args.bonsai_remote_port)):
        raise ValueError('Remote ports must be between 1024 and 65535')
    if args.wait_seconds <= 0: raise ValueError('--wait-seconds must be positive')
    legacy = ROOT / 'scripts' / 'start_full_llmtutor_stack.ps1'
    job_script = ROOT / 'scripts' / 'run_thinking_model.sbatch'
    if not legacy.is_file() or not job_script.is_file(): raise RuntimeError('Startup script or thinking-model Slurm script is missing')
    print('=== StreamTutor full startup: five tutor models + one judge ===', flush=True)
    print('Phase 1 starts Llama 3, Gemma 4 26B, Qwen 3.5 and the FARE judge.', flush=True)
    print('Phase 2 starts or reuses Gemma 4 E4B Thinking and Bonsai 2 9B Thinking.', flush=True)
    print('The older script reports only Phase 1; Phase 2 is shown separately below.', flush=True)
    if not args.reuse_stack:
        print('Starting your existing full-stack script. Its normal cleanup and job submission will run.', flush=True)
        subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(legacy)], cwd=str(ROOT), check=True)
    print('\n=== Phase 2: thinking-model jobs and tunnels ===', flush=True)
    # Version the upload by its hash to avoid overwriting an existing remote script.
    import hashlib
    digest = hashlib.sha256(job_script.read_bytes()).hexdigest()[:12]
    remote_script = f'scripts/run_thinking_model_{digest}.sbatch'
    ssh('mkdir -p ~/llmtutor/scripts ~/llmtutor/logs')
    subprocess.run(['scp', '-i', str(KEY), '-o', 'BatchMode=yes', str(job_script),
                    f'{USER}@{LOGIN}:llmtutor/{remote_script}'], check=True)
    models = [
        ('llmtutor_e4b', 'gemma4_e4b_thinking', 8005, args.gemma_remote_port),
        ('llmtutor_bonsai2', 'bonsai2_9b_thinking', 8006, args.bonsai_remote_port),
    ]
    pending = [(name, key, local, remote, ensure_job(name, key, remote, remote_script)) for name, key, local, remote in models]
    forwards = []
    for name, key, local, remote, jobid in pending:
        node = wait_node(name, jobid, args.wait_seconds)
        # Validate the server on the assigned node and configured remote port,
        # even for a reused Slurm job. A running job alone is not readiness.
        wait_remote_model(name, jobid, node, remote, key)
        if healthy(local):
            verify_model_identity(key, local_models(local))
            print(f'{name}: localhost:{local} reports the expected model; reusing endpoint.', flush=True)
            print('  The endpoint check does not prove which job owns the existing tunnel.', flush=True)
        elif not port_free(local):
            raise RuntimeError(f'Local port {local} is occupied but not healthy. Close its existing tunnel and retry.')
        else:
            forwards += ['-L', f'127.0.0.1:{local}:{node}:{remote}']
    if forwards:
        command = ['ssh', '-i', str(KEY), '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes',
                   '-o', 'ServerAliveInterval=30', '-N', *forwards, f'{USER}@{LOGIN}']
        tunnel = subprocess.Popen(command, creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        tunnel = None
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if all(healthy(item[2]) for item in pending):
            for name, key, local, remote, jobid in pending:
                verify_model_identity(key, local_models(local))
                print(f'{name}: localhost:{local} reports the expected model.', flush=True)
            print_service_summary()
            print('Thinking model endpoints are healthy. Open http://localhost:3000', flush=True)
            return
        if tunnel is not None and tunnel.poll() is not None:
            raise RuntimeError('SSH tunnel exited; check the tunnel window for errors')
        time.sleep(5)
    raise TimeoutError('Thinking model endpoints did not become healthy. Check their Slurm logs and remote ports.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, ValueError, TimeoutError, subprocess.SubprocessError) as exc:
        print(f'Startup failed: {exc}', file=sys.stderr)
        sys.exit(1)
