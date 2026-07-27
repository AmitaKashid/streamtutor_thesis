$ErrorActionPreference = "Stop"

$ClusterUser = "ovyj81af"
$ClusterLogin = "sofja.urz.uni-magdeburg.de"
$SshKey = "C:/Users/shaik/.ssh/llmtutor_ed25519"

$RemoteProject = "~/llmtutor"
$LocalProject = "C:\Users\shaik\Desktop\llmtutor"

$GemmaModel = "gemma4_26b_q3"
$GemmaPort = 8002

$LlamaModel = "llama3_8b_q4"
$LlamaPort = 8000

function Invoke-ClusterSSH($Command) {
    & ssh.exe `
        -T `
        -n `
        -i $SshKey `
        -o IdentitiesOnly=yes `
        -o BatchMode=yes `
        -o ConnectTimeout=20 `
        -o ServerAliveInterval=10 `
        -o ServerAliveCountMax=3 `
        "$ClusterUser@$ClusterLogin" `
        "$Command"
}
function Run-SSH($Command) {
    Write-Host "Running remote command: $Command"

    $output = Invoke-ClusterSSH "$Command; echo __REMOTE_DONE__"
    $exitCode = $LASTEXITCODE

    if ($output) {
        $output | ForEach-Object { Write-Host $_ }
    }

    Write-Host "Remote command exit code: $exitCode"

    if ($exitCode -ne 0) {
        throw "SSH command failed with exit code ${exitCode}: $Command"
    }

    if ($output -notcontains "__REMOTE_DONE__") {
        Write-Host "Warning: remote completion marker was not seen."
    }
}
function Wait-ForRemoteFile($Path, $Label, $TimeoutSeconds = 300) {
    Write-Host "Waiting for $Label..."
    $start = Get-Date

    while (((Get-Date) - $start).TotalSeconds -lt $TimeoutSeconds) {
        $value = Invoke-ClusterSSH "cat $Path 2>/dev/null"

        if ($value) {
            $line = ($value | Select-Object -First 1).Trim()
            if ($line.Length -gt 0) {
                Write-Host "${Label}: $line"
                return $line
            }
        }

        Start-Sleep -Seconds 5
    }

    throw "Timeout waiting for $Label at $Path"
}

function Wait-ForModelLoaded($JobName, $Label, $TimeoutSeconds = 1200) {
    Write-Host "Waiting for $Label model to finish loading..."
    $start = Get-Date

    while (((Get-Date) - $start).TotalSeconds -lt $TimeoutSeconds) {
        $loaded = Invoke-ClusterSSH "cd $RemoteProject && grep -R '\[SERVER\] Model loaded.' logs/${JobName}_*.out 2>/dev/null | tail -1"

        $uvicornReady = Invoke-ClusterSSH "cd $RemoteProject && grep -R 'Uvicorn running on' logs/${JobName}_*.err logs/${JobName}_*.out 2>/dev/null | tail -1"

        $errorText = Invoke-ClusterSSH "cd $RemoteProject && grep -R -E 'cudaMalloc failed|out of memory|Failed to load model|Traceback|command not found|module: command not found|xpython' logs/${JobName}_*.err 2>/dev/null | tail -5"

        if ($loaded -or $uvicornReady) {
            Write-Host "$Label loaded and server started."
            return
        }

        if ($errorText) {
            Write-Host ""
            Write-Host "$Label failed while loading. Recent error:"
            Write-Host $errorText
            throw "$Label model failed to load. Check logs/${JobName}_*.err"
        }

        Write-Host "$Label is still loading..."
        Start-Sleep -Seconds 15
    }

    throw "Timeout waiting for $Label model to load."
}

function Test-Health($Url, $Label, $TimeoutSeconds = 300) {
    Write-Host "Testing health for $Label at $Url..."
    $start = Get-Date

    while (((Get-Date) - $start).TotalSeconds -lt $TimeoutSeconds) {
        try {
            $response = curl.exe -s --max-time 3 $Url
            if ($LASTEXITCODE -eq 0 -and $response) {
                Write-Host "${Label} health OK: $response"
                return
            }
        } catch {}

        Start-Sleep -Seconds 5
    }

    throw "Health check failed for $Label at $Url"
}

Write-Host "======================================"
Write-Host "Starting two-model LLMTutor stack"
Write-Host "Gemma first, then Llama"
Write-Host "======================================"

Write-Host "`n[0/7] Cleaning old runtime files..."
Run-SSH "cd $RemoteProject && rm -f runtime/*.host runtime/*.port runtime/*.jobid"

Write-Host "`n[1/7] Submitting Gemma job..."
Run-SSH "cd $RemoteProject && sbatch --job-name=llmtutor_gemma scripts/run_remote_model.sbatch $GemmaModel $GemmaPort"

$GemmaHost = Wait-ForRemoteFile "$RemoteProject/runtime/$GemmaModel.host" "Gemma host" 600

Wait-ForModelLoaded "llmtutor_gemma" "Gemma" 1200

Write-Host "`n[2/7] Submitting Llama job..."
Run-SSH "cd $RemoteProject && sbatch --job-name=llmtutor_llama scripts/run_remote_model.sbatch $LlamaModel $LlamaPort"

$LlamaHost = Wait-ForRemoteFile "$RemoteProject/runtime/$LlamaModel.host" "Llama host" 300

Wait-ForModelLoaded "llmtutor_llama" "Llama" 600

Write-Host "`n[3/7] Starting SSH tunnel..."

$TunnelCommand = "ssh -N -v -i $SshKey -o IdentitiesOnly=yes -L 8000:$LlamaHost`:8000 -L 8002:$GemmaHost`:8002 $ClusterUser@$ClusterLogin"

Write-Host "Tunnel command:"
Write-Host $TunnelCommand

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    $TunnelCommand
)

Write-Host ""
Write-Host "A new SSH tunnel window has opened."
Write-Host "It should not ask for password now because the SSH key is configured."
Read-Host "After the tunnel window is open, press Enter here to continue"

Start-Sleep -Seconds 3

Write-Host "`n[4/7] Checking remote model health through tunnel..."
Test-Health "http://127.0.0.1:8002/health" "Gemma" 300
Test-Health "http://127.0.0.1:8000/health" "Llama" 300

Write-Host "`n[5/7] Starting frontend_api..."
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd $LocalProject; .\.venv\Scripts\activate; uvicorn frontend_api:app --reload --port 8001"
)

Start-Sleep -Seconds 5

Write-Host "`n[6/7] Starting frontend..."
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd $LocalProject\frontend; npm run dev"
)

Write-Host "`n[7/7] Done."
Write-Host "Keep the opened PowerShell windows running."
Write-Host ""
Write-Host "Model mapping should be:"
Write-Host "  $LlamaModel  -> http://127.0.0.1:8000"
Write-Host "  $GemmaModel  -> http://127.0.0.1:8002"