$ErrorActionPreference = "Stop"

# =========================
# LLMTutor full stack starter
# Starts:
#   1) Remote Gemma + Llama + Qwen + Judge Slurm jobs
#   2) SSH tunnel for local ports 8000/8002/8003/8004
#   3) Qwen streaming TTS server on 8094
#   4) Mega-ASR STT server on 8095
#   5) Parakeet STT server on 8096
#   6) frontend_api on 8001
#   7) Next.js frontend on 3000
# =========================

$ClusterUser = "ovyj81af"
$ClusterLogin = "sofja.urz.uni-magdeburg.de"
$SshKey = "C:/Users/shaik/.ssh/llmtutor_ed25519"

$RemoteProject = "~/llmtutor"
$LocalProject = "C:\Users\shaik\Desktop\llmtutor"
$QwenRuntime = Join-Path $LocalProject "qwen_stream_runtime"
$MegaASRRuntime = Join-Path $LocalProject "mega_asr_runtime"
$ParakeetRuntime = Join-Path $LocalProject "parakeet_runtime"
$FrontendDir = Join-Path $LocalProject "frontend"

$GemmaModel = "gemma4_26b_q3"
$GemmaPort = 8002

$LlamaModel = "llama3_8b_q4"
$LlamaPort = 8000

$QwenLlmModel = "qwen3_5_9b_q6"
$QwenLlmPort = 8003

$JudgeModel = "glm4_32b_0414_q4"
$JudgePort = 8004

$FrontendApiPort = 8001
$QwenStreamPort = 8094
$MegaASRPort = 8095
$ParakeetPort = 8096
$FrontendPort = 3000

$MainPython = Join-Path $LocalProject ".venv\Scripts\python.exe"
$ModelRegistryFile = Join-Path $LocalProject "llm\llmcli\models\registry.py"
$QwenRegistryFile = Join-Path $LocalProject "llm\llmcli\models\qwen_gguf.py"
$GlmRegistryFile = Join-Path $LocalProject "llm\llmcli\models\glm_gguf.py"
$LlamaCppProviderFile = Join-Path $LocalProject "llm\llmcli\providers\llamacpp.py"
$RemoteLlmServerFile = Join-Path $LocalProject "app\remote_llm_server.py"
$QwenPython = Join-Path $QwenRuntime ".venv\Scripts\python.exe"
$MegaASRPython = Join-Path $MegaASRRuntime ".venv\Scripts\python.exe"
$ParakeetPython = Join-Path $ParakeetRuntime ".venv\Scripts\python.exe"

function Assert-FileExists($Path, $Label) {
    if (-not (Test-Path $Path)) {
        throw "$Label not found at: $Path"
    }
}

function Stop-Port($Port, $Label) {
    Write-Host "Checking local port $Port ($Label)..."
    $lines = netstat -ano | Select-String ":$Port"
    if (-not $lines) {
        Write-Host "  Port $Port is free."
        return
    }

    $pids = @()
    foreach ($line in $lines) {
        $parts = ($line.ToString() -split "\s+") | Where-Object { $_ -ne "" }
        if ($parts.Length -ge 5 -and $parts[1] -match ":$Port$") {
            $pidValue = $parts[-1]
            if ($pidValue -match "^\d+$") {
                $pids += [int]$pidValue
            }
        }
    }

    $pids = $pids | Sort-Object -Unique
    foreach ($pidValue in $pids) {
        if ($pidValue -eq $PID) {
            continue
        }
        Write-Host "  Killing PID $pidValue on port $Port"
        try {
            taskkill /PID $pidValue /F | Out-Null
        } catch {
            Write-Host "  Warning: could not kill PID $pidValue"
        }
    }
}

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

function Copy-FileToCluster($LocalPath, $RemotePath, $Label) {
    Write-Host "Uploading $Label..."
    & scp.exe `
        -i $SshKey `
        -o IdentitiesOnly=yes `
        -o BatchMode=yes `
        -o ConnectTimeout=20 `
        $LocalPath `
        "${ClusterUser}@${ClusterLogin}:$RemotePath"

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upload $Label."
    }
}

function Submit-RemoteModelJob(
    $JobName,
    $Model,
    $Port,
    $Label,
    $Partition = $null
) {
    Write-Host "Submitting $Label job..."

    $sbatchOptions = "--parsable --job-name=$JobName"

    if ($Partition) {
        $sbatchOptions += " --partition=$Partition"
    }

    $output = Invoke-ClusterSSH `
        "cd $RemoteProject && sbatch $sbatchOptions scripts/run_remote_model.sbatch $Model $Port"

    $exitCode = $LASTEXITCODE

    if ($output) {
        $output | ForEach-Object { Write-Host $_ }
    }

    if ($exitCode -ne 0) {
        throw "Failed to submit $Label job."
    }

    $jobIdLine = $output |
        Where-Object { $_.ToString().Trim() -match "^\d+(;.*)?$" } |
        Select-Object -First 1

    if (-not $jobIdLine) {
        throw "Could not read the Slurm job ID for $Label."
    }

    $jobId = (($jobIdLine.ToString().Trim()) -split ";")[0]
    Write-Host "${Label} JobID: $jobId"
    return $jobId
}

function Wait-ForRemoteFile($Path, $JobId, $Label, $TimeoutSeconds = 300) {
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

        $state = Invoke-ClusterSSH "sacct -n -j $JobId --format=State 2>/dev/null | head -1 | xargs"
        if ($state -match "^(FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL)") {
            throw "Slurm job $JobId for $Label ended with state $state before creating $Path."
        }

        Write-Host "$Label is waiting for a node (JobID $JobId, state $state)..."
        Start-Sleep -Seconds 5
    }

    throw "Timeout waiting for $Label at $Path (JobID $JobId)."
}

function Wait-ForModelLoaded($JobName, $JobId, $Label, $TimeoutSeconds = 1200) {
    Write-Host "Waiting for $Label model to finish loading..."
    $start = Get-Date
    $logBase = "logs/${JobName}_${JobId}"

    while (((Get-Date) - $start).TotalSeconds -lt $TimeoutSeconds) {
        $ready = Invoke-ClusterSSH "cd $RemoteProject && grep -E '\[SERVER\] Model loaded.|Uvicorn running on' ${logBase}.out ${logBase}.err 2>/dev/null | tail -1"
        $errorText = Invoke-ClusterSSH "cd $RemoteProject && grep -E 'cudaMalloc failed|out of memory|Failed to load model|Traceback|command not found|module: command not found|xpython' ${logBase}.err 2>/dev/null | tail -5"
        $state = Invoke-ClusterSSH "sacct -n -j $JobId --format=State 2>/dev/null | head -1 | xargs"

        if ($ready) {
            Write-Host "$Label loaded and server started."
            return
        }

        if ($state -match "^(FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL)") {
            $recentLogs = Invoke-ClusterSSH "cd $RemoteProject && tail -n 80 ${logBase}.err ${logBase}.out 2>/dev/null"
            if ($recentLogs) {
                Write-Host $recentLogs
            }
            throw "$Label Slurm job $JobId ended with state $state."
        }

        if ($errorText) {
            Write-Host ""
            Write-Host "$Label failed while loading. Recent error:"
            Write-Host $errorText
            throw "$Label model failed to load. Check ${logBase}.err"
        }

        Write-Host "$Label is still loading (JobID $JobId, state $state)..."
        Start-Sleep -Seconds 15
    }

    throw "Timeout waiting for $Label model to load (JobID $JobId)."
}

function Test-Health($Url, $Label, $TimeoutSeconds = 300, $RequiredText = $null) {
    Write-Host "Testing health for $Label at $Url..."
    $start = Get-Date

    while (((Get-Date) - $start).TotalSeconds -lt $TimeoutSeconds) {
        try {
            $response = curl.exe -s --max-time 5 $Url
            if ($LASTEXITCODE -eq 0 -and $response) {
                if ($RequiredText -and ($response -notlike "*$RequiredText*")) {
                    Write-Host "${Label} responded, but did not include required text '$RequiredText': $response"
                } else {
                    Write-Host "${Label} health OK: $response"
                    return $response
                }
            }
        } catch {}

        Start-Sleep -Seconds 5
    }

    throw "Health check failed for $Label at $Url"
}

function Ensure-PythonPackage($PythonExe, $ImportName, $PipArgs, $Label) {
    Write-Host "Checking dependency: $Label"
    & $PythonExe -c "import $ImportName; print('OK')" 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  $Label already installed."
        return
    }

    Write-Host "  Installing $Label..."
    & $PythonExe -m pip install --upgrade @PipArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install $Label"
    }
}

function Test-QwenWebSocket($TimeoutSeconds = 180) {
    Write-Host "Testing Qwen WebSocket stream endpoint..."

    $testScript = @"
import json
import websocket

ws = websocket.create_connection('ws://127.0.0.1:$QwenStreamPort/stream', timeout=20)
ws.send(json.dumps({
    'type': 'synthesize',
    'text': 'Hello, this is a quick Qwen streaming test.',
    'speaker': 'Aiden',
    'language': 'en',
    'chunk_size': 4,
}))
msg1 = ws.recv()
msg2 = ws.recv()
ws.close()
print(msg1[:160])
print(msg2[:160])
if 'audio_start' not in msg1 or 'audio_chunk' not in msg2:
    raise SystemExit('Qwen WebSocket did not return audio_start followed by audio_chunk')
"@

    $tmp = Join-Path $env:TEMP "llmtutor_qwen_ws_test.py"
    Set-Content -Path $tmp -Value $testScript -Encoding UTF8

    $start = Get-Date
    while (((Get-Date) - $start).TotalSeconds -lt $TimeoutSeconds) {
        & $QwenPython $tmp
        if ($LASTEXITCODE -eq 0) {
            Remove-Item $tmp -ErrorAction SilentlyContinue
            Write-Host "Qwen WebSocket OK."
            return
        }
        Start-Sleep -Seconds 5
    }

    Remove-Item $tmp -ErrorAction SilentlyContinue
    throw "Qwen WebSocket test failed at ws://127.0.0.1:$QwenStreamPort/stream"
}

Write-Host "======================================"
Write-Host "Starting full LLMTutor stack"
Write-Host "Remote LLMs (Llama, Gemma, Qwen) + Judge + Qwen streaming TTS + STT + API + frontend"
Write-Host "======================================"

Write-Host "`n[0/14] Validating local files..."
Assert-FileExists $SshKey "SSH key"
Assert-FileExists $LocalProject "Local LLMTutor project"
Assert-FileExists $QwenRuntime "Qwen streaming runtime folder"
Assert-FileExists $MegaASRRuntime "Mega-ASR runtime folder"
Assert-FileExists $ParakeetRuntime "Parakeet runtime folder"
Assert-FileExists $MainPython "Main Python venv"
Assert-FileExists $QwenPython "Qwen runtime Python venv"
Assert-FileExists $MegaASRPython "Mega-ASR runtime Python venv"
Assert-FileExists $ParakeetPython "Parakeet runtime Python venv"
Assert-FileExists (Join-Path $QwenRuntime "qwen_stream_server.py") "qwen_stream_server.py"
Assert-FileExists (Join-Path $MegaASRRuntime "mega_asr_server.py") "mega_asr_server.py"
Assert-FileExists (Join-Path $ParakeetRuntime "parakeet_service\main.py") "parakeet_service/main.py"
Assert-FileExists (Join-Path $LocalProject "frontend_api.py") "frontend_api.py"
Assert-FileExists $ModelRegistryFile "Model registry file"
Assert-FileExists $QwenRegistryFile "Qwen LLM registry file"
Assert-FileExists $GlmRegistryFile "GLM judge registry file"
Assert-FileExists $LlamaCppProviderFile "llama.cpp provider file"
Assert-FileExists $RemoteLlmServerFile "Remote LLM server file"

Write-Host "`n[1/14] Cleaning local ports..."
Stop-Port $FrontendPort "frontend"
Stop-Port $FrontendApiPort "frontend_api"
Stop-Port $QwenStreamPort "Qwen streaming TTS"
Stop-Port $MegaASRPort "Mega-ASR STT"
Stop-Port $ParakeetPort "Parakeet STT"
Stop-Port $LlamaPort "Llama tunnel"
Stop-Port $GemmaPort "Gemma tunnel"
Stop-Port $QwenLlmPort "Qwen LLM tunnel"
Stop-Port $JudgePort "Judge tunnel"

Write-Host "`n[2/14] Ensuring Python dependencies..."
Ensure-PythonPackage $QwenPython "websockets" @('uvicorn[standard]', 'websockets', 'wsproto', 'websocket-client') "Qwen WebSocket server/client dependencies"
Ensure-PythonPackage $MegaASRPython "fastapi" @('fastapi', 'uvicorn[standard]', 'soundfile', 'pydantic') "Mega-ASR API server dependencies"
Ensure-PythonPackage $ParakeetPython "fastapi" @('fastapi', 'uvicorn[standard]', 'python-multipart', 'soundfile') "Parakeet API server dependencies"
Ensure-PythonPackage $MainPython "websocket" @('websocket-client') "Main app websocket-client"
Ensure-PythonPackage $MainPython "requests" @('requests') "Main app requests dependency"

Write-Host "`n[3/14] Preparing cluster runtime..."

# Only these jobs belong to this launcher.
$llmTutorJobNames = "llmtutor_gemma,llmtutor_llama,llmtutor_qwen_llm,llmtutor_judge"

# Retrieve only numeric job IDs owned by the current cluster user.
$activeJobOutput = Invoke-ClusterSSH `
    "squeue -h -u $ClusterUser -n $llmTutorJobNames -o %A"

$activeJobIds = @(
    $activeJobOutput -split "\s+" |
        Where-Object { $_ -match "^\d+$" } |
        Select-Object -Unique
)

if ($activeJobIds.Count -gt 0) {
    $jobIdList = $activeJobIds -join " "

    Write-Host (
        "Stopping previous LLMTutor Slurm jobs: {0}" -f $jobIdList
    ) -ForegroundColor Yellow

    # Cancel explicit IDs rather than every job belonging to the user.
    $null = Invoke-ClusterSSH "scancel $jobIdList"

    # Wait until Slurm has actually removed them before resubmitting.
    $cancelDeadline = (Get-Date).AddSeconds(60)

    do {
        Start-Sleep -Seconds 2

        $remainingOutput = Invoke-ClusterSSH `
            "squeue -h -u $ClusterUser -n $llmTutorJobNames -o %A"

        $remainingJobIds = @(
            $remainingOutput -split "\s+" |
                Where-Object { $_ -match "^\d+$" } |
                Select-Object -Unique
        )
    }
    while (
        $remainingJobIds.Count -gt 0 -and
        (Get-Date) -lt $cancelDeadline
    )

    if ($remainingJobIds.Count -gt 0) {
        throw (
            "Timed out waiting for old LLMTutor jobs to stop: {0}" -f
            ($remainingJobIds -join ", ")
        )
    }

    Write-Host "Previous LLMTutor jobs stopped." -ForegroundColor Green
}
else {
    Write-Host "No previous LLMTutor Slurm jobs found."
}

# Remove runtime markers only after the corresponding jobs are gone.
$null = Invoke-ClusterSSH `
    "cd $RemoteProject && mkdir -p runtime && rm -f runtime/*.host runtime/*.port runtime/*.jobid"

Write-Host "Cluster runtime is clean." -ForegroundColor Green
Write-Host "Cluster model registration and provider files verified."

Write-Host "`n[4/14] Submitting all remote model jobs..."
$GemmaJobId = Submit-RemoteModelJob "llmtutor_gemma" $GemmaModel $GemmaPort "Gemma"
$LlamaJobId = Submit-RemoteModelJob "llmtutor_llama" $LlamaModel $LlamaPort "Llama"
$QwenLlmJobId = Submit-RemoteModelJob "llmtutor_qwen_llm" $QwenLlmModel $QwenLlmPort "Qwen LLM"
$JudgeJobId = Submit-RemoteModelJob `
    "llmtutor_judge" `
    $JudgeModel `
    $JudgePort `
    "Judge" `
    "gpu80GB"

Write-Host "`n[5/14] Waiting for all remote jobs to receive GPU nodes..."
$GemmaHost = Wait-ForRemoteFile "$RemoteProject/runtime/$GemmaModel.host" $GemmaJobId "Gemma host" 14400
$LlamaHost = Wait-ForRemoteFile "$RemoteProject/runtime/$LlamaModel.host" $LlamaJobId "Llama host" 14400
$QwenLlmHost = Wait-ForRemoteFile "$RemoteProject/runtime/$QwenLlmModel.host" $QwenLlmJobId "Qwen LLM host" 14400
$JudgeHost = Wait-ForRemoteFile "$RemoteProject/runtime/$JudgeModel.host" $JudgeJobId "Judge host" 14400

Write-Host "`n[6/14] Waiting for tutor models to load..."
Wait-ForModelLoaded "llmtutor_gemma" $GemmaJobId "Gemma" 1800
Wait-ForModelLoaded "llmtutor_llama" $LlamaJobId "Llama" 1200
Wait-ForModelLoaded "llmtutor_qwen_llm" $QwenLlmJobId "Qwen LLM" 1800

Write-Host "`n[7/14] Waiting for judge model to load..."
Wait-ForModelLoaded "llmtutor_judge" $JudgeJobId "Judge" 1200

Write-Host "`n[8/14] Starting SSH tunnel..."
$TunnelCommand = "ssh -N -v -i $SshKey -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes -L ${LlamaPort}:$LlamaHost`:${LlamaPort} -L ${GemmaPort}:$GemmaHost`:${GemmaPort} -L ${QwenLlmPort}:$QwenLlmHost`:${QwenLlmPort} -L ${JudgePort}:$JudgeHost`:${JudgePort} $ClusterUser@$ClusterLogin"
Write-Host "Tunnel command:"
Write-Host $TunnelCommand
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    $TunnelCommand
)
Start-Sleep -Seconds 5

Write-Host "`n[9/14] Checking remote model health through tunnel..."
Test-Health "http://127.0.0.1:$GemmaPort/health" "Gemma" 300
Test-Health "http://127.0.0.1:$LlamaPort/health" "Llama" 300
Test-Health "http://127.0.0.1:$QwenLlmPort/health" "Qwen LLM" 300
Test-Health "http://127.0.0.1:$JudgePort/health" "Judge" 300

Write-Host "`n[10/14] Starting Qwen streaming TTS server..."
$QwenCommand = "cd '$QwenRuntime'; .\.venv\Scripts\activate; python -m uvicorn qwen_stream_server:app --host 127.0.0.1 --port $QwenStreamPort"
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    $QwenCommand
)
Test-Health "http://127.0.0.1:$QwenStreamPort/health" "Qwen streaming TTS" 900 '"mode":"streaming"'
Test-QwenWebSocket 240


Write-Host "`n[11/14] Starting Mega-ASR STT server..."
$MegaASRCommand = "cd '$MegaASRRuntime'; .\.venv\Scripts\activate; python -m uvicorn mega_asr_server:app --host 127.0.0.1 --port $MegaASRPort"
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    $MegaASRCommand
)
Test-Health "http://127.0.0.1:$MegaASRPort/health" "Mega-ASR STT" 900 '"engine":"mega-asr"'

Write-Host "`n[12/14] Starting Parakeet STT server..."
$ParakeetCommand = "cd '$ParakeetRuntime'; .\.venv\Scripts\activate; `$env:DEVICE='cuda'; `$env:MODEL_PRECISION='fp16'; `$env:TARGET_SR='16000'; python -m uvicorn parakeet_service.main:app --host 127.0.0.1 --port $ParakeetPort"
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    $ParakeetCommand
)
Test-Health "http://127.0.0.1:$ParakeetPort/healthz" "Parakeet STT" 900 '"status":"ok"'

Write-Host "`n[13/14] Starting frontend_api..."
$ApiCommand = "cd '$LocalProject'; .\.venv\Scripts\activate; python -m uvicorn frontend_api:app --host 127.0.0.1 --port $FrontendApiPort"
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    $ApiCommand
)
Test-Health "http://127.0.0.1:$FrontendApiPort/health" "frontend_api" 120

Write-Host "`n[14/14] Starting frontend..."
$FrontendCommand = "cd '$FrontendDir'; npm run dev"
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    $FrontendCommand
)
Start-Sleep -Seconds 5

Write-Host ""
Write-Host "======================================"
Write-Host "LLMTutor stack started. Keep the opened PowerShell windows running."
Write-Host "======================================"
Write-Host "Health checks:"
Write-Host "  Llama              -> http://127.0.0.1:$LlamaPort/health"
Write-Host "  Gemma              -> http://127.0.0.1:$GemmaPort/health"
Write-Host "  Qwen LLM           -> http://127.0.0.1:$QwenLlmPort/health"
Write-Host "  Judge              -> http://127.0.0.1:$JudgePort/health"
Write-Host "  Qwen streaming TTS -> http://127.0.0.1:$QwenStreamPort/health"
Write-Host "  Mega-ASR STT       -> http://127.0.0.1:$MegaASRPort/health"
Write-Host "  Parakeet STT       -> http://127.0.0.1:$ParakeetPort/healthz"
Write-Host "  frontend_api       -> http://127.0.0.1:$FrontendApiPort/health"
Write-Host "  frontend           -> http://localhost:$FrontendPort"
Write-Host ""
Write-Host "For STT tests, in the UI select STT=mega-asr or STT=parakeet with TTS=kokoro first."
Write-Host "The backend should show: [STT CONFIG] Active backend class: MegaASRHTTPBackend."