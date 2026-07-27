$ErrorActionPreference = "Stop"

# =========================
# LLMTutor full stack starter
# Starts:
#   1) Remote Gemma + Llama Slurm jobs
#   2) SSH tunnel for local ports 8000/8002
#   3) Qwen streaming TTS server on 8094
#   4) Mega-ASR STT server on 8095
#   5) frontend_api on 8001
#   6) Next.js frontend on 3000
# =========================

$ClusterUser = "ovyj81af"
$ClusterLogin = "sofja.urz.uni-magdeburg.de"
$SshKey = "C:/Users/shaik/.ssh/llmtutor_ed25519"

$RemoteProject = "~/llmtutor"
$LocalProject = "C:\Users\shaik\Desktop\llmtutor"
$QwenRuntime = Join-Path $LocalProject "qwen_stream_runtime"
$MegaASRRuntime = Join-Path $LocalProject "mega_asr_runtime"
$FrontendDir = Join-Path $LocalProject "frontend"

$GemmaModel = "gemma4_26b_q3"
$GemmaPort = 8002

$LlamaModel = "llama3_8b_q4"
$LlamaPort = 8000

$FrontendApiPort = 8001
$QwenStreamPort = 8094
$MegaASRPort = 8095
$FrontendPort = 3000

$MainPython = Join-Path $LocalProject ".venv\Scripts\python.exe"
$QwenPython = Join-Path $QwenRuntime ".venv\Scripts\python.exe"
$MegaASRPython = Join-Path $MegaASRRuntime ".venv\Scripts\python.exe"

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
Write-Host "Remote LLMs + Qwen streaming TTS + Mega-ASR STT + API + frontend"
Write-Host "======================================"

Write-Host "`n[0/10] Validating local files..."
Assert-FileExists $SshKey "SSH key"
Assert-FileExists $LocalProject "Local LLMTutor project"
Assert-FileExists $QwenRuntime "Qwen streaming runtime folder"
Assert-FileExists $MegaASRRuntime "Mega-ASR runtime folder"
Assert-FileExists $MainPython "Main Python venv"
Assert-FileExists $QwenPython "Qwen runtime Python venv"
Assert-FileExists $MegaASRPython "Mega-ASR runtime Python venv"
Assert-FileExists (Join-Path $QwenRuntime "qwen_stream_server.py") "qwen_stream_server.py"
Assert-FileExists (Join-Path $MegaASRRuntime "mega_asr_server.py") "mega_asr_server.py"
Assert-FileExists (Join-Path $LocalProject "frontend_api.py") "frontend_api.py"

Write-Host "`n[1/10] Cleaning local ports..."
Stop-Port $FrontendPort "frontend"
Stop-Port $FrontendApiPort "frontend_api"
Stop-Port $QwenStreamPort "Qwen streaming TTS"
Stop-Port $MegaASRPort "Mega-ASR STT"
Stop-Port $LlamaPort "Llama tunnel"
Stop-Port $GemmaPort "Gemma tunnel"

Write-Host "`n[2/10] Ensuring Python dependencies..."
Ensure-PythonPackage $QwenPython "websockets" @('uvicorn[standard]', 'websockets', 'wsproto', 'websocket-client') "Qwen WebSocket server/client dependencies"
Ensure-PythonPackage $MegaASRPython "fastapi" @('fastapi', 'uvicorn[standard]', 'soundfile', 'pydantic') "Mega-ASR API server dependencies"
Ensure-PythonPackage $MainPython "websocket" @('websocket-client') "Main app websocket-client"
Ensure-PythonPackage $MainPython "requests" @('requests') "Main app requests dependency"

Write-Host "`n[3/10] Cleaning old remote runtime files..."
Run-SSH "cd $RemoteProject && rm -f runtime/*.host runtime/*.port runtime/*.jobid"

Write-Host "`n[4/10] Submitting Gemma job..."
Run-SSH "cd $RemoteProject && sbatch --job-name=llmtutor_gemma scripts/run_remote_model.sbatch $GemmaModel $GemmaPort"
$GemmaHost = Wait-ForRemoteFile "$RemoteProject/runtime/$GemmaModel.host" "Gemma host" 600
Wait-ForModelLoaded "llmtutor_gemma" "Gemma" 1200

Write-Host "`n[5/10] Submitting Llama job..."
Run-SSH "cd $RemoteProject && sbatch --job-name=llmtutor_llama scripts/run_remote_model.sbatch $LlamaModel $LlamaPort"
$LlamaHost = Wait-ForRemoteFile "$RemoteProject/runtime/$LlamaModel.host" "Llama host" 300
Wait-ForModelLoaded "llmtutor_llama" "Llama" 600

Write-Host "`n[6/10] Starting SSH tunnel..."
$TunnelCommand = "ssh -N -v -i $SshKey -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes -L ${LlamaPort}:$LlamaHost`:${LlamaPort} -L ${GemmaPort}:$GemmaHost`:${GemmaPort} $ClusterUser@$ClusterLogin"
Write-Host "Tunnel command:"
Write-Host $TunnelCommand
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    $TunnelCommand
)
Start-Sleep -Seconds 5

Write-Host "`n[7/10] Checking remote model health through tunnel..."
Test-Health "http://127.0.0.1:$GemmaPort/health" "Gemma" 300
Test-Health "http://127.0.0.1:$LlamaPort/health" "Llama" 300

Write-Host "`n[8/10] Starting Qwen streaming TTS server..."
$QwenCommand = "cd '$QwenRuntime'; .\.venv\Scripts\activate; python -m uvicorn qwen_stream_server:app --host 127.0.0.1 --port $QwenStreamPort"
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    $QwenCommand
)
Test-Health "http://127.0.0.1:$QwenStreamPort/health" "Qwen streaming TTS" 900 '"mode":"streaming"'
Test-QwenWebSocket 240


Write-Host "`n[9/11] Starting Mega-ASR STT server..."
$MegaASRCommand = "cd '$MegaASRRuntime'; .\.venv\Scripts\activate; python -m uvicorn mega_asr_server:app --host 127.0.0.1 --port $MegaASRPort"
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    $MegaASRCommand
)
Test-Health "http://127.0.0.1:$MegaASRPort/health" "Mega-ASR STT" 900 '"engine":"mega-asr"'

Write-Host "`n[10/11] Starting frontend_api..."
$ApiCommand = "cd '$LocalProject'; .\.venv\Scripts\activate; python -m uvicorn frontend_api:app --host 127.0.0.1 --port $FrontendApiPort"
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    $ApiCommand
)
Test-Health "http://127.0.0.1:$FrontendApiPort/health" "frontend_api" 120

Write-Host "`n[11/11] Starting frontend..."
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
Write-Host "  Qwen streaming TTS -> http://127.0.0.1:$QwenStreamPort/health"
Write-Host "  Mega-ASR STT       -> http://127.0.0.1:$MegaASRPort/health"
Write-Host "  frontend_api       -> http://127.0.0.1:$FrontendApiPort/health"
Write-Host "  frontend           -> http://localhost:$FrontendPort"
Write-Host ""
Write-Host "For Mega-ASR test, in the UI select: STT=mega-asr, TTS=kokoro first."
Write-Host "The backend should show: [STT CONFIG] Active backend class: MegaASRHTTPBackend."
