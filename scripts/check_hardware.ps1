# LLMTutor System Hardware Checker (Windows only)
# Run this script on any WINDOWS machine to check if it meets requirements.
#
# HOW TO RUN:
#   Option A (easiest) -- Double-click this file, choose "Run with PowerShell"
#   Option B -- Open PowerShell in this folder and type:
#               powershell -ExecutionPolicy Bypass -File check_hardware.ps1
#   Option C -- From any PowerShell window:
#               powershell -ExecutionPolicy Bypass -File "C:\path\to\check_hardware.ps1"
#
# NOTE: This script is Windows-only (uses WMI). For Linux use check_hardware.sh

$PASS  = "[PASS ]"
$WARN  = "[WARN ]"
$FAIL  = "[FAIL ]"

$score    = 0
$maxScore = 0

function Check($label, $result, $detail, $points) {
    $script:maxScore += $points
    if ($result -eq "PASS") {
        $script:score += $points
        Write-Host "$PASS $label : $detail" -ForegroundColor Green
    } elseif ($result -eq "WARN") {
        $script:score += [int]($points / 2)
        Write-Host "$WARN $label : $detail" -ForegroundColor Yellow
    } else {
        Write-Host "$FAIL $label : $detail" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  LLMTutor Hardware Compatibility Checker" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# =============================================================================
# GPU CHECK
# =============================================================================
Write-Host "--- GPU ---" -ForegroundColor White

$nvidiaGPU = $null
$amdGPU    = $null

# --- Try NVIDIA ---
try {
    $nvsmi = & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -eq 0 -and $nvsmi) {
        $parts     = ($nvsmi -split ",")
        $gpuName   = $parts[0].Trim()
        $vramMB    = [int]($parts[1].Trim())
        $driver    = $parts[2].Trim()
        $vramGB    = [math]::Round($vramMB / 1024, 1)
        $nvidiaGPU = $gpuName

        Write-Host "  GPU detected : $gpuName" -ForegroundColor Cyan
        Write-Host "  VRAM         : ${vramGB} GB"
        Write-Host "  Driver       : $driver"

        if ($vramGB -ge 16) {
            Check "GPU VRAM" "PASS" "${vramGB} GB -- Perfect for all services simultaneously" 20
        } elseif ($vramGB -ge 12) {
            Check "GPU VRAM" "PASS" "${vramGB} GB -- Excellent (12+ GB recommended)" 20
        } elseif ($vramGB -ge 8) {
            Check "GPU VRAM" "WARN" "${vramGB} GB -- Tight. Mega-ASR + Qwen-TTS share ~9-10 GB. May OOM." 20
        } else {
            Check "GPU VRAM" "FAIL" "${vramGB} GB -- Too low. Minimum 8 GB VRAM required." 20
        }

        # CUDA version -- parse from full nvidia-smi output
        $nvsmiFullOutput = (& nvidia-smi 2>$null) -join "`n"
        if ($nvsmiFullOutput -match "CUDA Version:\s*(\d+\.\d+)") {
            $cudaStr   = $Matches[1]
            $cudaParts = $cudaStr.Split(".")
            $cudaMaj   = [int]$cudaParts[0]
            $cudaMin   = [int]$cudaParts[1]
            if ($cudaMaj -gt 12 -or ($cudaMaj -eq 12 -and $cudaMin -ge 4)) {
                Check "CUDA Version" "PASS" "CUDA $cudaStr -- Compatible (need >= 12.4)" 15
            } else {
                Check "CUDA Version" "WARN" "CUDA $cudaStr -- Too old. Update NVIDIA driver." 15
            }
        } else {
            # Fallback: try nvcc
            try {
                $nvccOut = & nvcc --version 2>$null | Select-String "release"
                if ($nvccOut -and $nvccOut -match "release (\d+\.\d+)") {
                    Check "CUDA Version" "PASS" "CUDA $($Matches[1]) (from nvcc)" 15
                } else {
                    Check "CUDA Version" "WARN" "CUDA version undetectable. Ensure NVIDIA driver >= 550 is installed." 15
                }
            } catch {
                Check "CUDA Version" "WARN" "CUDA version undetectable. Ensure NVIDIA driver >= 550 is installed." 15
            }
        }

        Check "GPU Ecosystem" "PASS" "NVIDIA -- Full support: CUDA, faster-whisper, Parakeet (NeMo), all TTS" 10
    }
} catch {}

# --- Try AMD if no NVIDIA ---
if (-not $nvidiaGPU) {
    $gpuObj = Get-WmiObject -Class Win32_VideoController |
              Where-Object { $_.Name -match "AMD|Radeon|ATI" } |
              Select-Object -First 1

    if ($gpuObj) {
        $amdGPU = $gpuObj.Name
        $vramGB = [math]::Round($gpuObj.AdapterRAM / 1GB, 1)
        Write-Host "  GPU detected : $($gpuObj.Name)" -ForegroundColor Cyan
        Write-Host "  VRAM (WMI)   : ${vramGB} GB  (WMI may under-report on AMD -- check AMD Software)"
        Write-Host "  Driver       : $($gpuObj.DriverVersion)"

        if ($vramGB -ge 12) {
            Check "GPU VRAM" "PASS" "${vramGB} GB -- Sufficient for Mega-ASR + Qwen-TTS via ROCm" 20
        } elseif ($vramGB -ge 8) {
            Check "GPU VRAM" "WARN" "${vramGB} GB -- Borderline. Monitor VRAM usage." 20
        } else {
            Check "GPU VRAM" "FAIL" "${vramGB} GB -- Too low. (Note: WMI often mis-reports AMD VRAM. Check AMD Software app)" 20
        }

        $rocmPath = "C:\Program Files\AMD\ROCm"
        if (Test-Path $rocmPath) {
            $rocmDirs = Get-ChildItem $rocmPath -Directory | Select-Object -Last 1
            Check "AMD ROCm Stack" "PASS" "ROCm installed (version: $($rocmDirs.Name))" 15
        } else {
            Check "AMD ROCm Stack" "FAIL" "ROCm not installed. Required for GPU acceleration. See rocm.docs.amd.com/en/latest/deploy/windows/" 15
        }

        Check "GPU Ecosystem" "WARN" "AMD ROCm -- PyTorch OK, Mega-ASR OK, Qwen-TTS OK. Parakeet STT NOT supported." 10

    } else {
        $anyGPU = Get-WmiObject -Class Win32_VideoController | Select-Object -First 1
        $anyName = if ($anyGPU) { $anyGPU.Name } else { "None detected" }
        Check "GPU" "FAIL" "No NVIDIA or AMD GPU detected. Found: $anyName" 45
    }
}

Write-Host ""

# =============================================================================
# RAM CHECK
# =============================================================================
Write-Host "--- System RAM ---" -ForegroundColor White

$ramBytes = (Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory
$ramGB    = [math]::Round($ramBytes / 1GB, 1)
Write-Host "  Total RAM : ${ramGB} GB"

if ($ramGB -ge 32) {
    Check "System RAM" "PASS" "${ramGB} GB -- Excellent (32 GB recommended)" 15
} elseif ($ramGB -ge 16) {
    Check "System RAM" "WARN" "${ramGB} GB -- Minimum met. 4 servers + OS may feel tight." 15
} else {
    Check "System RAM" "FAIL" "${ramGB} GB -- Too low. Minimum 16 GB. 32 GB recommended." 15
}

Write-Host ""

# =============================================================================
# CPU CHECK
# =============================================================================
Write-Host "--- CPU ---" -ForegroundColor White

$cpu     = Get-WmiObject -Class Win32_Processor | Select-Object -First 1
$cores   = $cpu.NumberOfCores
$threads = $cpu.NumberOfLogicalProcessors
Write-Host "  CPU     : $($cpu.Name)"
Write-Host "  Cores   : $cores physical / $threads logical"
Write-Host "  Speed   : $($cpu.MaxClockSpeed) MHz"

if ($cores -ge 12) {
    Check "CPU Cores" "PASS" "$cores cores -- Excellent for parallel server workload" 10
} elseif ($cores -ge 8) {
    Check "CPU Cores" "PASS" "$cores cores -- Good (8+ recommended)" 10
} elseif ($cores -ge 6) {
    Check "CPU Cores" "WARN" "$cores cores -- OK but marginal with 4 services + frontend" 10
} else {
    Check "CPU Cores" "FAIL" "$cores cores -- Too few. Need at least 6." 10
}

Write-Host ""

# =============================================================================
# STORAGE CHECK
# =============================================================================
Write-Host "--- Storage ---" -ForegroundColor White

# Check if C: is SSD
try {
    $diskInfo = Get-PhysicalDisk | Select-Object -First 1
    $diskType = if ($diskInfo) { $diskInfo.MediaType } else { "Unknown" }
    if ($diskType -match "SSD|Solid") {
        Check "Disk Type" "PASS" "SSD detected -- Fast model loading" 5
    } elseif ($diskType -match "HDD|Unspecified") {
        Check "Disk Type" "WARN" "HDD detected -- Model loading will be slow. SSD strongly recommended." 5
    } else {
        Check "Disk Type" "WARN" "Disk type: $diskType" 5
    }
} catch {
    Check "Disk Type" "WARN" "Could not detect disk type." 5
}

$cDrive = Get-PSDrive C -ErrorAction SilentlyContinue
if ($cDrive) {
    $freeGB  = [math]::Round($cDrive.Free / 1GB, 1)
    $totalGB = [math]::Round(($cDrive.Used + $cDrive.Free) / 1GB, 1)
    Write-Host "  C: Drive  : ${freeGB} GB free / ${totalGB} GB total"
    Write-Host ""
    Write-Host "  Space breakdown required:" -ForegroundColor Gray
    Write-Host "    GGUF models (Llama-3 Q4+Q5+Q6):  ~16  GB" -ForegroundColor Gray
    Write-Host "    Mega-ASR checkpoint:              ~5-8 GB" -ForegroundColor Gray
    Write-Host "    Qwen3-TTS models (HF cache):      ~4-6 GB" -ForegroundColor Gray
    Write-Host "    4x Python virtual envs:           ~25  GB" -ForegroundColor Gray
    Write-Host "    Code + data + frontend:           ~5   GB" -ForegroundColor Gray
    Write-Host "    TOTAL estimate:                   ~55-80 GB" -ForegroundColor Gray
    Write-Host ""

    if ($freeGB -ge 120) {
        Check "Free Disk Space" "PASS" "${freeGB} GB free -- Plenty of room" 10
    } elseif ($freeGB -ge 80) {
        Check "Free Disk Space" "PASS" "${freeGB} GB free -- Sufficient" 10
    } elseif ($freeGB -ge 50) {
        Check "Free Disk Space" "WARN" "${freeGB} GB free -- Tight. May not fit all GGUF models." 10
    } else {
        Check "Free Disk Space" "FAIL" "${freeGB} GB free -- Not enough. Clear space or use another drive." 10
    }
}

Write-Host ""

# =============================================================================
# SOFTWARE CHECK
# =============================================================================
Write-Host "--- Installed Software ---" -ForegroundColor White

# Python 3.11
$py311Found = $false
try {
    $pyOut = & py -3.11 --version 2>$null
    if ($pyOut -match "3\.11") { $py311Found = $true; Check "Python 3.11" "PASS" "$pyOut" 10 }
} catch {}
if (-not $py311Found) {
    try {
        $pyOut = & python --version 2>$null
        if ($pyOut -match "3\.11") { $py311Found = $true; Check "Python 3.11" "PASS" "$pyOut" 10 }
    } catch {}
}
if (-not $py311Found) {
    $anyPy = & python --version 2>$null
    Check "Python 3.11" "FAIL" "Not found (found: $anyPy). Get from: python.org/downloads/release/python-3119/" 10
}

# Node.js
try {
    $nodeVer = & node --version 2>$null
    $nodeMaj = [int]($nodeVer -replace "v(\d+)\..*", '$1')
    if ($nodeMaj -ge 18) {
        Check "Node.js" "PASS" "$nodeVer (need v18+)" 5
    } else {
        Check "Node.js" "WARN" "$nodeVer -- Too old. Update at nodejs.org." 5
    }
} catch {
    Check "Node.js" "FAIL" "Not installed. Required for Next.js frontend. Get from nodejs.org." 5
}

# Git
try {
    $gitVer = & git --version 2>$null
    if ($gitVer) { Check "Git" "PASS" "$gitVer" 3 }
} catch {
    Check "Git" "FAIL" "Not installed. Get from git-scm.com." 3
}

# FFmpeg
try {
    $ffLine = (& ffmpeg -version 2>$null) | Select-Object -First 1
    if ($ffLine) { Check "FFmpeg" "PASS" "Installed" 3 }
} catch {
    Check "FFmpeg" "WARN" "Not found in PATH. Get from gyan.dev/ffmpeg/builds/" 3
}

# OpenSSH
try {
    $sshVer = & ssh -V 2>&1
    if ($sshVer -match "OpenSSH") { Check "OpenSSH Client" "PASS" "$sshVer" 3 }
} catch {
    Check "OpenSSH Client" "FAIL" "Not found. Run as Admin: Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0" 3
}

Write-Host ""

# =============================================================================
# NETWORK / CLUSTER CHECK
# =============================================================================
Write-Host "--- Network & Cluster ---" -ForegroundColor White

# Internet -- use TCP connection (ICMP ping is often blocked by Windows firewall)
try {
    $tcpClient = New-Object System.Net.Sockets.TcpClient
    $connect   = $tcpClient.BeginConnect("www.google.com", 443, $null, $null)
    $waited    = $connect.AsyncWaitHandle.WaitOne(3000, $false)
    if ($waited -and $tcpClient.Connected) {
        $tcpClient.Close()
        Check "Internet" "PASS" "Connected (TCP to google.com:443 OK)" 5
    } else {
        $tcpClient.Close()
        Check "Internet" "FAIL" "No internet. Required for cluster SSH tunnel and HuggingFace downloads." 5
    }
} catch {
    Check "Internet" "WARN" "Could not test internet connectivity." 5
}

# SSH key
$possibleKeys = @(
    "$env:USERPROFILE\.ssh\llmtutor_ed25519",
    "C:\Users\shaik\.ssh\llmtutor_ed25519",
    "C:\Users\Admin\.ssh\llmtutor_ed25519"
)
$foundKey = $null
foreach ($k in $possibleKeys) {
    if (Test-Path $k) { $foundKey = $k; break }
}

if ($foundKey) {
    Check "Cluster SSH Key" "PASS" "Found at: $foundKey" 5

    Write-Host "  Testing SSH to cluster (timeout: 12s)..." -ForegroundColor Gray
    try {
        $clusterResult = & ssh -i $foundKey `
            -o IdentitiesOnly=yes `
            -o BatchMode=yes `
            -o ConnectTimeout=12 `
            "ovyj81af@sofja.urz.uni-magdeburg.de" `
            "echo ClusterOK" 2>$null

        if ($clusterResult -match "ClusterOK") {
            Check "Cluster SSH" "PASS" "Connected to sofja.urz.uni-magdeburg.de!" 10
        } else {
            Check "Cluster SSH" "WARN" "Key found but cluster not reachable. Check university VPN." 10
        }
    } catch {
        Check "Cluster SSH" "WARN" "Key found but SSH test failed. Check university VPN / firewall." 10
    }
} else {
    Check "Cluster SSH Key" "FAIL" "llmtutor_ed25519 not found in ~/.ssh/. Copy from old machine." 5
    Check "Cluster SSH"     "FAIL" "Cannot test -- SSH key missing." 10
}

Write-Host ""

# =============================================================================
# FINAL VERDICT
# =============================================================================
$pct = [int](($score / $maxScore) * 100)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  SCORE  : $score / $maxScore ($pct%)" -ForegroundColor White
Write-Host ""

if ($pct -ge 85) {
    Write-Host "  VERDICT: EXCELLENT - This system is ready for LLMTutor!" -ForegroundColor Green
} elseif ($pct -ge 65) {
    Write-Host "  VERDICT: GOOD - Minor issues. Should work with small adjustments." -ForegroundColor Yellow
} elseif ($pct -ge 45) {
    Write-Host "  VERDICT: MARGINAL - Several problems. Some features will not work." -ForegroundColor Yellow
} else {
    Write-Host "  VERDICT: NOT SUITABLE - Major requirements missing." -ForegroundColor Red
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  IDEAL SYSTEM SPEC for LLMTutor:" -ForegroundColor Cyan
Write-Host "  GPU    : NVIDIA RTX 3080 Ti / RTX 4070 Ti or better (12+ GB VRAM)"
Write-Host "  VRAM   : 12-16 GB (10 GB minimum for Mega-ASR + Qwen-TTS together)"
Write-Host "  RAM    : 32 GB (16 GB minimum)"
Write-Host "  CPU    : 8+ cores (Intel i7/i9 12th+ gen or AMD Ryzen 7/9)"
Write-Host "  Disk   : NVMe SSD with 100+ GB free"
Write-Host "  CUDA   : 12.4+ (comes with NVIDIA driver >= 550)"
Write-Host "  Python : 3.11.9 exactly"
Write-Host "  Node   : v18 or newer"
Write-Host ""
Write-Host "  MINIMUM (degraded performance):"
Write-Host "  GPU    : NVIDIA RTX 3070 / GTX 1080 Ti (8 GB VRAM)"
Write-Host "  RAM    : 16 GB"
Write-Host "  CPU    : 6 cores"
Write-Host "  Disk   : 80 GB free"
Write-Host ""
