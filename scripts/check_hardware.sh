#!/usr/bin/env bash
# LLMTutor System Hardware Checker (Linux / WSL2)
#
# HOW TO RUN:
#   chmod +x check_hardware.sh   (only needed once, to make it executable)
#   ./check_hardware.sh
#
# Works on: Ubuntu 20.04+, Debian, any Linux distro with bash
# Also works inside WSL2 on Windows

PASS="[PASS ]"
WARN="[WARN ]"
FAIL="[FAIL ]"
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'  # No Colour

score=0
maxScore=0

check() {
    local label="$1"
    local result="$2"
    local detail="$3"
    local points="$4"
    maxScore=$((maxScore + points))
    if [ "$result" = "PASS" ]; then
        score=$((score + points))
        echo -e "${GREEN}${PASS} ${label} : ${detail}${NC}"
    elif [ "$result" = "WARN" ]; then
        score=$((score + points / 2))
        echo -e "${YELLOW}${WARN} ${label} : ${detail}${NC}"
    else
        echo -e "${RED}${FAIL} ${label} : ${detail}${NC}"
    fi
}

echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}  LLMTutor Hardware Compatibility Checker (Linux)${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# =============================================================================
# GPU CHECK
# =============================================================================
echo -e "--- GPU ---"

nvidia_found=false

if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$GPU_INFO" ]; then
        nvidia_found=true
        GPU_NAME=$(echo "$GPU_INFO" | cut -d',' -f1 | xargs)
        VRAM_MB=$(echo "$GPU_INFO"  | cut -d',' -f2 | xargs)
        DRIVER=$(echo "$GPU_INFO"   | cut -d',' -f3 | xargs)
        VRAM_GB=$(echo "scale=1; $VRAM_MB / 1024" | bc)

        echo "  GPU detected : $GPU_NAME"
        echo "  VRAM         : ${VRAM_GB} GB"
        echo "  Driver       : $DRIVER"

        VRAM_INT=$(echo "$VRAM_GB" | cut -d'.' -f1)
        if   [ "$VRAM_INT" -ge 16 ]; then check "GPU VRAM" "PASS" "${VRAM_GB} GB -- Perfect for all services simultaneously" 20
        elif [ "$VRAM_INT" -ge 12 ]; then check "GPU VRAM" "PASS" "${VRAM_GB} GB -- Excellent (12+ GB recommended)" 20
        elif [ "$VRAM_INT" -ge 8  ]; then check "GPU VRAM" "WARN" "${VRAM_GB} GB -- Tight. Mega-ASR + Qwen-TTS share ~9-10 GB. May OOM." 20
        else                              check "GPU VRAM" "FAIL" "${VRAM_GB} GB -- Too low. Minimum 8 GB VRAM required." 20
        fi

        CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[\d.]+")
        if [ -n "$CUDA_VER" ]; then
            CUDA_MAJ=$(echo "$CUDA_VER" | cut -d'.' -f1)
            CUDA_MIN=$(echo "$CUDA_VER" | cut -d'.' -f2)
            if [ "$CUDA_MAJ" -gt 12 ] || { [ "$CUDA_MAJ" -eq 12 ] && [ "$CUDA_MIN" -ge 4 ]; }; then
                check "CUDA Version" "PASS" "CUDA $CUDA_VER -- Compatible (need >= 12.4)" 15
            else
                check "CUDA Version" "WARN" "CUDA $CUDA_VER -- Too old. Update NVIDIA driver." 15
            fi
        else
            check "CUDA Version" "WARN" "Could not read CUDA version. Check nvidia-smi output manually." 15
        fi

        check "GPU Ecosystem" "PASS" "NVIDIA -- Full support: CUDA, faster-whisper, Parakeet (NeMo), all TTS" 10
    fi
fi

if [ "$nvidia_found" = false ]; then
    # Check AMD ROCm
    if command -v rocminfo &> /dev/null || [ -d "/opt/rocm" ]; then
        GPU_NAME=$(rocminfo 2>/dev/null | grep -m1 "Marketing Name" | sed 's/.*: //')
        echo "  GPU detected : ${GPU_NAME:-AMD ROCm GPU}"
        check "GPU VRAM" "WARN" "AMD ROCm GPU -- check VRAM manually with: rocm-smi --showmeminfo vram" 20
        check "AMD ROCm"  "PASS" "ROCm installed at /opt/rocm" 15
        check "GPU Ecosystem" "WARN" "AMD -- ROCm PyTorch OK, Mega-ASR OK, Qwen-TTS OK. Parakeet NOT supported." 10
    else
        check "GPU" "FAIL" "No NVIDIA or AMD GPU detected. Run: lspci | grep -i vga" 45
    fi
fi

echo ""

# =============================================================================
# RAM CHECK
# =============================================================================
echo "--- System RAM ---"

RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
RAM_GB=$(echo "scale=1; $RAM_KB / 1024 / 1024" | bc)
echo "  Total RAM : ${RAM_GB} GB"
RAM_INT=$(echo "$RAM_GB" | cut -d'.' -f1)

if   [ "$RAM_INT" -ge 32 ]; then check "System RAM" "PASS" "${RAM_GB} GB -- Excellent (32 GB recommended)" 15
elif [ "$RAM_INT" -ge 16 ]; then check "System RAM" "WARN" "${RAM_GB} GB -- Minimum met. 4 servers + OS may feel tight." 15
else                              check "System RAM" "FAIL" "${RAM_GB} GB -- Too low. Minimum 16 GB. 32 GB recommended." 15
fi

echo ""

# =============================================================================
# CPU CHECK
# =============================================================================
echo "--- CPU ---"

CPU_NAME=$(grep -m1 "model name" /proc/cpuinfo | cut -d':' -f2 | xargs)
CORES=$(grep -c "^processor" /proc/cpuinfo)
echo "  CPU   : $CPU_NAME"
echo "  Cores : $CORES logical"

if   [ "$CORES" -ge 12 ]; then check "CPU Cores" "PASS" "$CORES cores -- Excellent for parallel server workload" 10
elif [ "$CORES" -ge 8  ]; then check "CPU Cores" "PASS" "$CORES cores -- Good (8+ recommended)" 10
elif [ "$CORES" -ge 6  ]; then check "CPU Cores" "WARN" "$CORES cores -- OK but marginal with 4 services + frontend" 10
else                           check "CPU Cores" "FAIL" "$CORES cores -- Too few. Need at least 6." 10
fi

echo ""

# =============================================================================
# STORAGE CHECK
# =============================================================================
echo "--- Storage ---"

# Check disk type for / (root)
DISK_TYPE="Unknown"
ROOT_DISK=$(df / | tail -1 | awk '{print $1}' | sed 's/[0-9]*$//' | sed 's|/dev/||')
if [ -f "/sys/block/${ROOT_DISK}/queue/rotational" ]; then
    ROTATIONAL=$(cat "/sys/block/${ROOT_DISK}/queue/rotational")
    if [ "$ROTATIONAL" = "0" ]; then DISK_TYPE="SSD/NVMe"; else DISK_TYPE="HDD"; fi
fi

echo "  Disk type : $DISK_TYPE"
if [[ "$DISK_TYPE" == *"SSD"* ]] || [[ "$DISK_TYPE" == *"NVMe"* ]]; then
    check "Disk Type" "PASS" "SSD/NVMe -- Fast model loading" 5
else
    check "Disk Type" "WARN" "$DISK_TYPE -- HDD is slow for loading models. SSD strongly recommended." 5
fi

FREE_KB=$(df --output=avail / | tail -1)
FREE_GB=$(echo "scale=1; $FREE_KB / 1024 / 1024" | bc)
TOTAL_KB=$(df --output=size / | tail -1)
TOTAL_GB=$(echo "scale=1; $TOTAL_KB / 1024 / 1024" | bc)
echo "  Free space: ${FREE_GB} GB free / ${TOTAL_GB} GB total"
echo ""
echo "  Space breakdown required:"
echo "    GGUF models (Llama-3 Q4+Q5+Q6):  ~16  GB"
echo "    Mega-ASR checkpoint:              ~5-8 GB"
echo "    Qwen3-TTS models (HF cache):      ~4-6 GB"
echo "    4x Python virtual envs:           ~25  GB"
echo "    Code + data + frontend:           ~5   GB"
echo "    TOTAL estimate:                   ~55-80 GB"
echo ""

FREE_INT=$(echo "$FREE_GB" | cut -d'.' -f1)
if   [ "$FREE_INT" -ge 120 ]; then check "Free Disk Space" "PASS" "${FREE_GB} GB free -- Plenty of room" 10
elif [ "$FREE_INT" -ge 80  ]; then check "Free Disk Space" "PASS" "${FREE_GB} GB free -- Sufficient" 10
elif [ "$FREE_INT" -ge 50  ]; then check "Free Disk Space" "WARN" "${FREE_GB} GB free -- Tight. May not fit all GGUF models." 10
else                                check "Free Disk Space" "FAIL" "${FREE_GB} GB free -- Not enough. Clear space." 10
fi

echo ""

# =============================================================================
# SOFTWARE CHECK
# =============================================================================
echo "--- Installed Software ---"

# Python 3.11
PY311=""
if command -v python3.11 &>/dev/null; then
    PY311=$(python3.11 --version 2>&1)
elif command -v python3 &>/dev/null; then
    PY_VER=$(python3 --version 2>&1)
    [[ "$PY_VER" == *"3.11"* ]] && PY311="$PY_VER"
fi

if [ -n "$PY311" ]; then
    check "Python 3.11" "PASS" "$PY311" 10
else
    ANY_PY=$(python3 --version 2>&1 || echo "Not found")
    check "Python 3.11" "FAIL" "Not found (found: $ANY_PY). Install: sudo apt install python3.11 python3.11-venv" 10
fi

# Node.js
if command -v node &>/dev/null; then
    NODE_VER=$(node --version)
    NODE_MAJ=$(echo "$NODE_VER" | tr -d 'v' | cut -d'.' -f1)
    if [ "$NODE_MAJ" -ge 18 ]; then check "Node.js" "PASS" "$NODE_VER (need v18+)" 5
    else                             check "Node.js" "WARN" "$NODE_VER -- Too old. Install from nodejs.org." 5
    fi
else
    check "Node.js" "FAIL" "Not installed. Run: curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo bash - && sudo apt install nodejs" 5
fi

# Git
if command -v git &>/dev/null; then check "Git" "PASS" "$(git --version)" 3
else                                  check "Git" "FAIL" "Not installed. Run: sudo apt install git" 3
fi

# FFmpeg
if command -v ffmpeg &>/dev/null; then check "FFmpeg" "PASS" "Installed ($(ffmpeg -version 2>&1 | head -1 | awk '{print $3}'))" 3
else                                     check "FFmpeg" "WARN" "Not installed. Run: sudo apt install ffmpeg" 3
fi

# SSH client
if command -v ssh &>/dev/null; then check "OpenSSH Client" "PASS" "$(ssh -V 2>&1)" 3
else                                  check "OpenSSH Client" "FAIL" "Not installed. Run: sudo apt install openssh-client" 3
fi

echo ""

# =============================================================================
# NETWORK / CLUSTER CHECK
# =============================================================================
echo "--- Network & Cluster ---"

# Internet
if curl -s --connect-timeout 3 https://www.google.com -o /dev/null 2>/dev/null; then
    check "Internet" "PASS" "Connected" 5
else
    check "Internet" "FAIL" "No internet. Required for cluster SSH tunnel and HuggingFace model downloads." 5
fi

# SSH key
KEY_FOUND=""
for KEY_PATH in "$HOME/.ssh/llmtutor_ed25519" "/home/shaik/.ssh/llmtutor_ed25519" "/home/admin/.ssh/llmtutor_ed25519"; do
    if [ -f "$KEY_PATH" ]; then KEY_FOUND="$KEY_PATH"; break; fi
done

if [ -n "$KEY_FOUND" ]; then
    check "Cluster SSH Key" "PASS" "Found at: $KEY_FOUND" 5

    echo "  Testing SSH to cluster (timeout: 12s)..."
    RESULT=$(ssh -i "$KEY_FOUND" \
        -o IdentitiesOnly=yes \
        -o BatchMode=yes \
        -o ConnectTimeout=12 \
        "ovyj81af@sofja.urz.uni-magdeburg.de" \
        "echo ClusterOK" 2>/dev/null)

    if [[ "$RESULT" == *"ClusterOK"* ]]; then
        check "Cluster SSH" "PASS" "Connected to sofja.urz.uni-magdeburg.de!" 10
    else
        check "Cluster SSH" "WARN" "Key found but cluster not reachable. Check university VPN." 10
    fi
else
    check "Cluster SSH Key" "FAIL" "llmtutor_ed25519 not found in ~/.ssh/. Copy from old machine." 5
    check "Cluster SSH"     "FAIL" "Cannot test -- SSH key missing." 10
fi

echo ""

# =============================================================================
# FINAL VERDICT
# =============================================================================
PCT=$(echo "scale=0; $score * 100 / $maxScore" | bc)

echo -e "${CYAN}============================================================${NC}"
echo ""
echo "  SCORE  : $score / $maxScore ($PCT%)"
echo ""

if   [ "$PCT" -ge 85 ]; then echo -e "  ${GREEN}VERDICT: EXCELLENT - This system is ready for LLMTutor!${NC}"
elif [ "$PCT" -ge 65 ]; then echo -e "  ${YELLOW}VERDICT: GOOD - Minor issues. Should work with small adjustments.${NC}"
elif [ "$PCT" -ge 45 ]; then echo -e "  ${YELLOW}VERDICT: MARGINAL - Several problems. Some features will not work.${NC}"
else                          echo -e "  ${RED}VERDICT: NOT SUITABLE - Major requirements missing.${NC}"
fi

echo ""
echo -e "${CYAN}============================================================${NC}"
echo ""
echo "  IDEAL SYSTEM SPEC for LLMTutor:"
echo "  GPU    : NVIDIA RTX 3080 Ti / RTX 4070 Ti or better (12+ GB VRAM)"
echo "  VRAM   : 12-16 GB  (10 GB minimum for Mega-ASR + Qwen-TTS)"
echo "  RAM    : 32 GB  (16 GB minimum)"
echo "  CPU    : 8+ cores"
echo "  Disk   : NVMe SSD with 100+ GB free"
echo "  CUDA   : 12.4+  (comes with NVIDIA driver >= 550)"
echo "  Python : 3.11.x exactly"
echo "  Node   : v18 or newer"
echo ""
echo "  MINIMUM (degraded performance):"
echo "  GPU    : NVIDIA RTX 3070 / GTX 1080 Ti (8 GB VRAM)"
echo "  RAM    : 16 GB"
echo "  CPU    : 6 cores"
echo "  Disk   : 80 GB free"
echo ""
