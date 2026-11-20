$ErrorActionPreference = 'Stop'

# The normal Windows launcher starts remote jobs but does not copy these two
# Python files to Sofja. Stage both files first and keep the previous versions.
$project = Split-Path -Parent $PSScriptRoot
$key = Join-Path $env:USERPROFILE '.ssh\llmtutor_ed25519'
$remote = 'ovyj81af@sofja.urz.uni-magdeburg.de'
$id = (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + ([guid]::NewGuid().ToString('N').Substring(0, 8))
$stage = '.llmtutor-metrics-stage-' + $id
$backup = 'backups/llama3-metrics-' + $id
$server = Join-Path $project 'app\remote_llm_server.py'
$provider = Join-Path $project 'llm\llmcli\providers\llamacpp.py'

foreach ($path in @($key, $server, $provider)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required file not found: $path"
    }
}

function Invoke-Executable([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}

$sshOptions = @('-i', $key, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20')
$prepare = "set -eu; cd ~/llmtutor; test -f app/remote_llm_server.py; test -f llm/llmcli/providers/llamacpp.py; test -f llm/llmcli/providers/response_protocols.py; mkdir $stage"
Invoke-Executable 'ssh.exe' ($sshOptions + @($remote, $prepare))
Invoke-Executable 'scp.exe' ($sshOptions + @($server, "${remote}:llmtutor/$stage/remote_llm_server.py"))
Invoke-Executable 'scp.exe' ($sshOptions + @($provider, "${remote}:llmtutor/$stage/llamacpp.py"))

# Back up both originals before replacing either; temporary files are placed
# beside their destinations so each rename occurs on the same filesystem.
$activate = "set -eu; cd ~/llmtutor; test -s $stage/remote_llm_server.py; test -s $stage/llamacpp.py; mkdir -p backups; mkdir $backup; cp -p app/remote_llm_server.py $backup/remote_llm_server.py; cp -p llm/llmcli/providers/llamacpp.py $backup/llamacpp.py; cp $stage/remote_llm_server.py app/.remote_llm_server.$id.tmp; cp $stage/llamacpp.py llm/llmcli/providers/.llamacpp.$id.tmp; mv app/.remote_llm_server.$id.tmp app/remote_llm_server.py; mv llm/llmcli/providers/.llamacpp.$id.tmp llm/llmcli/providers/llamacpp.py"
Invoke-Executable 'ssh.exe' ($sshOptions + @($remote, $activate))

Write-Host "Remote files updated. Previous files saved at ~/llmtutor/$backup"
Write-Host 'Restart using scripts\start_all_models.py, then check http://127.0.0.1:8000/health.'
