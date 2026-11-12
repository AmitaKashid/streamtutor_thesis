param(
    [Parameter(Mandatory = $true)][int]$GemmaJobId,
    [Parameter(Mandatory = $true)][int]$BonsaiJobId,
    [int]$GemmaRemotePort = 8005,
    [int]$BonsaiRemotePort = 18086,
    [string]$ClusterUser = 'ovyj81af',
    [string]$ClusterLogin = 'sofja.urz.uni-magdeburg.de',
    [string]$SshKey = "$env:USERPROFILE\.ssh\llmtutor_ed25519"
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $SshKey)) { throw "SSH key not found: $SshKey" }
if ($GemmaRemotePort -lt 1 -or $GemmaRemotePort -gt 65535 -or $BonsaiRemotePort -lt 1 -or $BonsaiRemotePort -gt 65535) { throw 'Invalid remote port.' }
function Get-JobNode([int]$JobId) {
    $output = & ssh.exe -i $SshKey -o BatchMode=yes -o ConnectTimeout=15 "$ClusterUser@$ClusterLogin" "squeue -h -u $ClusterUser -j $JobId -t RUNNING -o %N"
    if ($LASTEXITCODE -ne 0) { throw "Could not query job $JobId." }
    $nodes = @($output | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    if ($nodes.Count -ne 1 -or $nodes[0] -notmatch '^[a-zA-Z0-9.-]+$') { throw "Job $JobId is not running on one node: $($nodes -join ', ')" }
    return $nodes[0]
}
$gemmaNode = Get-JobNode $GemmaJobId
$bonsaiNode = Get-JobNode $BonsaiJobId
Write-Host "Gemma: localhost:8005 -> ${gemmaNode}:${GemmaRemotePort}"
Write-Host "Bonsai: localhost:8006 -> ${bonsaiNode}:${BonsaiRemotePort}"
Write-Host 'Keep this window open; Ctrl+C closes these tunnels.'
& ssh.exe -i $SshKey -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -N `
    -L "127.0.0.1:8005:${gemmaNode}:${GemmaRemotePort}" `
    -L "127.0.0.1:8006:${bonsaiNode}:${BonsaiRemotePort}" `
    "$ClusterUser@$ClusterLogin"
if ($LASTEXITCODE -ne 0) { throw "Tunnel exited with code $LASTEXITCODE." }
