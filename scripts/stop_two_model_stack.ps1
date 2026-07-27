$ClusterUser = "ovyj81af"
$ClusterLogin = "sofja.urz.uni-magdeburg.de"

Write-Host "Stopping cluster model jobs..."
ssh "$ClusterUser@$ClusterLogin" "cd ~/llmtutor && ./scripts/stop_model_jobs.sh"

Write-Host "Now manually close the local PowerShell windows running:"
Write-Host "  SSH tunnel"
Write-Host "  uvicorn frontend_api"
Write-Host "  npm run dev"