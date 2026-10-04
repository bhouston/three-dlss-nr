param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 3300
)

$ErrorActionPreference = 'Stop'
$nodeExe = Join-Path $PSScriptRoot '.local\tools\node-v26.10.0-win-x64\node.exe'
$serverEntry = Join-Path $PSScriptRoot 'packages\website\.output\server\index.mjs'
$connections = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
if (!$connections) {
    Write-Host "The demo is not running on port $Port."
    return
}

foreach ($ownerId in ($connections.OwningProcess | Select-Object -Unique)) {
    $owner = Get-CimInstance Win32_Process -Filter "ProcessId = $ownerId"
    if ($owner.ExecutablePath -ne $nodeExe -or !$owner.CommandLine -or !$owner.CommandLine.Contains($serverEntry)) {
        throw "Port $Port is used by another application. That process was left untouched."
    }
    Stop-Process -Id $ownerId
    Write-Host "Stopped three-dlss-nr on port $Port."
}
