param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 3300,
    [switch]$NoBrowser,
    [switch]$Presentation
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$nodeExe = Join-Path $projectRoot '.local\tools\node-v26.10.0-win-x64\node.exe'
$serverEntry = Join-Path $projectRoot 'packages\website\.output\server\index.mjs'
$url = if ($Presentation) { "http://localhost:$Port/" } else { "http://localhost:$Port/local-demo.html" }

function Open-DemoBrowser {
    $browserCandidates = @(
        'C:\Program Files\Google\Chrome\Application\chrome.exe',
        'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
        'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
    )
    $browserExe = $browserCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if ($browserExe) { Start-Process -FilePath $browserExe -ArgumentList $url }
    else { Start-Process $url }
}

if (!(Test-Path -LiteralPath $nodeExe)) {
    throw 'The project-local Node.js 26 runtime is missing. See INDITAS.md.'
}
if (!(Test-Path -LiteralPath $serverEntry)) {
    throw 'The website build is missing. Run .\pnpm-local.ps1 build first.'
}

$listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
if ($listener) {
    $ownsPort = $false
    foreach ($connection in $listener) {
        $owner = Get-CimInstance Win32_Process -Filter "ProcessId = $($connection.OwningProcess)"
        if ($owner.ExecutablePath -eq $nodeExe -and $owner.CommandLine -and $owner.CommandLine.Contains($serverEntry)) {
            $ownsPort = $true
        }
    }
    if (!$ownsPort) {
        throw "Port $Port is used by another application. Run .\start-demo.ps1 -Port 3301."
    }
    $response = Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 10
    if ($response.StatusCode -ne 200 -or $response.Content -notmatch 'three-dlss-nr') {
        throw 'The existing demo server did not return the expected page.'
    }
    Write-Host "The demo is already running: $url"
    if (!$NoBrowser) { Open-DemoBrowser }
    return
}

$previousHost = $env:NITRO_HOST
$previousPort = $env:NITRO_PORT
Push-Location $projectRoot
try {
    $env:NITRO_HOST = '127.0.0.1'
    $env:NITRO_PORT = "$Port"
    Write-Host "three-dlss-nr: $url"
    Write-Host 'Use Chrome or Edge with hardware acceleration. Ctrl+C stops the server.'
    if (!$NoBrowser) { Open-DemoBrowser }
    & $nodeExe $serverEntry
    if ($LASTEXITCODE -notin @(0, -1, -1073741510)) {
        throw "The demo server exited with code $LASTEXITCODE."
    }
} finally {
    $env:NITRO_HOST = $previousHost
    $env:NITRO_PORT = $previousPort
    Pop-Location
}
