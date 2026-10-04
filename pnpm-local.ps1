$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$nodeDirectory = Join-Path $projectRoot '.local\tools\node-v26.10.0-win-x64'
$nodeExe = Join-Path $nodeDirectory 'node.exe'
$pnpmEntry = Join-Path $projectRoot '.local\tools\pnpm\bin\pnpm.cjs'
$previousPath = $env:PATH
Push-Location $projectRoot
try {
    $env:PATH = $nodeDirectory + ';' + $previousPath
    & $nodeExe $pnpmEntry @args
    $commandExitCode = $LASTEXITCODE
} finally {
    $env:PATH = $previousPath
    Pop-Location
}
exit $commandExitCode
