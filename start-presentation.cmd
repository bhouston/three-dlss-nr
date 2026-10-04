@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-demo.ps1" -Presentation %*
if errorlevel 1 pause
