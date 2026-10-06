@echo off
rem Double-click to start VideoAnnotator. Runs videoannotator-start.ps1 beside
rem it; Bypass because Windows' default execution policy blocks scripts.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0videoannotator-start.ps1" %*
if "%~1"=="" if errorlevel 1 pause
