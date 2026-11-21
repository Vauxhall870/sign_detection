# Starts backend (port 8000) and frontend (port 5173) in separate windows.
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

$backendCmd = "Set-Location '$root\backend'; .\.venv\Scripts\Activate.ps1; uvicorn app.main:app --reload --port 8000"
$frontendCmd = "Set-Location '$root\frontend'; if (-not (Test-Path node_modules)) { npm install }; npm run dev"

Start-Process powershell -ArgumentList '-NoExit', '-Command', $backendCmd
Start-Process powershell -ArgumentList '-NoExit', '-Command', $frontendCmd

Start-Sleep -Seconds 5
Start-Process 'http://localhost:5173'
