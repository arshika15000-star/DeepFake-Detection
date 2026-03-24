# DeepTruth Startup Script
Write-Host "Starting DeepTruth AI Backend..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python app.py"

Write-Host "Starting DeepTruth AI Frontend..." -ForegroundColor Magenta
cd frontend
npm run dev
