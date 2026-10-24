Write-Host "Starting AI Lawyer KZ Backend..." -ForegroundColor Green
Set-Location $PSScriptRoot
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt
Write-Host "Starting FastAPI server on 0.0.0.0:8000..." -ForegroundColor Green
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
