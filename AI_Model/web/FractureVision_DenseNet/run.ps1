# FractureVision DenseNet Launcher
Write-Host "🚀 Starting FractureVision DenseNet AI..." -ForegroundColor Cyan

# Check for model
$modelPath = "../best_fracturenet.keras"
if (-not (Test-Path $modelPath)) {
    Write-Host "⚠️ Warning: 'best_fracturenet.keras' not found in the root directory." -ForegroundColor Yellow
    Write-Host "Please ensure you have trained the model and saved it as 'best_fracturenet.keras'." -ForegroundColor Gray
}

# Install requirements if needed
# Write-Host "📦 Checking dependencies..." -ForegroundColor Gray
# pip install -r requirements.txt

# Run server
cd backend
python main.py
