# PowerShell setup script for Ollama model

Write-Host "🚀 Setting up Ollama for Archaeological Artifact Identifier" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Check if Docker is running
try {
    docker info | Out-Null
    Write-Host "✅ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Start Ollama container if not running
Write-Host ""
Write-Host "📦 Starting Ollama container..." -ForegroundColor Yellow
docker-compose up -d ollama

# Wait for Ollama to be ready
Write-Host ""
Write-Host "⏳ Waiting for Ollama to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

$maxRetries = 30
$retryCount = 0
$ollamaReady = $false

while ($retryCount -lt $maxRetries) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Host "✅ Ollama is ready!" -ForegroundColor Green
            $ollamaReady = $true
            break
        }
    } catch {
        # Continue waiting
    }
    Write-Host "   Waiting... ($($retryCount + 1)/$maxRetries)" -ForegroundColor Gray
    Start-Sleep -Seconds 2
    $retryCount++
}

if (-not $ollamaReady) {
    Write-Host "❌ Ollama failed to start. Check logs with: docker-compose logs ollama" -ForegroundColor Red
    exit 1
}

# Check if model is already downloaded
Write-Host ""
Write-Host "🔍 Checking for qwen3-vl:32b model..." -ForegroundColor Yellow
$modelList = docker exec ollama ollama list

if ($modelList -match "qwen3-vl:32b") {
    Write-Host "✅ Model qwen3-vl:32b is already downloaded" -ForegroundColor Green
} else {
    Write-Host "📥 Downloading qwen3-vl:32b model (this may take 10-30 minutes)..." -ForegroundColor Yellow
    Write-Host "   Model size: ~20GB" -ForegroundColor Gray
    Write-Host ""
    
    docker exec ollama ollama pull qwen3-vl:32b
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Model downloaded successfully!" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to download model" -ForegroundColor Red
        Write-Host ""
        Write-Host "💡 Alternative: Use a smaller model" -ForegroundColor Yellow
        Write-Host "   Run: docker exec ollama ollama pull llava:7b" -ForegroundColor Cyan
        Write-Host "   Then update app.py to use 'llava:7b' instead" -ForegroundColor Cyan
        exit 1
    }
}

Write-Host ""
Write-Host "🎉 Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Next steps:" -ForegroundColor Cyan
Write-Host "   1. Start all services: docker-compose up -d" -ForegroundColor White
Write-Host "   2. Open browser: http://localhost:8501" -ForegroundColor White
Write-Host "   3. Upload an artifact image and analyze!" -ForegroundColor White
Write-Host ""
Write-Host "🔧 Useful commands:" -ForegroundColor Cyan
Write-Host "   - Check status: docker-compose ps" -ForegroundColor White
Write-Host "   - View logs: docker-compose logs -f" -ForegroundColor White
Write-Host "   - Restart: docker-compose restart" -ForegroundColor White
Write-Host "   - Stop: docker-compose down" -ForegroundColor White
Write-Host ""

