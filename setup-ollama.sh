#!/bin/bash
# Setup script for Ollama model

echo "🚀 Setting up Ollama for Archaeological Artifact Identifier"
echo "============================================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

echo "✅ Docker is running"

# Start Ollama container if not running
echo ""
echo "📦 Starting Ollama container..."
docker-compose up -d ollama

# Wait for Ollama to be ready
echo ""
echo "⏳ Waiting for Ollama to be ready..."
sleep 5

MAX_RETRIES=30
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama is ready!"
        break
    fi
    echo "   Waiting... ($((RETRY_COUNT + 1))/$MAX_RETRIES)"
    sleep 2
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "❌ Ollama failed to start. Check logs with: docker-compose logs ollama"
    exit 1
fi

# Check if model is already downloaded
echo ""
echo "🔍 Checking for qwen3-vl:32b model..."
if docker exec ollama ollama list | grep -q "qwen3-vl:32b"; then
    echo "✅ Model qwen3-vl:32b is already downloaded"
else
    echo "📥 Downloading qwen3-vl:32b model (this may take 10-30 minutes)..."
    echo "   Model size: ~20GB"
    echo ""
    docker exec ollama ollama pull qwen3-vl:32b
    
    if [ $? -eq 0 ]; then
        echo "✅ Model downloaded successfully!"
    else
        echo "❌ Failed to download model"
        echo ""
        echo "💡 Alternative: Use a smaller model"
        echo "   Run: docker exec ollama ollama pull llava:7b"
        echo "   Then update app.py to use 'llava:7b' instead"
        exit 1
    fi
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Start all services: docker-compose up -d"
echo "   2. Open browser: http://localhost:8501"
echo "   3. Upload an artifact image and analyze!"
echo ""
echo "🔧 Useful commands:"
echo "   - Check status: docker-compose ps"
echo "   - View logs: docker-compose logs -f"
echo "   - Restart: docker-compose restart"
echo "   - Stop: docker-compose down"
echo ""

