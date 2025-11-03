# Quick Start Guide

## Prerequisites

1. **Python 3.11+** installed
2. **Docker & Docker Compose** (for containerized deployment)
3. **Ollama** running on port 11434 with `qwen3-vl:32b` model

---

## Option 1: Run with Docker Compose (Recommended)

### Step 1: Setup Ollama and download the model

**Windows (PowerShell):**
```powershell
.\setup-ollama.ps1
```

**Linux/Mac:**
```bash
chmod +x setup-ollama.sh
./setup-ollama.sh
```

This script will:
- Start the Ollama container
- Download the `qwen3-vl:32b` model (~20GB, takes 10-30 minutes)
- Verify everything is working

**Note:** The model download is a one-time operation. Subsequent runs will be instant.

### Step 2: Start all services
```bash
docker-compose up -d
```

This will start:
- PostgreSQL database (port 5432)
- Ollama service (port 11434) with qwen3-vl:32b model
- Web application (port 8501)

### Step 3: Access the application
Open your browser and navigate to:
```
http://localhost:8501
```

### Step 4: Stop services
```bash
docker-compose down
```

---

## Option 2: Run Locally (Development)

### Step 1: Create virtual environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Set up environment variables
Create a `.env` file:
```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/artifacts
# Or for SQLite (simpler):
# DATABASE_URL=sqlite:///artifacts.db
```

### Step 4: Initialize database
```bash
python init_db.py
```

### Step 5: Start Ollama (if not running)
```bash
# Install Ollama first: https://ollama.ai
ollama serve

# In another terminal, pull the model:
ollama pull qwen3-vl:32b
```

### Step 6: Run the application
```bash
streamlit run app.py
```

### Step 7: Access the application
Open your browser and navigate to:
```
http://localhost:8501
```

---

## Using the Application

### 1. Identify Single Artifact
1. Navigate to **"Identify Artifact"** page
2. Upload an image (JPG, PNG, WEBP)
3. Select AI model:
   - **Ollama** - Full archaeological analysis (recommended)
   - **ViT** - Quick classification
   - **CLIP** - Generate embeddings for similarity search
4. Click **"Analyze Artifact"**
5. View results and optionally save to archive

### 2. Batch Processing
1. Navigate to **"Batch Processing"** page
2. Upload multiple images
3. Select AI model
4. Click **"Process All"**
5. View results for each image

### 3. View Archive
1. Navigate to **"Archive"** page
2. Browse saved artifacts in grid view
3. Click **"View Details"** for full information

### 4. Search Artifacts
1. Navigate to **"Search"** page
2. Enter keywords
3. View matching artifacts

---

## Troubleshooting

### Issue: "Request timeout" or "Connection refused" error
**Solution**: Ollama is not running or model not downloaded

**For Docker:**
```bash
# Check if Ollama is running
docker-compose ps

# Restart Ollama
docker-compose restart ollama

# Check logs
docker-compose logs ollama

# Pull the model manually
docker exec ollama ollama pull qwen3-vl:32b

# Verify model is downloaded
docker exec ollama ollama list
```

**For Local:**
```bash
# Start Ollama
ollama serve

# In another terminal, pull the model
ollama pull qwen3-vl:32b

# Verify
ollama list
```

### Issue: "Read timed out" after 120 seconds
**Solution**: The timeout has been increased to 300 seconds (5 minutes). If still timing out:
1. **Use a smaller model**: `llava:7b` instead of `qwen3-vl:32b`
2. **Check system resources**: Ensure enough RAM/CPU
3. **Try ViT model instead**: Doesn't require Ollama

To use a smaller model:
```bash
# Pull smaller model
docker exec ollama ollama pull llava:7b

# Update app.py line ~30:
# model: str = "llava:7b"
```

### Issue: Database connection error
**Solution**: Check DATABASE_URL in `.env` file or use SQLite:
```env
DATABASE_URL=sqlite:///artifacts.db
```

### Issue: Port 8501 already in use
**Solution**: Kill existing Streamlit process or use different port:
```bash
streamlit run app.py --server.port=8502
```

### Issue: Model loading takes too long
**Solution**:
- First download takes 10-30 minutes (~20GB for qwen3-vl:32b)
- Subsequent loads use cache (instant)
- Ensure good internet connection for first download
- Consider using smaller model (llava:7b is ~4GB)

### Issue: Out of memory
**Solution**:
- Use smaller model (llava:7b instead of qwen3-vl:32b)
- Use CPU-only mode (default)
- Reduce batch size
- Close other applications
- Increase Docker memory limit (Docker Desktop settings)

---

## Performance Tips

1. **Use caching**: Models are cached after first load
2. **Batch processing**: Process multiple images at once
3. **Connection pooling**: Database connections are pooled (10 + 20 overflow)
4. **Retry logic**: Network errors are automatically retried (3 attempts)

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///artifacts.db` | Database connection string |
| `HUGGINGFACE_TOKEN` | None | Optional HF token for models |
| `DEBUG` | `False` | Enable debug logging |

### Ollama Configuration

Edit `app.py` or `ai_analyzer.py` to change:
```python
OllamaClient(
    model="qwen3-vl:32b",      # Model name
    endpoint="http://localhost:11434",  # Ollama endpoint
    max_retries=3,              # Retry attempts
    timeout=120                 # Request timeout (seconds)
)
```

### Database Configuration

Edit `database.py` to change connection pool:
```python
engine = create_engine(
    _DB_URL,
    pool_size=10,        # Increase for more concurrent users
    max_overflow=20,     # Additional connections
    pool_recycle=3600,   # Connection lifetime (seconds)
)
```

---

## Development

### Run tests
```bash
python -m pytest tests/
```

### Check code quality
```bash
# Type checking
mypy app.py ai_analyzer.py database.py

# Linting
pylint app.py ai_analyzer.py database.py

# Format code
black app.py ai_analyzer.py database.py
```

### View logs
```bash
# Docker logs
docker-compose logs -f web

# Local logs
# Check terminal output
```

---

## Production Deployment

### 1. Use PostgreSQL (not SQLite)
```env
DATABASE_URL=postgresql://user:password@host:5432/dbname
```

### 2. Enable HTTPS
Configure reverse proxy (nginx, Traefik) with SSL certificates

### 3. Set resource limits
```yaml
# docker-compose.yml
services:
  web:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

### 4. Enable monitoring
- Add Prometheus metrics
- Set up health checks
- Configure logging aggregation

### 5. Backup database
```bash
# PostgreSQL backup
docker-compose exec db pg_dump -U postgres artifacts > backup.sql

# Restore
docker-compose exec -T db psql -U postgres artifacts < backup.sql
```

---

## Support

For issues or questions:
1. Check `IMPROVEMENTS.md` for detailed changes
2. Review error logs
3. Ensure all prerequisites are met
4. Verify Ollama is running and model is downloaded

---

## Next Steps

After getting the app running:
1. ✅ Upload test artifact images
2. ✅ Test batch processing
3. ✅ Verify archive functionality
4. ✅ Test search feature
5. ✅ Monitor performance
6. ✅ Set up backups (production)

Enjoy using the Archaeological Artifact Identifier! 🏺

