# Ollama Troubleshooting Guide

## Understanding the Timeout Error

The error you're seeing:
```
WARNING:__main__:Request timeout on attempt 2/3
WARNING:__main__:Request timeout on attempt 3/3
ERROR:__main__:Failed to generate response: HTTPConnectionPool(host='localhost', port=11434): Read timed out. (read timeout=120)
```

This means the application tried to connect to Ollama 3 times, but each request took longer than 120 seconds (now increased to 300 seconds).

---

## Root Causes & Solutions

### 1. Ollama Service Not Running ❌

**Check if Ollama is running:**

**Docker:**
```bash
docker-compose ps
# Should show ollama container as "Up"

# If not running:
docker-compose up -d ollama
```

**Local:**
```bash
# Check if Ollama process is running
# Windows:
tasklist | findstr ollama

# Linux/Mac:
ps aux | grep ollama

# If not running, start it:
ollama serve
```

---

### 2. Model Not Downloaded ❌

**The qwen3-vl:32b model is ~20GB and must be downloaded first!**

**Check if model exists:**
```bash
# Docker:
docker exec ollama ollama list

# Local:
ollama list
```

**Download the model:**
```bash
# Docker:
docker exec ollama ollama pull qwen3-vl:32b

# Local:
ollama pull qwen3-vl:32b
```

**This will take 10-30 minutes depending on your internet speed.**

---

### 3. Wrong Endpoint Configuration ❌

**The issue:** When running in Docker, the app tries to connect to `localhost:11434`, but it should connect to `ollama:11434` (the Docker service name).

**✅ FIXED:** The latest code auto-detects the environment:
- **Docker**: Uses `http://ollama:11434`
- **Local**: Uses `http://localhost:11434`
- **Custom**: Set `OLLAMA_ENDPOINT` environment variable

**Verify the endpoint:**
```bash
# Check docker-compose.yml has:
environment:
  OLLAMA_ENDPOINT: http://ollama:11434
```

---

### 4. Model Too Large / Slow ⚠️

**qwen3-vl:32b is a 32-billion parameter model - it's HUGE!**

**Inference time:**
- CPU: 2-5 minutes per image
- GPU: 10-30 seconds per image

**Solutions:**

#### Option A: Use a Smaller Model (Recommended)
```bash
# Pull a smaller, faster model
docker exec ollama ollama pull llava:7b

# Update app.py and ai_analyzer.py:
# Change line ~30 from:
model: str = "qwen3-vl:32b"
# To:
model: str = "llava:7b"
```

**llava:7b benefits:**
- Only ~4GB download
- 5-10x faster inference
- Still provides good results

#### Option B: Use ViT Model Instead
The app includes a ViT model that doesn't require Ollama:
1. In the UI, select **"vit"** instead of **"ollama"**
2. Much faster (1-2 seconds)
3. No Ollama required
4. Less detailed analysis

---

### 5. Timeout Too Short ⏱️

**✅ FIXED:** Timeout increased from 120s to 300s (5 minutes)

If still timing out, you can increase it further:

**Edit app.py and ai_analyzer.py:**
```python
# Line ~28, change:
timeout: int = 300  # Increase to 600 (10 minutes)
```

---

### 6. Network Issues 🌐

**Docker networking problems:**

```bash
# Test if web container can reach Ollama
docker exec fll_web curl http://ollama:11434/api/tags

# Should return JSON with model list
# If it fails, restart Docker networking:
docker-compose down
docker-compose up -d
```

---

## Quick Fixes Summary

### Fix 1: Restart Everything
```bash
docker-compose down
docker-compose up -d
```

### Fix 2: Use Setup Script
```bash
# Windows:
.\setup-ollama.ps1

# Linux/Mac:
./setup-ollama.sh
```

### Fix 3: Use Smaller Model
```bash
docker exec ollama ollama pull llava:7b
# Then update app.py to use "llava:7b"
```

### Fix 4: Use ViT Instead
In the Streamlit UI, select **"vit"** model instead of **"ollama"**

---

## Verification Steps

### Step 1: Check Ollama is Running
```bash
docker-compose ps
# ollama should show "Up"
```

### Step 2: Check Model is Downloaded
```bash
docker exec ollama ollama list
# Should show qwen3-vl:32b or llava:7b
```

### Step 3: Test Ollama Directly
```bash
# Simple test
docker exec ollama ollama run llava:7b "What is 2+2?"
# Should respond with "4" or similar
```

### Step 4: Check Logs
```bash
# Ollama logs
docker-compose logs ollama

# Web app logs
docker-compose logs web
```

---

## Performance Comparison

| Model | Size | Download Time | Inference Time (CPU) | Quality |
|-------|------|---------------|---------------------|---------|
| **qwen3-vl:32b** | ~20GB | 10-30 min | 2-5 min | Excellent |
| **llava:7b** | ~4GB | 2-5 min | 20-60 sec | Good |
| **ViT (no Ollama)** | ~400MB | 1 min | 1-2 sec | Basic |

---

## Recommended Configuration

### For Production / Best Quality:
```yaml
# docker-compose.yml
ollama:
  image: ollama/ollama:latest
  deploy:
    resources:
      limits:
        memory: 16G  # qwen3-vl needs lots of RAM
```

```python
# app.py
model: str = "qwen3-vl:32b"
timeout: int = 600  # 10 minutes
```

### For Development / Fast Testing:
```python
# app.py
model: str = "llava:7b"
timeout: int = 120  # 2 minutes
```

### For Instant Results:
Use **ViT model** in the UI (no Ollama needed)

---

## Still Having Issues?

### Check System Requirements:
- **RAM**: 8GB minimum, 16GB recommended for qwen3-vl:32b
- **Disk**: 25GB free space for model
- **CPU**: Multi-core recommended
- **GPU**: Optional but 10x faster

### Enable Debug Logging:
```python
# Add to app.py at the top:
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Docker Resources:
Docker Desktop → Settings → Resources:
- **Memory**: Increase to 8GB+
- **CPU**: Allocate 4+ cores
- **Disk**: Ensure 25GB+ available

---

## Alternative: Run Ollama Locally (Not in Docker)

If Docker is causing issues, run Ollama natively:

1. **Install Ollama**: https://ollama.ai
2. **Start Ollama**: `ollama serve`
3. **Pull model**: `ollama pull llava:7b`
4. **Update .env**:
   ```env
   OLLAMA_ENDPOINT=http://localhost:11434
   ```
5. **Run app locally**:
   ```bash
   streamlit run app.py
   ```

---

## Success Indicators ✅

You'll know it's working when:
1. ✅ `docker-compose ps` shows all services "Up"
2. ✅ `docker exec ollama ollama list` shows your model
3. ✅ Web UI loads at http://localhost:8501
4. ✅ Analysis completes in 30-300 seconds (depending on model)
5. ✅ No timeout errors in logs

---

## Contact & Support

If you're still stuck:
1. Check logs: `docker-compose logs -f`
2. Verify model: `docker exec ollama ollama list`
3. Test endpoint: `curl http://localhost:11434/api/tags`
4. Try smaller model: `llava:7b`
5. Use ViT as fallback

Good luck! 🏺

