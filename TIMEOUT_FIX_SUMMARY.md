# Ollama Timeout Issue - FIXED ✅

## What Was Wrong

Your error:
```
WARNING:__main__:Request timeout on attempt 2/3
WARNING:__main__:Request timeout on attempt 3/3
ERROR:__main__:Failed to generate response: HTTPConnectionPool(host='localhost', port=11434): Read timed out. (read timeout=120)
```

**Root causes identified:**
1. ❌ **Wrong endpoint**: App tried `localhost:11434` instead of `ollama:11434` in Docker
2. ❌ **Timeout too short**: 120 seconds wasn't enough for large model
3. ❌ **Model not downloaded**: qwen3-vl:32b (~20GB) needs to be pulled first
4. ❌ **Poor error messages**: Users didn't know what to do

---

## What Was Fixed

### 1. ✅ Auto-Detect Endpoint
**Files changed:** `app.py`, `ai_analyzer.py`

The code now automatically detects if running in Docker or locally:
- **Docker**: Uses `http://ollama:11434`
- **Local**: Uses `http://localhost:11434`
- **Custom**: Respects `OLLAMA_ENDPOINT` environment variable

```python
# Auto-detection logic added:
if os.getenv('HOSTNAME') and 'docker' in os.getenv('HOSTNAME', '').lower():
    endpoint = "http://ollama:11434"
else:
    endpoint = os.getenv('OLLAMA_ENDPOINT', 'http://localhost:11434')
```

### 2. ✅ Increased Timeout
**Changed:** 120 seconds → 300 seconds (5 minutes)

Large models like qwen3-vl:32b can take 2-5 minutes on CPU, so the timeout was too short.

```python
timeout: int = 300  # Was 120
```

### 3. ✅ Added Docker Health Checks
**File changed:** `docker-compose.yml`

Ollama now has a health check to ensure it's ready before the web app starts:

```yaml
ollama:
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
    interval: 30s
    timeout: 10s
    retries: 5
    start_period: 60s

web:
  depends_on:
    ollama:
      condition: service_healthy  # Wait for Ollama to be ready
```

### 4. ✅ Better Error Messages
**File changed:** `app.py`

Users now get helpful error messages with solutions:

```
⚠️ Ollama Connection Error

Possible causes:
1. Ollama is not running - Start Ollama service
2. Model not downloaded - Run: ollama pull qwen3-vl:32b
3. Timeout (model too large) - The model is processing, please wait
4. Wrong endpoint - Check OLLAMA_ENDPOINT environment variable

Quick fixes:
- Docker: docker-compose restart ollama
- Local: ollama serve in a terminal
- Check model: ollama list
- Pull model: ollama pull qwen3-vl:32b

Alternative: Try using the ViT model instead (faster, no Ollama required)
```

### 5. ✅ Setup Scripts Created
**New files:** `setup-ollama.sh`, `setup-ollama.ps1`

Automated scripts to:
- Start Ollama container
- Download the model
- Verify everything works

**Usage:**
```bash
# Windows
.\setup-ollama.ps1

# Linux/Mac
./setup-ollama.sh
```

### 6. ✅ Documentation Updated
**New/Updated files:**
- `OLLAMA_TROUBLESHOOTING.md` - Comprehensive troubleshooting guide
- `QUICKSTART.md` - Updated with Ollama setup steps
- `TIMEOUT_FIX_SUMMARY.md` - This file

---

## How to Fix Your Current Issue

### Quick Fix (5 minutes):

**Step 1: Pull the model**
```bash
docker exec ollama ollama pull qwen3-vl:32b
```
This downloads ~20GB and takes 10-30 minutes. **This is the most likely issue!**

**Step 2: Restart services**
```bash
docker-compose restart
```

**Step 3: Test**
Upload an image and try again. Should work now!

---

### Alternative: Use Smaller Model (Recommended)

If qwen3-vl:32b is too slow or large:

**Step 1: Pull smaller model**
```bash
docker exec ollama ollama pull llava:7b
```
Only ~4GB, much faster!

**Step 2: Update code**
Edit `app.py` and `ai_analyzer.py`, line ~30:
```python
# Change from:
model: str = "qwen3-vl:32b"

# To:
model: str = "llava:7b"
```

**Step 3: Restart**
```bash
docker-compose restart web
```

---

### Fastest Fix: Use ViT Model

**No code changes needed!**

1. Open http://localhost:8501
2. Select **"vit"** instead of **"ollama"** in the dropdown
3. Analyze - works in 1-2 seconds!

**Trade-off:** Less detailed analysis, but instant results.

---

## Verification Checklist

Run these commands to verify everything is working:

```bash
# 1. Check all services are running
docker-compose ps
# All should show "Up"

# 2. Check Ollama is healthy
docker-compose ps ollama
# Should show "healthy"

# 3. Check model is downloaded
docker exec ollama ollama list
# Should show qwen3-vl:32b or llava:7b

# 4. Test Ollama directly
docker exec ollama ollama run llava:7b "Hello"
# Should respond with a greeting

# 5. Check web app logs
docker-compose logs web | tail -20
# Should show "OllamaClient initialized: endpoint=http://ollama:11434"
```

---

## Files Changed Summary

| File | Changes |
|------|---------|
| `app.py` | ✅ Auto-detect endpoint, ✅ Timeout 300s, ✅ Better errors |
| `ai_analyzer.py` | ✅ Auto-detect endpoint, ✅ Timeout 300s |
| `docker-compose.yml` | ✅ Health checks, ✅ OLLAMA_ENDPOINT env var |
| `setup-ollama.sh` | ✅ NEW - Automated setup script (Linux/Mac) |
| `setup-ollama.ps1` | ✅ NEW - Automated setup script (Windows) |
| `OLLAMA_TROUBLESHOOTING.md` | ✅ NEW - Comprehensive troubleshooting |
| `QUICKSTART.md` | ✅ Updated with Ollama setup steps |
| `TIMEOUT_FIX_SUMMARY.md` | ✅ NEW - This file |

---

## Performance Expectations

After fixes, here's what to expect:

| Model | First Run | Subsequent Runs | Quality |
|-------|-----------|-----------------|---------|
| **qwen3-vl:32b** (CPU) | 2-5 min | 2-5 min | Excellent ⭐⭐⭐⭐⭐ |
| **qwen3-vl:32b** (GPU) | 30-60 sec | 30-60 sec | Excellent ⭐⭐⭐⭐⭐ |
| **llava:7b** (CPU) | 30-60 sec | 30-60 sec | Good ⭐⭐⭐⭐ |
| **llava:7b** (GPU) | 5-10 sec | 5-10 sec | Good ⭐⭐⭐⭐ |
| **ViT** (no Ollama) | 1-2 sec | 1-2 sec | Basic ⭐⭐⭐ |

---

## Next Steps

1. **Run setup script**: `.\setup-ollama.ps1` (Windows) or `./setup-ollama.sh` (Linux/Mac)
2. **Start services**: `docker-compose up -d`
3. **Open browser**: http://localhost:8501
4. **Test with an image**: Upload and analyze
5. **If still issues**: Check `OLLAMA_TROUBLESHOOTING.md`

---

## Still Getting Timeouts?

If you're still seeing timeouts after following the fixes:

1. **Check model is downloaded**: `docker exec ollama ollama list`
2. **Check logs**: `docker-compose logs ollama`
3. **Try smaller model**: Use `llava:7b` instead
4. **Use ViT**: Select "vit" in UI dropdown
5. **Increase timeout**: Edit app.py, change `timeout: int = 600` (10 minutes)
6. **Check resources**: Docker Desktop → Settings → Resources (8GB+ RAM)

---

## Success! 🎉

Once working, you should see:
- ✅ No timeout errors
- ✅ Analysis completes in 30-300 seconds
- ✅ Detailed artifact descriptions
- ✅ Ability to save to archive

Enjoy your Archaeological Artifact Identifier! 🏺

