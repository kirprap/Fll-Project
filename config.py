import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL')

# AI configuration: use an optional Hugging Face token instead of Gemini
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')  # optional

# Application settings
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

# Tier configuration for UI/analysis
TIER_OPTIONS = [
    "INSTANT (1-2s)",
    "FAST (20-40s)",
    "BALANCED (30-60s)",
    "QUALITY (1-2min)",
]

TIER_KEY_MAP = {
    "INSTANT (1-2s)": "INSTANT",
    "FAST (20-40s)": "FAST",
    "BALANCED (30-60s)": "BALANCED",
    "QUALITY (1-2min)": "QUALITY",
}

TIER_INFO = {
    "INSTANT": "Uses ViT (basic classification)",
    "FAST": "Uses LLaVA 7B (good quality)",
    "BALANCED": "Uses Qwen2-VL 7B (better quality)",
    "QUALITY": "Uses Qwen3-VL (best quality)",
}

EXPECTED_TIME = {
    "INSTANT": "1-2 seconds",
    "FAST": "20-40 seconds",
    "BALANCED": "30-60 seconds",
    "QUALITY": "1-2 minutes",
}