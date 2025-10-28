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