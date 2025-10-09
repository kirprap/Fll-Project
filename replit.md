# Archaeological Artifact Identifier

## Overview
A Streamlit web application designed for archaeologists to upload photos of artifacts and receive AI-powered identification, including details about name, estimated value, age/period, cultural context, and more.

## Purpose
This application helps archaeologists in the field quickly identify artifacts using advanced AI vision technology (Google Gemini), providing instant analysis without needing extensive reference materials on-site.

## Current State
- Fully functional Streamlit application
- Integrated with Google Gemini Vision API for artifact analysis
- Sample artifact database with common archaeological finds
- Session-based search history
- Mobile-friendly, professional interface

## Recent Changes
**Date: October 9, 2025**
- Initial project setup with Streamlit framework
- Created main app with photo upload interface
- Integrated Google Gemini Vision API (gemini-2.5-flash model)
- Built sample artifact database with 15+ reference items across 5 categories
- Implemented search history tracking
- Added comprehensive sidebar with usage tips and disclaimers

## Project Architecture

### Main Files
- `app.py` - Main Streamlit application with UI and workflow logic
- `ai_analyzer.py` - Google Gemini Vision API integration for artifact analysis
- `artifact_database.py` - Sample artifact database and helper functions
- `.streamlit/config.toml` - Streamlit server configuration

### Key Features
1. **Photo Upload**: Drag-and-drop interface supporting PNG, JPG, JPEG formats
2. **AI Analysis**: Uses Google Gemini Vision API to identify artifacts
3. **Detailed Results**: Provides name, value, age, description, cultural context, material, function, rarity, and confidence score
4. **Sample Database**: Reference collection of common archaeological finds organized by category (pottery, coins, stone tools, metal artifacts, decorative objects)
5. **Search History**: Session-based tracking of all analyzed artifacts
6. **Mobile-Friendly**: Responsive design suitable for field use

### Technology Stack
- **Framework**: Streamlit (Python web framework)
- **AI Service**: Google Gemini (gemini-2.5-flash model)
- **Image Processing**: Pillow (PIL)
- **Data Management**: Pandas
- **Dependencies**: google-genai SDK, streamlit, pillow, pandas

### Environment Variables
- `GEMINI_API_KEY`: Google Gemini API key for vision analysis (required)
- `SESSION_SECRET`: Session management secret

## User Preferences
- Using Google Gemini for free-tier AI vision capabilities
- Python-based implementation with Streamlit
- Focus on field archaeologist use cases
- Clean, professional interface without unnecessary styling

## Next Phase Ideas
1. Add persistent PostgreSQL database for storing identified artifacts
2. Implement visual similarity search across previously uploaded artifacts
3. Create detailed artifact profiles with historical context and references
4. Add batch upload capability for multiple photos
5. Implement expert verification workflow
6. Add export functionality (PDF reports, CSV data)
7. Multi-language support for international archaeology teams

## Notes
- Server configured to run on port 5000 with headless mode
- Free tier of Gemini API is generous and suitable for this use case
- AI identifications are estimates and should be verified by professionals
- Values shown are approximate market estimates