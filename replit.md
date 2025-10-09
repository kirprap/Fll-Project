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
**Date: October 9, 2025 - Phase 2 Complete**
- Added PostgreSQL database with complete schema for persistent artifact storage
- Implemented database persistence layer with save/retrieve functionality
- Built searchable archive with text-based search across artifacts
- Added batch upload capability for processing multiple artifacts simultaneously
- Implemented visual similarity search using AI comparison
- Created detailed artifact profile management with provenance, historical context, and references
- Built expert verification workflow with status tracking (pending/verified/rejected)
- Organized app into 5 tabs: Identify, Batch Upload, Archive, Expert Verification, Statistics

**Date: October 9, 2025 - Initial Setup**
- Initial project setup with Streamlit framework
- Created main app with photo upload interface
- Integrated Google Gemini Vision API (gemini-2.5-flash model)
- Built sample artifact database with 15+ reference items across 5 categories
- Implemented search history tracking
- Added comprehensive sidebar with usage tips and disclaimers

## Project Architecture

### Main Files
- `app.py` - Main Streamlit application with UI and workflow logic (5 tabs)
- `ai_analyzer.py` - Google Gemini Vision API integration for artifact analysis and similarity comparison
- `artifact_database.py` - Sample artifact reference database and helper functions
- `database.py` - PostgreSQL ORM models and database operations using SQLAlchemy
- `init_db.py` - Database initialization script
- `.streamlit/config.toml` - Streamlit server configuration

### Key Features
1. **Photo Upload**: Drag-and-drop interface supporting PNG, JPG, JPEG formats
2. **AI Analysis**: Uses Google Gemini Vision API to identify artifacts
3. **Detailed Results**: Provides name, value, age, description, cultural context, material, function, rarity, and confidence score
4. **Batch Processing**: Upload and process multiple artifacts simultaneously with progress tracking
5. **Persistent Storage**: All identified artifacts saved to PostgreSQL database
6. **Searchable Archive**: Search artifacts by name, material, description, cultural context
7. **Visual Similarity Search**: AI-powered comparison to find similar artifacts in archive
8. **Expert Verification**: Review queue with pending/verified/rejected status tracking
9. **Artifact Profiles**: Add detailed provenance, historical context, and scholarly references
10. **Sample Database**: Reference collection of common archaeological finds organized by category
11. **Session History**: Track current session's identifications
12. **Statistics Dashboard**: View total artifacts, recent identifications, and archive metrics
13. **Mobile-Friendly**: Responsive design suitable for field use

### Technology Stack
- **Framework**: Streamlit (Python web framework)
- **AI Service**: Google Gemini (gemini-2.5-flash model)
- **Image Processing**: Pillow (PIL)
- **Data Management**: Pandas
- **Dependencies**: google-genai SDK, streamlit, pillow, pandas

### Environment Variables
- `GEMINI_API_KEY`: Google Gemini API key for vision analysis (required)
- `SESSION_SECRET`: Session management secret
- `DATABASE_URL`: PostgreSQL database connection string (auto-configured)
- `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `PGDATABASE`: PostgreSQL credentials (auto-configured)

## User Preferences
- Using Google Gemini for free-tier AI vision capabilities
- Python-based implementation with Streamlit
- Focus on field archaeologist use cases
- Clean, professional interface without unnecessary styling

## Completed Features (Phase 2)
1. ✅ PostgreSQL database for persistent artifact storage
2. ✅ Visual similarity search using AI comparison
3. ✅ Detailed artifact profiles with provenance, historical context, and references
4. ✅ Batch upload capability for multiple photos with progress tracking
5. ✅ Expert verification workflow with status tracking and comments
6. ✅ Searchable archive with filter capabilities

## Future Enhancement Ideas
1. Export functionality (PDF reports, CSV data, database backups)
2. Multi-language support for international archaeology teams
3. User authentication and multi-user support
4. Advanced filtering (date ranges, value ranges, materials)
5. Image gallery view mode for archive
6. Automated tagging and categorization
7. Integration with external archaeological databases
8. Mobile app version

## Notes
- Server configured to run on port 5000 with headless mode
- Free tier of Gemini API is generous and suitable for this use case
- AI identifications are estimates and should be verified by professionals
- Values shown are approximate market estimates