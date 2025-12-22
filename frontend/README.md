# ArtiQuest Frontend

React + TypeScript frontend for the ArtiQuest artifact management system.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start development server:
```bash
npm run dev
```

The app will be available at `http://localhost:3000`

## Building for Production

```bash
npm run build
```

The built files will be in the `dist` directory.

## Environment Variables

Create a `.env` file in the frontend directory:

```
VITE_API_URL=http://localhost:8000
```

## Features

- User authentication with role-based access control
- Artifact gallery with search
- Artifact upload and AI analysis
- User management (admin only)
- Audit logs (admin only)
- Password change functionality

## Backend API

The frontend expects the FastAPI backend to be running on `http://localhost:8000` (or the URL specified in `VITE_API_URL`).

Make sure the backend has the following endpoints:
- `/auth/login` - User authentication
- `/api/artifacts` - Artifact CRUD operations
- `/api/analyze` - AI analysis
- `/api/users` - User management (admin)
- `/api/audit-logs` - Audit logs (admin)

