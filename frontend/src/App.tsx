import { Routes, Route } from 'react-router-dom';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { canAccess } from './utils/permissions';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import ArtifactGallery from './pages/ArtifactGallery';
import UploadArtifact from './pages/UploadArtifact';
import UserManagement from './pages/UserManagement';
import AuditLogs from './pages/AuditLogs';
import ChangePassword from './pages/ChangePassword';
import Layout from './components/Layout';
import { ProtectedRoute } from './components/ProtectedRoute';
import Unauthorized from './pages/Unauthorized';

function AppRoutes() {
  const { user, loading } = useAuth();

  if (loading) {
    return <div className="app-loading">Loading application...</div>;
  }

  return (
    <Routes>
      {/* Public routes */}
      <Route path="/login" element={<Login />} />
      <Route path="/unauthorized" element={<Unauthorized />} />
      
      {/* Protected routes */}
      <Route
        path="/"
        element={
          <ProtectedRoute>
            <Layout />
          </ProtectedRoute>
        }
      >
        <Route index element={<Dashboard />} />
        <Route path="gallery" element={<ArtifactGallery />} />
        <Route path="upload" element={<UploadArtifact />} />\
        
        {/* Role-based protected routes */}
        {canAccess(user?.role, 'user-management') ? (
          <Route 
            path="users" 
            element={
              <ProtectedRoute requiredRole="admin">
                <UserManagement />
              </ProtectedRoute>
            } 
          />
        ) : null}
        
        {canAccess(user?.role, 'audit-logs') ? (
          <Route 
            path="audit-logs" 
            element={
              <ProtectedRoute requiredRole="admin">
                <AuditLogs />
              </ProtectedRoute>
            } 
          />
        ) : null}
        
        <Route path="change-password" element={<ChangePassword />} />
        
        {/* 404 Not Found - Keep this at the bottom */}
        <Route path="*" element={<div>404 - Page Not Found</div>} />
      </Route>
    </Routes>
  );
}

function App() {
  return (
    <AuthProvider>
      <AppRoutes />
    </AuthProvider>
  )
}

export default App

