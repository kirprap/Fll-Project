import { Outlet, useNavigate, useLocation } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { canAccess } from '../utils/permissions'
import './Layout.css'


export default function Layout() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()

  const getMenuItems = () => {
    if (!user) return []

    const menuItems = [
      { path: '/', label: 'Dashboard', icon: '📊', permission: null },
      { path: '/upload', label: 'Upload Artifacts', icon: '📤', permission: 'upload' },
      { path: '/gallery', label: 'Gallery Artifacts', icon: '🏺', permission: 'gallery' },
      { path: '/users', label: 'User Management', icon: '👥', permission: 'user-management' },
      { path: '/audit-logs', label: 'Audit Logs', icon: '📜', permission: 'audit-logs' },
    ]

    // Filter menu items based on user permissions
    // Admin can access everything, so all items will be shown
    return menuItems.filter(item => 
      item.permission === null || canAccess(user.role, item.permission)
    )
  }

  const menuItems = getMenuItems()

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="sidebar-header">
          <h1>🏺 ArtiQuest</h1>
          {user && (
            <div className="user-info">
              <p>Welcome, {user.name}!</p>
              <p className="user-role">{user.role}</p>
            </div>
          )}
        </div>
        <nav className="sidebar-nav">
          {menuItems.map((item) => (
            <button
              key={item.path}
              className={`nav-item ${location.pathname === item.path ? 'active' : ''}`}
              onClick={() => navigate(item.path)}
            >
              <span className="nav-icon">{item.icon}</span>
              <span className="nav-label">{item.label}</span>
            </button>
          ))}
          <button
            className="nav-item"
            onClick={() => navigate('/change-password')}
          >
            <span className="nav-icon">🔑</span>
            <span className="nav-label">Change Password</span>
          </button>
          <button className="nav-item logout" onClick={logout}>
            <span className="nav-icon">🚪</span>
            <span className="nav-label">Logout</span>
          </button>
          <footer className="sidebar-footer">
            <p> ArtiQuest AI 2025 </p>
            <p> Developed by FLL Team: MoonWalkers (63051)</p>
          </footer>
        </nav>
      </aside>
      <main className="main-content">
        <Outlet />
      </main>
    </div>
  )
}

