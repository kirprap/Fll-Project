import { useEffect, useState } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { artifactApi, userApi } from '../services/api'
import { Artifact, AuditLog } from '../types'
import { canAccess } from '../utils/permissions'
import './Dashboard.css'

export default function Dashboard() {
  const { user } = useAuth()
  const [artifacts, setArtifacts] = useState<Artifact[]>([])
  const [auditLogs, setAuditLogs] = useState<AuditLog[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    try {
      const [artifactsData, logsData] = await Promise.all([
        artifactApi.getAll(),
        canAccess(user?.role, 'audit-logs') ? userApi.getAuditLogs() : Promise.resolve([]),
      ])
      setArtifacts(artifactsData)
      if (canAccess(user?.role, 'audit-logs')) {
        setAuditLogs(logsData)
      }
    } catch (error) {
      console.error('Failed to load dashboard data:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return <div className="dashboard-loading">Loading...</div>
  }

  const isAdmin = canAccess(user?.role, 'audit-logs')
  const roleTitle = isAdmin ? 'Admin Dashboard' : 'User Dashboard'

  return (
    <div className="dashboard">
      <h1>📊 {roleTitle}</h1>

      {isAdmin ? (
        <>
          <div className="dashboard-stats">
            <div className="stat-card">
              <div className="stat-value">{artifacts.length}</div>
              <div className="stat-label">Total Artifacts</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{auditLogs.length}</div>
              <div className="stat-label">Recent Actions</div>
            </div>
          </div>

          <div className="dashboard-section">
            <h2>User Role Distribution</h2>
            <div className="role-distribution">
              {/* This would be better with a chart library, but showing basic stats */}
              <p>User management available in User Management section</p>
            </div>
          </div>
        </>
      ) : (
        <>
          <div className="dashboard-info">
            <h2>Welcome, {user?.name}!</h2>
            <div className="user-details">
              <p><strong>Email:</strong> {user?.email}</p>
              <p><strong>Role:</strong> {user?.role}</p>
            </div>
          </div>

          <div className="dashboard-section">
            <h2>Your Recent Activities</h2>
            {auditLogs.length > 0 ? (
              <div className="activity-list">
                {auditLogs.map((log, idx) => (
                  <div key={idx} className="activity-item">
                    <span className="activity-time">{log.timestamp}</span>
                    <span className="activity-action">{log.action}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p>No recent activities</p>
            )}
          </div>
        </>
      )}
    </div>
  )
}

