import { useEffect, useState } from 'react'
import { userApi } from '../services/api'
import { AuditLog } from '../types'
import './AuditLogs.css'

export default function AuditLogs() {
  const [logs, setLogs] = useState<AuditLog[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadLogs()
  }, [])

  const loadLogs = async () => {
    try {
      const data = await userApi.getAuditLogs()
      setLogs(data)
    } catch (error) {
      console.error('Failed to load audit logs:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return <div className="loading">Loading audit logs...</div>
  }

  return (
    <div className="audit-logs">
      <h1>📜 Audit Logs (Last 50 Actions)</h1>

      <div className="logs-section">
        {logs.length === 0 ? (
          <p>No audit logs found</p>
        ) : (
          <table className="logs-table">
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Username</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {logs.map((log, idx) => (
                <tr key={idx}>
                  <td>{log.timestamp}</td>
                  <td>{log.username}</td>
                  <td>{log.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

