import { useState } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { userApi } from '../services/api'
import './ChangePassword.css'

export default function ChangePassword() {
  const { user } = useAuth()
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [success, setSuccess] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    if (newPassword !== confirmPassword) {
      setError('Passwords do not match')
      return
    }

    if (newPassword.length < 6) {
      setError('Password must be at least 6 characters long')
      return
    }

    if (!user?.username) {
      setError('User not authenticated. Please log in again.')
      return
    }

    setLoading(true)
    try {
      await userApi.changePassword(user.username, newPassword)
      setSuccess(true)
      setNewPassword('')
      setConfirmPassword('')
      setTimeout(() => setSuccess(false), 3000)
    } catch (err: any) {
      setError(err.message || 'Failed to update password')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="change-password">
      <h1>🔑 Change Password</h1>

      <form onSubmit={handleSubmit} className="password-form">
        <div className="form-group">
          <label htmlFor="new-password">New Password</label>
          <input
            id="new-password"
            type="password"
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
            required
            minLength={6}
          />
        </div>

        <div className="form-group">
          <label htmlFor="confirm-password">Confirm Password</label>
          <input
            id="confirm-password"
            type="password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            required
            minLength={6}
          />
        </div>

        {error && <div className="error-message">{error}</div>}
        {success && <div className="success-message">✅ Password updated successfully!</div>}

        <button type="submit" disabled={loading} className="submit-button">
          {loading ? 'Updating...' : 'Update Password'}
        </button>
      </form>
    </div>
  )
}

