import Cookies  from 'js-cookie';
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { api } from '../services/api'
import './Login.css'


export default function Login() {
  const [username, setUsername] = useState('') 
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const { login } = useAuth()
  const navigate = useNavigate()
  
  const handleLogin = () => {
    // Saves Cookies (Hopefully)
    Cookies.set(username, password, { expires: 7, path:'/' });
    console.log(username); // Shows user
    console.log(password)
    console.log("It worked")
  
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)



    
    try {
      // First, try to authenticate with the backend
      // Since we don't have a proper auth endpoint yet, we'll check credentials directly
      // This is a temporary solution - in production, use proper JWT/auth tokens
      const response = await api.post('/auth/login', { username, password })
      
      if (response.data) {
        await login(username, password)
        navigate('/')
      } else {
        setError('Invalid username or password')
      }
    } catch (err: any) {
      // If auth endpoint fails, show error - never bypass password verification
      if (err.response?.status === 401) {
        setError('Invalid username or password')
      } else if (err.response?.status === 404) {
        setError('Authentication service unavailable. Please contact administrator.')
      } else {
        setError('Login failed. Please check your credentials and try again.')
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="login-container">
      <div className="login-box">
        <h1>🏺 ArtiQuest</h1>
        <h2>Login</h2>
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="username">Username</label>
            <input
              id="username"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              autoFocus
            />
          </div>
          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>
          {error && <div className="error-message">{error}</div>}
          <button type="submit" disabled={loading} className="login-button" onClick={handleLogin}>
            {loading ? 'Logging in...' : 'Login'}
          </button>
        </form>
      </div>
    </div>
  )
}

