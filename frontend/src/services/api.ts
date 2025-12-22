import axios from 'axios'

const API_BASE_URL =  'http://localhost:8000'

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Add auth interceptor if needed
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized - clear auth and redirect
      localStorage.removeItem('user')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

// Auth endpoints (to be implemented in backend)
export const authApi = {
  login: async (username: string, password: string) => {
    // This will need to be implemented in the backend
    // For now, we'll use a mock or direct database check
    const response = await api.post('/auth/login', { username, password })
    return response.data
  },
  logout: async () => {
    await api.post('/auth/logout')
  },
}

// Artifact endpoints
export const artifactApi = {
  getAll: async () => {
    const response = await api.get('/api/artifacts')
    return response.data
  },
  getById: async (id: number) => {
    const response = await api.get(`/api/artifacts/${id}`)
    return response.data
  },
  search: async (query: string) => {
    const response = await api.get('/api/artifacts/search', { params: { q: query } })
    return response.data
  },
  create: async (artifact: any) => {
    const response = await api.post('/api/artifacts', artifact)
    return response.data
  },
  update: async (id: number, data: any) => {
    const response = await api.put(`/api/artifacts/${id}`, data)
    return response.data
  },
  verify: async (id: number, verification: { verification_status: string; reason: string; verified_by: string }) => {
    const response = await api.post(`/api/artifacts/${id}/verify`, verification)
    return response.data
  },
  analyze: async (imageData: string, tier: string = 'fast') => {
    const response = await api.post('/api/analyze', { image_data: imageData, tier })
    return response.data
  },
  batchAnalyze: async (images: string[], tier: string = 'fast') => {
    const response = await api.post('/api/analyze/batch', { images, tier })
    return response.data
  },
}

// User management endpoints (to be implemented in backend)
export const userApi = {
  getAll: async () => {
    const response = await api.get('/api/users')
    return response.data
  },
  create: async (user: any) => {
    const response = await api.post('/api/users', user)
    return response.data
  },
  getAuditLogs: async () => {
    const response = await api.get('/api/audit-logs')
    return response.data
  },
  changePassword: async (username: string, newPassword: string) => {
    const response = await api.post('/api/users/change-password', { username, new_password: newPassword })
    return response.data
  },
}