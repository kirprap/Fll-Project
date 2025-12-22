export type UserRole = 'admin' | 'user' | 'field' | 'onsite'

export interface User {
  username: string
  name: string
  email: string
  role: UserRole
}

export interface Artifact {
  id: number
  name: string
  description?: string
  tags: string[]
  tier: string
  thumbnail?: string
  image_data?: string
  uploaded_at?: string
  uploaded_by?: string
  analyzed_at?: string
  confidence?: number
  form_data?: FormData
  verification_status?: 'pending' | 'verified' | 'rejected'
  verified_by?: string
  verified_at?: string
  has_3d_model?: boolean
  model_3d_data?: string
  model_3d_format?: string
}

export interface FormData {
  length?: number
  width?: number
  thickness?: number
  weight?: number
  color?: string
  location?: string
  description?: string
  artifact_name?: string
  tags?: string[]
}

export interface AnalysisResult {
  name: string
  description: string
  confidence: number
  method: string
  tier: string
  analysis_time: string
  embedding?: number[]
}

export interface AuditLog {
  timestamp: string
  username: string
  action: string
}

