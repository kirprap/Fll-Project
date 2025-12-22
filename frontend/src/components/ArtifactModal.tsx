import { useState, useEffect } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { artifactApi } from '../services/api'
import { Artifact, FormData } from '../types'
import { canAccess } from '../utils/permissions'
import './ArtifactModal.css'

interface ArtifactModalProps {
  artifact: Artifact
  onClose: () => void
  onUpdate: () => void
}

export default function ArtifactModal({ artifact, onClose, onUpdate }: ArtifactModalProps) {
  const { user } = useAuth()
  const [fullArtifact, setFullArtifact] = useState<Artifact>(artifact)
  const [editMode, setEditMode] = useState(false)
  const [loading, setLoading] = useState(false)
  const [verificationReason, setVerificationReason] = useState('')
  
  // Editable fields state
  const [editName, setEditName] = useState('')
  const [editDescription, setEditDescription] = useState('')
  const [editTags, setEditTags] = useState<string[]>([])

  useEffect(() => {
    loadFullArtifact()
  }, [artifact.id])

  // Initialize edit fields when artifact loads
  useEffect(() => {
    if (fullArtifact) {
      setEditName(fullArtifact.name || '')
      setEditDescription(fullArtifact.description || '')
      setEditTags(fullArtifact.tags || [])
    }
  }, [fullArtifact])

  const loadFullArtifact = async () => {
    try {
      const data = await artifactApi.getById(artifact.id)
      setFullArtifact(data)
    } catch (error) {
      console.error('Failed to load artifact details:', error)
    }
  }

  const handleSave = async () => {
    setLoading(true)
    try {
      await artifactApi.update(artifact.id, {
        name: editName.trim(),
        description: editDescription.trim() || null,
        tags: editTags.join(','),
      })
      await loadFullArtifact()
      setEditMode(false)
      onUpdate()
      alert('Artifact updated successfully!')
    } catch (error: any) {
      alert(`Failed to update artifact: ${error.message}`)
    } finally {
      setLoading(false)
    }
  }

  const handleVerify = async (status: 'verified' | 'rejected') => {
    if (!verificationReason.trim()) {
      alert('Please provide a reason for your decision')
      return
    }

    setLoading(true)
    try {
      await artifactApi.verify(artifact.id, {
        verification_status: status,
        reason: verificationReason.trim(),
        verified_by: user?.username || 'unknown',
      })
      await loadFullArtifact()
      onUpdate()
      setVerificationReason('')
      alert(`Artifact ${status === 'verified' ? 'approved' : 'rejected'} successfully!`)
    } catch (error: any) {
      alert(`Failed to ${status} artifact: ${error.message}`)
    } finally {
      setLoading(false)
    }
  }

  const formData: FormData | null = fullArtifact.form_data
    ? (typeof fullArtifact.form_data === 'string'
        ? JSON.parse(fullArtifact.form_data)
        : fullArtifact.form_data)
    : null

  // Admin has all permissions, including edit and verify
  const canEdit = canAccess(user?.role, 'edit')
  const canVerify = canAccess(user?.role, 'verify')

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>{fullArtifact.name}</h2>
          <div className="modal-actions">
            {canEdit && (
              <button
                className="edit-button"
                onClick={() => {
                  if (editMode) {
                    handleSave()
                  } else {
                    setEditMode(true)
                  }
                }}
                disabled={loading}
              >
                {editMode ? '💾 Save' : '✏️ Edit'}
              </button>
            )}
            {canEdit && editMode && (
              <button
                className="cancel-button"
                onClick={() => {
                  setEditMode(false)
                  // Reset to original values
                  setEditName(fullArtifact.name || '')
                  setEditDescription(fullArtifact.description || '')
                  setEditTags(fullArtifact.tags || [])
                }}
                disabled={loading}
              >
                ✕ Cancel
              </button>
            )}
            <button className="close-button" onClick={onClose}>
              ✕
            </button>
          </div>
        </div>

        <div className="modal-body">
          <div className="modal-left">
            {fullArtifact.image_data && (
              <img
                src={fullArtifact.image_data}
                alt={fullArtifact.name}
                className="artifact-image-full"
              />
            )}
            {fullArtifact.has_3d_model && (
              <div className="model-3d-section">
                <h3>🎯 3D Model</h3>
                <p>3D model available ({fullArtifact.model_3d_format?.toUpperCase()} format)</p>
                {fullArtifact.model_3d_data && (
                  <a
                    href={fullArtifact.model_3d_data}
                    download={`artifact_${fullArtifact.id}.${fullArtifact.model_3d_format}`}
                    className="download-button"
                  >
                    📥 Download 3D Model
                  </a>
                )}
              </div>
            )}
          </div>

          <div className="modal-right">
            <div className="info-section">
              <h3>Basic Information</h3>
              <p><strong>ID:</strong> {fullArtifact.id}</p>
              <p><strong>Tier:</strong> {fullArtifact.tier || 'N/A'}</p>
              <p><strong>Uploaded:</strong> {fullArtifact.uploaded_at || 'N/A'}</p>
            </div>

            <div className="info-section">
              <h3>Name</h3>
              {editMode ? (
                <input
                  type="text"
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  className="edit-input"
                  placeholder="Artifact name"
                />
              ) : (
                <p>{fullArtifact.name}</p>
              )}
            </div>

            <div className="info-section">
              <h3>Description</h3>
              {editMode ? (
                <textarea
                  value={editDescription}
                  onChange={(e) => setEditDescription(e.target.value)}
                  className="edit-textarea"
                  placeholder="Artifact description"
                  rows={4}
                />
              ) : (
                <p>{fullArtifact.description || 'No description'}</p>
              )}
            </div>

            <div className="info-section">
              <h3>Tags</h3>
              {editMode ? (
                <div>
                  <input
                    type="text"
                    value={editTags.join(', ')}
                    onChange={(e) => {
                      const tagsStr = e.target.value
                      setEditTags(tagsStr.split(',').map(t => t.trim()).filter(t => t.length > 0))
                    }}
                    className="edit-input"
                    placeholder="Comma-separated tags (e.g., pottery, ancient, ceramic)"
                  />
                  <div className="tags-list" style={{ marginTop: '8px' }}>
                    {editTags.map((tag, idx) => (
                      <span key={idx} className="tag">🏷️ {tag}</span>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="tags-list">
                  {fullArtifact.tags && fullArtifact.tags.length > 0 ? (
                    fullArtifact.tags.map((tag, idx) => (
                      <span key={idx} className="tag">🏷️ {tag}</span>
                    ))
                  ) : (
                    <p>No tags</p>
                  )}
                </div>
              )}
            </div>

            <div className="info-section">
              <h3>Verification Status</h3>
              <span className={`status-badge status-${fullArtifact.verification_status || 'pending'}`}>
                {fullArtifact.verification_status || 'pending'}
              </span>
              {fullArtifact.verified_by && (
                <p><strong>Verified by:</strong> {fullArtifact.verified_by}</p>
              )}
            </div>

            {formData && (
              <div className="info-section">
                <h3>📐 Physical Measurements & Details</h3>
                <div className="measurements-grid">
                  {formData.length && (
                    <div className="measurement">
                      <strong>Length:</strong> {formData.length} cm
                    </div>
                  )}
                  {formData.width && (
                    <div className="measurement">
                      <strong>Width:</strong> {formData.width} cm
                    </div>
                  )}
                  {formData.thickness && (
                    <div className="measurement">
                      <strong>Thickness:</strong> {formData.thickness} cm
                    </div>
                  )}
                  {formData.weight && (
                    <div className="measurement">
                      <strong>Weight:</strong> {formData.weight} g
                    </div>
                  )}
                  {formData.color && (
                    <div className="measurement">
                      <strong>Color:</strong> {formData.color}
                    </div>
                  )}
                  {formData.location && (
                    <div className="measurement">
                      <strong>Location:</strong> {formData.location}
                    </div>
                  )}
                </div>
                {formData.description && (
                  <p><strong>Physical Description:</strong> {formData.description}</p>
                )}
              </div>
            )}
          </div>
        </div>

        {canVerify && (
          <div className="modal-footer">
            <h3>🔐 Verification Actions</h3>
            {fullArtifact.uploaded_by && (
              <p>📤 Uploaded by: <strong>{fullArtifact.uploaded_by}</strong></p>
            )}
            <textarea
              value={verificationReason}
              onChange={(e) => setVerificationReason(e.target.value)}
              placeholder="Please provide a detailed reason for your decision. This will be sent to the uploader."
              className="reason-input"
              rows={4}
            />
            <div className="verification-buttons">
              <button
                className="approve-button"
                onClick={() => handleVerify('verified')}
                disabled={loading || !verificationReason.trim()}
              >
                ✅ Approve
              </button>
              <button
                className="reject-button"
                onClick={() => handleVerify('rejected')}
                disabled={loading || !verificationReason.trim()}
              >
                ❌ Reject
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

