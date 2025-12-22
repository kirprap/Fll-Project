import { Artifact } from '../types'
import './ArtifactCard.css'

interface ArtifactCardProps {
  artifact: Artifact
  onClick: () => void
}

export default function ArtifactCard({ artifact, onClick }: ArtifactCardProps) {
  return (
    <div className="artifact-card" onClick={onClick}>
      <div className="artifact-image">
        {artifact.thumbnail || artifact.image_data ? (
          <img
            src={artifact.thumbnail || artifact.image_data}
            alt={artifact.name}
            onError={(e) => {
              e.currentTarget.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="200" height="200"%3E%3Crect fill="%23ddd" width="200" height="200"/%3E%3Ctext fill="%23999" font-family="sans-serif" font-size="14" x="50%25" y="50%25" text-anchor="middle" dy=".3em"%3ENo Image%3C/text%3E%3C/svg%3E'
            }}
          />
        ) : (
          <div className="no-image">No Image</div>
        )}
      </div>
      <div className="artifact-info">
        <h3>{artifact.name || 'Unknown'}</h3>
        {artifact.verification_status && (
          <span className={`status-badge status-${artifact.verification_status}`}>
            {artifact.verification_status}
          </span>
        )}
      </div>
    </div>
  )
}

