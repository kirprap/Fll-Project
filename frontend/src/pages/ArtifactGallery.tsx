import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { artifactApi } from '../services/api'
import { Artifact } from '../types'
import ArtifactCard from '../components/ArtifactCard'
import ArtifactModal from '../components/ArtifactModal'
import './ArtifactGallery.css'

export default function ArtifactGallery() {
  const [artifacts, setArtifacts] = useState<Artifact[]>([])
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedArtifact, setSelectedArtifact] = useState<Artifact | null>(null)
  const [searchParams, setSearchParams] = useSearchParams()

  // ---- NEW: Sorting state ----
  const [sortBy, setSortBy] = useState<string>('name')
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc')

  useEffect(() => {
    const query = searchParams.get('q') || ''
    setSearchQuery(query)
    loadArtifacts(query)
  }, [searchParams, sortBy, sortOrder]) // reload when sorting changes

  const sortArtifacts = (items: Artifact[]) => {
    return [...items].sort((a, b) => {
      let comparison = 0;
      
      switch (sortBy) {
        case 'name':
          comparison = (a.name || '').localeCompare(b.name || '');
          break;
        case 'uploaded_at': {
          const parseDate = (dateStr: string | undefined): number => {
            if (!dateStr) return 0;
            // Handle both ISO strings and other formats
            const date = new Date(dateStr);
            // If date is invalid, return 0 to sort it at the beginning/end
            return isNaN(date.getTime()) ? 0 : date.getTime();
          };
          const dateA = parseDate(a.uploaded_at);
          const dateB = parseDate(b.uploaded_at);
          comparison = dateA - dateB;
          break;
        }
        case 'confidence':
          comparison = (a.confidence || 0) - (b.confidence || 0);
          break;
        case 'tier':
          comparison = (a.tier || '').localeCompare(b.tier || '');
          break;
        default:
          comparison = 0;
      }
      
      return sortOrder === 'asc' ? comparison : -comparison;
    });
  }

  const loadArtifacts = async (query: string = '') => {
    setLoading(true)
    try {
      let data

      if (query.trim()) {
        // Try search API first
        try {
          data = await artifactApi.search(query)
        } catch (searchError: any) {
          console.warn('Search failed. Falling back to client-side filtering:', searchError)
          const allArtifacts = await artifactApi.getAll()
          const q = query.toLowerCase()
          data = allArtifacts.filter((a: any) =>
            a.name?.toLowerCase().includes(q) ||
            a.description?.toLowerCase().includes(q) ||
            a.tags?.some((tag: string) => tag.toLowerCase().includes(q))
          )
        }
      } else {
        data = await artifactApi.getAll();
      }
      
      // Apply sorting to the data
      data = sortArtifacts(data);

      setArtifacts(data)
    } catch (error) {
      console.error('Failed to load artifacts:', error)
      setArtifacts([])
    } finally {
      setLoading(false)
    }
  }

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (searchQuery) {
      setSearchParams({ q: searchQuery })
    } else {
      setSearchParams({})
    }
  }

  if (loading) {
    return <div className="gallery-loading">Loading artifacts...</div>
  }

  return (
    <div className="gallery">
      <div className="gallery-header">
        <h1>🏛️ Artifact Archive</h1>

        {/* ---------- Search bar ---------- */}
        <form onSubmit={handleSearch} className="search-form">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search by name, description, material, or tags"
            className="search-input"
          />
          <button type="submit" className="search-button">
            🔎 Search
          </button>
        </form>

        {/* ---------- NEW: Sorting UI ---------- */}
        <div className="sort-controls">
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="sort-select"
          >
            <option value="name">Name</option>
            <option value="uploaded_at">Upload Date</option>
            <option value="confidence">Confidence</option>
            <option value="tier">Tier</option>
          </select>

          <select
            value={sortOrder}
            onChange={(e) => setSortOrder(e.target.value as 'asc' | 'desc')}
            className="sort-select"
          >
            <option value="asc">⬆️ Ascending</option>
            <option value="desc">⬇️ Descending</option>
          </select>
        </div>
      </div>

      {/* ---------- Stats ---------- */}
      <div className="gallery-stats">
        <div className="stat">
          <strong>Total Artifacts:</strong> {artifacts.length}
        </div>
        {searchQuery && (
          <div className="stat">
            <strong>Search Results:</strong> {artifacts.length}
          </div>
        )}
      </div>

      {/* ---------- Grid or Empty ---------- */}
      {artifacts.length === 0 ? (
        <div className="empty-state">
          <p>📭 No artifacts found. Start by uploading one!</p>
        </div>
      ) : (
        <div className="gallery-grid">
          {artifacts.map((artifact) => (
            <ArtifactCard
              key={artifact.id}
              artifact={artifact}
              onClick={() => setSelectedArtifact(artifact)}
            />
          ))}
        </div>
      )}

      {/* ---------- Modal ---------- */}
      {selectedArtifact && (
        <ArtifactModal
          artifact={selectedArtifact}
          onClose={() => setSelectedArtifact(null)}
          onUpdate={loadArtifacts}
        />
      )}
    </div>
  )
}
