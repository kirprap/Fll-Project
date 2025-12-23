import { useState, useEffect, useRef } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { artifactApi } from '../services/api'
import { FormData, AnalysisResult } from '../types'
import './UploadArtifact.css'

export default function UploadArtifact() {
  const { user } = useAuth()
  const [uploadMode, setUploadMode] = useState<'single' | 'batch'>('single')
  const [imageSource, setImageSource] = useState<'upload' | 'camera'>('upload')
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [formData, setFormData] = useState<Partial<FormData>>({})
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)
  const [tier, setTier] = useState('fast')
  const [loading, setLoading] = useState(false)
  const [saved, setSaved] = useState(false)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const videoRef = useRef<HTMLVideoElement | null>(null)


interface FormData {
  length?: number
  width?: number
  thickness?: number
  weight?: number
  color?: string
  location?: string
  description?: string
  tags?: string[]
  Rome?: string 
  rome?: string 
}
  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setImageFile(file)
      const reader = new FileReader()
      reader.onloadend = () => {
        setImagePreview(reader.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' }, // Use back camera on mobile
      })
      setStream(mediaStream)
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream
      }
    } catch (error) {
      console.error('Error accessing camera:', error)
      alert('Could not access camera. Please check permissions.')
    }
  }

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop())
      setStream(null)
      if (videoRef.current) {
        videoRef.current.srcObject = null
      }
    }
  }

  const capturePhoto = () => {
    if (!videoRef.current) return

    const canvas = document.createElement('canvas')
    canvas.width = videoRef.current.videoWidth
    canvas.height = videoRef.current.videoHeight
    const ctx = canvas.getContext('2d')
    if (ctx) {
      ctx.drawImage(videoRef.current, 0, 0)
      canvas.toBlob((blob) => {
        if (blob) {
          const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' })
          setImageFile(file)
          setImagePreview(canvas.toDataURL('image/jpeg'))
          stopCamera()
          setImageSource('upload') // Switch back to upload mode after capture
        }
      }, 'image/jpeg', 0.95)
    }
  }

  // Cleanup camera stream on unmount
  useEffect(() => {
    return () => {
      stopCamera()
    }
  }, [])

  // Handle source change
  useEffect(() => {
    if (imageSource === 'camera') {
      startCamera()
    } else {
      stopCamera()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageSource])

  const convertToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.readAsDataURL(file)
      reader.onload = () => resolve(reader.result as string)
      reader.onerror = (error) => reject(error)
    })
  }

  const handleAnalyze = async () => {
    if (!imageFile) return

    setLoading(true)
    try {
      const imageData = await convertToBase64(imageFile)
      const result = await artifactApi.analyze(imageData, tier)
      setAnalysisResult(result)
    } catch (error: any) {
      alert(`Analysis failed: ${error.message}`)
    } finally {
      setLoading(false)
    }
  }
  const handleFolder = async () => {


  }

  const handleSave = async () => {
    if (!imageFile || !analysisResult) return

    setLoading(true)
    try {
      const imageData = await convertToBase64(imageFile)
      const tags = formData.tags || analysisResult.name.split(' ').filter(Boolean)
      
      await artifactApi.create({
        name: analysisResult.name,
        description: analysisResult.description,
        tags: tags,
        tier: tier,
        image_data: imageData,
         folder: formData.rome,
        form_data: Object.keys(formData).length > 0 ? formData : undefined,
        uploaded_by: user?.username,
      })

      setSaved(true)
      setTimeout(() => {
        // Reset form
        setImageFile(null)
        setImagePreview(null)
        setFormData({})
        setAnalysisResult(null)
        setSaved(false)
      }, 2000)
    } catch (error: any) {
      alert(`Failed to save artifact: ${error.message}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="upload-page">
      <h1>📤 Upload & Identify Artifact</h1>

      <div className="upload-mode-selector">
        <button
          className={uploadMode === 'single' ? 'active' : ''}
          onClick={() => setUploadMode('single')}
        >
          Single (with details)
        </button>
        <button
          className={uploadMode === 'batch' ? 'active' : ''}
          onClick={() => setUploadMode('batch')}
        >
          Batch (multiple images)
        </button>
      </div>

      {uploadMode === 'single' ? (
        <div className="upload-single">
          <div className="upload-grid">
            <div className="upload-left">
              <h2>📷 Artifact Image</h2>
              
              <div className="image-source-selector">
                <label>
                  <input
                    type="radio"
                    value="upload"
                    checked={imageSource === 'upload'}
                    onChange={(e) => setImageSource(e.target.value as 'upload' | 'camera')}
                  />
                  Upload
                </label>
                <label>
                  <input
                    type="radio"
                    value="camera"
                    checked={imageSource === 'camera'}
                    onChange={(e) => setImageSource(e.target.value as 'upload' | 'camera')}
                  />
                  Camera
                </label>
              </div>

              {imageSource === 'upload' ? (
                <>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleImageSelect}
                    className="file-input"
                  />
                  {imagePreview && (
                    <img src={imagePreview} alt="Preview" className="image-preview" />
                  )}
                </>
              ) : (
                <div className="camera-container">
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    className="camera-preview"
                  />
                  <div className="camera-controls">
                    <button onClick={capturePhoto} className="capture-button">
                      📸 Capture Photo
                    </button>
                    <button onClick={stopCamera} className="stop-camera-button">
                      Stop Camera
                    </button>
                  </div>
                  {imagePreview && (
                    <div className="captured-preview">
                      <p>Captured Image:</p>
                      <img src={imagePreview} alt="Captured" className="image-preview" />
                    </div>
                  )}
                </div>
              )}

              <h2>📋 Artifact Details</h2>
              <div className="form-grid">
                <div>
                  <label>Length (cm)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={formData.length || ''}
                    onChange={(e) =>
                      setFormData({ ...formData, length: parseFloat(e.target.value) || undefined })
                    }
                  />
                </div>
                <div>
                  <label>Width (cm)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={formData.width || ''}
                    onChange={(e) =>
                      setFormData({ ...formData, width: parseFloat(e.target.value) || undefined })
                    }
                  />
                </div>
                <div>
                  <label>Thickness (cm)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={formData.thickness || ''}
                    onChange={(e) =>
                      setFormData({
                        ...formData,
                        thickness: parseFloat(e.target.value) || undefined,
                      })
                    }
                  />
                </div>
                <div>
                  <label>Weight (g)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={formData.weight || ''}
                    onChange={(e) =>
                      setFormData({ ...formData, weight: parseFloat(e.target.value) || undefined })
                    }
                  />
                </div>
                <div>
                  <label>Color</label>
                  <input
                    type="color"
                    value={formData.color || '#808080'}
                    onChange={(e) => setFormData({ ...formData, color: e.target.value })}
                  />
                </div>
                <div>
                  <label>Location</label>
                  <input
                    type="text"
                    value={formData.location || ''}
                    onChange={(e) => setFormData({ ...formData, location: e.target.value })}
                    placeholder="e.g., Site A, Grid 5"
                  />
                </div>
              </div>
              <div>
                <label>Physical Description</label>
                <textarea
                  value={formData.description || ''}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  rows={4}
                  placeholder="Describe the artifact's appearance, condition, material, etc."
                />
              </div>
            <button
                onClick={handleFolder}
                
                className="handle-folder"
              >
                {loading ? 'Analyzing...' : 'Create'}
              </button>
              <p className='Folder-p'>
                or use a
              </p>
              <button
                onClick={handleFolder}
                
                className="handle-folder"
              >
                {loading ? 'Analyzing...' : 'Existing'}
              </button>
                 <p className='Folder-p2'>
                Folder
              </p>

            </div>

            <div className="upload-right">
              <h2>🤖 AI Analysis</h2>
              <div>
                <label>Analysis Quality</label>
                <select value={tier} onChange={(e) => setTier(e.target.value)}>
                  <option value="fast">Fast (~20-40s)</option>
                  <option value="balanced">Balanced (~30-60s)</option>
                  <option value="thorough">Thorough (~1-2 min)</option>
                </select>
              </div>

              <button
                onClick={handleAnalyze}
                disabled={!imageFile || loading}
                className="analyze-button"
              >
                {loading ? 'Analyzing...' : '🔍 Analyze Artifact'}
              </button>

              {analysisResult && (
                <div className="analysis-results">
                  <h3>Analysis Results</h3>
                  <div className="results-grid">
                    <div>
                      <strong>Confidence:</strong>{' '}
                      {(analysisResult.confidence * 100).toFixed(1)}%
                    </div>
                    <div>
                      <strong>Tier:</strong> {analysisResult.tier}
                    </div>
                    <div>
                      <strong>Method:</strong> {analysisResult.method}
                    </div>
                    <div>
                      <strong>Time:</strong> {analysisResult.analysis_time}
                    </div>
                  </div>
                  <div>
                    <strong>Name:</strong> {analysisResult.name}
                  </div>
                  <div>
                    <strong>Description:</strong> {analysisResult.description}
                  </div>

                  <button
                    onClick={handleSave}
                    disabled={loading || saved}
                    className="save-button"
                  >
                    {saved ? '✅ Saved!' : '💾 Save to Archive'}
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      ) : (
        <div className="upload-batch">
          <p>Batch upload functionality coming soon...</p>
        </div>
      )}
    </div>
  )
}

