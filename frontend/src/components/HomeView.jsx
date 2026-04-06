import { useState, useRef } from 'react'
import { uploadGame } from '../api'
import GameHistory from './GameHistory'

export default function HomeView({ games, onSubmit, onOpenGame, onDeleteGame }) {
  const [label, setLabel] = useState('')
  const [file, setFile] = useState(null)
  const [homographyMode, setHomographyMode] = useState('auto')
  const [loading, setLoading] = useState(false)
  const [uploadPct, setUploadPct] = useState(0)
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)
  const fileRef = useRef()

  const handleDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    const dropped = e.dataTransfer.files[0]
    if (dropped && dropped.type.startsWith('video/')) {
      setFile(dropped)
      setError('')
    } else if (dropped) {
      setError('Please drop a video file')
    }
  }

  const handleDragOver = (e) => { e.preventDefault(); setDragging(true) }
  const handleDragLeave = (e) => { e.preventDefault(); setDragging(false) }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')

    if (!file) {
      setError('Select a video file')
      return
    }
    setLoading(true)
    setUploadPct(0)
    try {
      const response = await uploadGame(
        file,
        label.trim(),
        setUploadPct,
        false,
        homographyMode
      )
      onSubmit({
        jobId: response.job_id,
        source: file.name,
        label: label.trim(),
        status: response.needs_calibration ? 'awaiting_calibration' : 'processing',
        homographyMode,
      })
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="home-view">
      <div className="hero fade-in">
        <div className="hero-badge fade-in-delay-1">
          <span className="pulse" />
          CV-Powered Tactical Analysis
        </div>
        <h1 className="fade-in-delay-2">
          Break down any<br />
          <span>water polo game</span>
        </h1>
        <p className="fade-in-delay-3">
          Upload a game video. Our CV pipeline detects players, tracks formations,
          and generates tactical analysis — powered by AI.
        </p>

        <form className="submit-form fade-in-delay-4" onSubmit={handleSubmit}>
          <div className="mode-toggle homography-toggle">
            <button
              type="button"
              className={homographyMode === 'auto' ? 'active' : ''}
              onClick={() => setHomographyMode('auto')}
            >
              Auto Homography
            </button>
            <button
              type="button"
              className={homographyMode === 'manual' ? 'active' : ''}
              onClick={() => setHomographyMode('manual')}
            >
              Manual Homography
            </button>
          </div>
          <div
            className={`file-drop-zone ${dragging ? 'dragging' : ''} ${file ? 'has-file' : ''} ${error ? 'error' : ''}`}
            onClick={() => fileRef.current.click()}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
          >
            <input
              ref={fileRef}
              type="file"
              accept="video/*"
              style={{ display: 'none' }}
              onChange={(e) => { setFile(e.target.files[0] || null); setError(''); }}
            />
            <div className="file-drop-icon">
              {file ? (
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
              ) : (
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="2" y="3" width="20" height="14" rx="2" />
                  <path d="m10 8 5 3-5 3V8z" />
                  <path d="M8 21h8M12 17v4" />
                </svg>
              )}
            </div>
            <div className="file-drop-text">
              {file ? (
                <>
                  <span className="file-drop-name">{file.name}</span>
                  <span className="file-drop-meta">{(file.size / 1024 / 1024).toFixed(1)} MB · click to change</span>
                </>
              ) : (
                <>
                  <span className="file-drop-label">Drop video here</span>
                  <span className="file-drop-meta">or click to browse · MP4, MOV, AVI</span>
                </>
              )}
            </div>
          </div>
          {error && <span className="error-text">{error}</span>}
          {loading && uploadPct > 0 && (
            <span className="hint-text">Upload {uploadPct}%</span>
          )}
          <div className="label-row">
            <input
              type="text"
              placeholder="Game label (optional) — e.g. vs UCLA, Oct 12"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
            />
          </div>
          <button className="btn-primary" type="submit" disabled={loading}>
            {loading && <span className="spinner" />}
            {loading
              ? `Uploading${uploadPct > 0 ? ` ${uploadPct}%` : '...'}`
              : 'Analyze Game'}
          </button>
        </form>

        {/* Decorative wave SVG */}
        <svg className="wave-bg" viewBox="0 0 800 100" preserveAspectRatio="none">
          <path d="M0 50 Q100 10 200 50 T400 50 T600 50 T800 50 V100 H0Z" fill="url(#waveGrad)" opacity="0.5">
            <animate attributeName="d" dur="8s" repeatCount="indefinite"
              values="M0 50 Q100 10 200 50 T400 50 T600 50 T800 50 V100 H0Z;
                      M0 50 Q100 80 200 50 T400 50 T600 50 T800 50 V100 H0Z;
                      M0 50 Q100 10 200 50 T400 50 T600 50 T800 50 V100 H0Z" />
          </path>
          <defs>
            <linearGradient id="waveGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#00d4ff" />
              <stop offset="100%" stopColor="#3b82f6" />
            </linearGradient>
          </defs>
        </svg>
      </div>

      <GameHistory
        games={games}
        onOpen={onOpenGame}
        onDelete={onDeleteGame}
      />
    </div>
  )
}
