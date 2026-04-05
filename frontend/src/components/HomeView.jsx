import { useState, useRef } from 'react'
import { submitGame, uploadGame } from '../api'
import GameHistory from './GameHistory'

const URL_RE = /^https?:\/\/(www\.)?(youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/shorts\/)[\w-]+/

export default function HomeView({ games, onSubmit, onOpenGame, onDeleteGame }) {
  const [url, setUrl] = useState('')
  const [label, setLabel] = useState('')
  const [file, setFile] = useState(null)
  const [mode, setMode] = useState('file') // 'url' | 'file'
  const [loading, setLoading] = useState(false)
  const [uploadPct, setUploadPct] = useState(0)
  const [error, setError] = useState('')
  const fileRef = useRef()

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')

    if (mode === 'url') {
      if (!URL_RE.test(url.trim())) {
        setError('Enter a valid YouTube URL')
        return
      }
      setLoading(true)
      setUploadPct(0)
      try {
        const { job_id } = await submitGame(url.trim(), label.trim())
        onSubmit(job_id, url.trim(), label.trim())
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    } else {
      if (!file) {
        setError('Select a video file')
        return
      }
      setLoading(true)
      setUploadPct(0)
      try {
        const { job_id } = await uploadGame(file, label.trim(), setUploadPct)
        onSubmit(job_id, file.name, label.trim())
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
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
          Upload a game video or paste a YouTube link. Our CV pipeline detects
          players, tracks formations, and generates tactical analysis — powered by AI.
        </p>

        <form className="submit-form fade-in-delay-4" onSubmit={handleSubmit}>
          <div className="mode-toggle">
            <button type="button" className={mode === 'file' ? 'active' : ''} onClick={() => { setMode('file'); setError('') }}>Upload Video</button>
            <button type="button" className={mode === 'url' ? 'active' : ''} onClick={() => { setMode('url'); setError('') }}>YouTube URL</button>
          </div>
          {mode === 'file' ? (
            <div className={`input-group ${error ? 'error' : ''}`}>
              <input
                ref={fileRef}
                type="file"
                accept="video/*"
                onChange={(e) => { setFile(e.target.files[0] || null); setError(''); }}
              />
            </div>
          ) : (
            <div className={`input-group ${error ? 'error' : ''}`}>
              <input
                type="text"
                placeholder="https://youtube.com/watch?v=..."
                value={url}
                onChange={(e) => { setUrl(e.target.value); setError(''); }}
                autoFocus
              />
              <span className="hint-text">Requires local proxy: python local_proxy.py</span>
            </div>
          )}
          {error && <span className="error-text">{error}</span>}
          {mode === 'file' && loading && uploadPct > 0 && (
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
              ? (mode === 'file'
                ? `Uploading${uploadPct > 0 ? ` ${uploadPct}%` : '...'}`
                : 'Submitting...')
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
