import { useState, useCallback } from 'react'
import { useGameHistory } from './hooks/useGameHistory'
import HomeView from './components/HomeView'
import CalibrationView from './components/CalibrationView'
import ProcessingView from './components/ProcessingView'
import ResultsView from './components/ResultsView'

// Dev-only: ?demo=results or ?demo=processing to preview views with mock data
function getDemoMode() {
  if (import.meta.env.PROD) return null
  const params = new URLSearchParams(window.location.search)
  return params.get('demo')
}

export default function App() {
  const demoMode = getDemoMode()
  // view: 'home' | 'calibration' | 'processing' | 'results'
  const [view, setView] = useState(demoMode || 'home')
  const [activeJobId, setActiveJobId] = useState(demoMode ? '__demo__' : null)
  const [activeLabel, setActiveLabel] = useState(demoMode ? 'Demo Game vs UCLA' : '')
  const { games, addGame, removeGame, updateGame } = useGameHistory()

  const handleSubmit = useCallback(
    ({ jobId, source, label, status = 'processing', homographyMode = 'auto' }) => {
      addGame({
        job_id: jobId,
        label: label || 'Untitled Game',
        youtube_url: source,
        timestamp: Date.now(),
        status,
        homography_mode: homographyMode,
      })
      setActiveJobId(jobId)
      setActiveLabel(label || 'Untitled Game')
      setView(status === 'awaiting_calibration' ? 'calibration' : 'processing')
    },
    [addGame]
  )

  const handleCalibrationDone = useCallback(() => {
    updateGame(activeJobId, { status: 'processing' })
    setView('processing')
  }, [activeJobId, updateGame])

  const handleProcessingDone = useCallback(() => {
    updateGame(activeJobId, { status: 'done' })
    setView('results')
  }, [activeJobId, updateGame])

  const handleOpenGame = useCallback((game) => {
    setActiveJobId(game.job_id)
    setActiveLabel(game.label)
    if (game.status === 'done') {
      setView('results')
    } else if (game.status === 'awaiting_calibration') {
      setView('calibration')
    } else {
      setView('processing')
    }
  }, [])

  const handleGoHome = useCallback(() => {
    setView('home')
    setActiveJobId(null)
  }, [])

  return (
    <div className="app-shell">
      <nav className="top-bar">
        <div className="logo" onClick={handleGoHome}>
          <div className="logo-icon">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <path d="M8 12 L11 16 L16 9" />
            </svg>
          </div>
          FILM COACH
        </div>
        <div className="nav-links">
          <button
            className={`nav-link ${view === 'home' ? 'active' : ''}`}
            onClick={handleGoHome}
          >
            Home
          </button>
          {activeJobId && view !== 'home' && (
            <button className="nav-link active">
              {activeLabel}
            </button>
          )}
        </div>
      </nav>

      {view === 'home' && (
        <HomeView
          games={games}
          onSubmit={handleSubmit}
          onOpenGame={handleOpenGame}
          onDeleteGame={removeGame}
        />
      )}

      {view === 'processing' && (
        <ProcessingView
          jobId={activeJobId}
          onDone={handleProcessingDone}
          onBack={handleGoHome}
        />
      )}

      {view === 'calibration' && (
        <CalibrationView
          jobId={activeJobId}
          label={activeLabel}
          onDone={handleCalibrationDone}
          onBack={handleGoHome}
        />
      )}

      {view === 'results' && (
        <ResultsView
          jobId={activeJobId}
          label={activeLabel}
        />
      )}
    </div>
  )
}
