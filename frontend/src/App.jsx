import { useState, useCallback } from 'react'
import { useGameHistory } from './hooks/useGameHistory'
import HomeView from './components/HomeView'
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
  // view: 'home' | 'processing' | 'results'
  const [view, setView] = useState(demoMode || 'home')
  const [activeJobId, setActiveJobId] = useState(demoMode ? '__demo__' : null)
  const [activeLabel, setActiveLabel] = useState(demoMode ? 'Demo Game vs UCLA' : '')
  const { games, addGame, removeGame, updateGame } = useGameHistory()

  const handleSubmit = useCallback(
    (jobId, youtubeUrl, label) => {
      addGame({
        job_id: jobId,
        label: label || 'Untitled Game',
        youtube_url: youtubeUrl,
        timestamp: Date.now(),
        status: 'processing',
      })
      setActiveJobId(jobId)
      setActiveLabel(label || 'Untitled Game')
      setView('processing')
    },
    [addGame]
  )

  const handleProcessingDone = useCallback(() => {
    updateGame(activeJobId, { status: 'done' })
    setView('results')
  }, [activeJobId, updateGame])

  const handleOpenGame = useCallback((game) => {
    setActiveJobId(game.job_id)
    setActiveLabel(game.label)
    if (game.status === 'done') {
      setView('results')
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

      {view === 'results' && (
        <ResultsView
          jobId={activeJobId}
          label={activeLabel}
        />
      )}
    </div>
  )
}
