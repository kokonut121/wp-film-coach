import { deleteJob } from '../api'

export default function GameHistory({ games, onOpen, onDelete }) {
  if (!games.length) return null

  const handleDelete = async (e, game) => {
    e.stopPropagation()
    try {
      await deleteJob(game.job_id)
    } catch {
      // Job may not exist on server — remove locally anyway
    }
    onDelete(game.job_id)
  }

  const formatDate = (ts) => {
    const d = new Date(ts)
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
  }

  const statusLabel = (status) => {
    if (status === 'done') return 'Complete'
    if (status === 'awaiting_calibration') return 'Awaiting calibration'
    return 'Processing...'
  }

  return (
    <div className="history-section fade-in">
      <h3>Previous Analyses</h3>
      {games.map((game) => (
        <div
          key={game.job_id}
          className="history-item"
          onClick={() => onOpen(game)}
        >
          <div className="history-item-info">
            <span className="history-item-label">{game.label || 'Untitled'}</span>
            <span className="history-item-meta">
              {formatDate(game.timestamp)} &middot; {statusLabel(game.status)}
            </span>
          </div>
          <div className="history-item-actions">
            <button
              className="btn-icon danger"
              onClick={(e) => handleDelete(e, game)}
              title="Delete"
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6" />
              </svg>
            </button>
          </div>
        </div>
      ))}
    </div>
  )
}
