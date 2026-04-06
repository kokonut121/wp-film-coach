import { useEffect } from 'react'
import { useJobStatus } from '../hooks/useJobStatus'

const STAGES = [
  { key: 'downloading', icon: '↓', label: 'Download' },
  { key: 'detecting', icon: '◎', label: 'Detect' },
  { key: 'tracking', icon: '⟿', label: 'Track' },
  { key: 'homography', icon: '▦', label: 'Map' },
  { key: 'classifying', icon: '⚡', label: 'Classify' },
  { key: 'generating_report', icon: '✎', label: 'Report' },
]

function stageIndex(stage) {
  const idx = STAGES.findIndex((s) => s.key === stage)
  return idx >= 0 ? idx : -1
}

export default function ProcessingView({ jobId, onDone, onBack }) {
  const { stage, pct, error, debug } = useJobStatus(jobId)
  const currentIdx = stageIndex(stage)

  useEffect(() => {
    if (stage === 'done') onDone()
  }, [stage, onDone])

  return (
    <div className="processing-view">
      <h2 className="fade-in">Processing Game</h2>
      <p className="subtitle fade-in-delay-1">
        Running the full CV pipeline on GPU
      </p>

      {/* Stage pipeline */}
      <div className="stage-pipeline fade-in-delay-2">
        {STAGES.map((s, i) => (
          <div key={s.key} style={{ display: 'flex', alignItems: 'center' }}>
            {i > 0 && (
              <div className={`stage-connector ${i <= currentIdx ? 'done' : ''}`} />
            )}
            <div
              className={`stage-node ${
                i === currentIdx ? 'active' : i < currentIdx ? 'done' : ''
              }`}
            >
              <div className="stage-icon">
                {i < currentIdx ? (
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                    <path d="M5 12l5 5L20 7" />
                  </svg>
                ) : (
                  s.icon
                )}
              </div>
              <span className="stage-label">{s.label}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Percentage + bar */}
      {!error && (
        <>
          <div className="progress-pct fade-in-delay-3">
            {pct ?? 0}%
          </div>
          <div className="progress-stage-label">
            {stage === 'queued' ? 'Waiting for GPU...' : stage?.replace(/_/g, ' ')}
          </div>
          <div className="progress-container">
            <div className="progress-bar-bg">
              <div
                className="progress-bar-fill"
                style={{ width: `${pct ?? 0}%` }}
              />
            </div>
          </div>
          <div className="processing-note">
            You can close this tab — your analysis will continue in the cloud.
            <br />
            Check back anytime from the home page.
          </div>
          {debug?.enabled && (
            <DebugPanel debug={debug} />
          )}
        </>
      )}

      {error && (
        <div className="processing-error fade-in">
          <strong>Error:</strong> {error}
          <br />
          <button
            className="nav-link"
            style={{ marginTop: 12 }}
            onClick={onBack}
          >
            Back to Home
          </button>
        </div>
      )}
    </div>
  )
}

function DebugPanel({ debug }) {
  const stageEntries = Object.entries(debug.stages || {})

  return (
    <div className="debug-panel fade-in">
      <div className="debug-panel-header">
        <span>Debug Mode</span>
        <span>{debug.current_stage?.replace(/_/g, ' ') || 'running'}</span>
      </div>

      {debug.input && (
        <div className="debug-block">
          <div className="debug-block-title">Input</div>
          <div className="debug-kv">
            {Object.entries(debug.input).map(([key, value]) => (
              <div key={key} className="debug-kv-row">
                <span>{prettyKey(key)}</span>
                <strong>{String(value)}</strong>
              </div>
            ))}
          </div>
        </div>
      )}

      {stageEntries.map(([stageKey, values]) => (
        <div key={stageKey} className="debug-block">
          <div className="debug-block-title">{prettyKey(stageKey)}</div>
          <div className="debug-kv">
            {Object.entries(values).map(([key, value]) => (
              <div key={key} className="debug-kv-row">
                <span>{prettyKey(key)}</span>
                <strong>{Array.isArray(value) ? value.join(', ') : String(value)}</strong>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

function prettyKey(value) {
  return value.replace(/_/g, ' ')
}
