import { useMemo, useRef, useState } from 'react'
import { getCalibrationFrameUrl, submitCalibration } from '../api'

const LINE_KEYS = [
  'left_side',
  'top_side',
  'right_side',
  'bottom_side',
  'm2_left',
  'm5_left',
  'half',
  'm5_right',
  'm2_right',
]

const LINE_LABELS = {
  left_side: 'Left pool side',
  top_side: 'Top pool side',
  right_side: 'Right pool side',
  bottom_side: 'Bottom pool side',
  m2_left: 'Left 2m line',
  m5_left: 'Left 5m line',
  half: 'Half line',
  m5_right: 'Right 5m line',
  m2_right: 'Right 2m line',
}

export default function CalibrationView({ jobId, label, onDone, onBack }) {
  const imageRef = useRef(null)
  const [naturalSize, setNaturalSize] = useState({ width: 1, height: 1 })
  const [lines, setLines] = useState([])
  const [draftLine, setDraftLine] = useState(null)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const frameUrl = useMemo(() => `${getCalibrationFrameUrl(jobId)}?t=${jobId}`, [jobId])

  const nextKey = LINE_KEYS[lines.length]

  const getPoint = (event) => {
    if (!imageRef.current) return null
    const rect = imageRef.current.getBoundingClientRect()
    const scaleX = naturalSize.width / rect.width
    const scaleY = naturalSize.height / rect.height
    return {
      x: Math.round((event.clientX - rect.left) * scaleX * 10) / 10,
      y: Math.round((event.clientY - rect.top) * scaleY * 10) / 10,
    }
  }

  const handlePointerDown = (event) => {
    if (!nextKey || submitting) return
    const point = getPoint(event)
    if (!point) return
    setDraftLine({ key: nextKey, x1: point.x, y1: point.y, x2: point.x, y2: point.y })
    setError('')
  }

  const handlePointerMove = (event) => {
    if (!draftLine) return
    const point = getPoint(event)
    if (!point) return
    setDraftLine((prev) => ({ ...prev, x2: point.x, y2: point.y }))
  }

  const handlePointerUp = (event) => {
    if (!draftLine) return
    const point = getPoint(event)
    const finalLine = point ? { ...draftLine, x2: point.x, y2: point.y } : draftLine
    setDraftLine(null)

    if (Math.abs(finalLine.x1 - finalLine.x2) < 1 && Math.abs(finalLine.y1 - finalLine.y2) < 1) {
      setError('Drag a visible segment so the line has distinct endpoints.')
      return
    }

    setLines((prev) => [...prev, finalLine])
  }

  const handleUndo = () => {
    setDraftLine(null)
    setLines((prev) => prev.slice(0, -1))
  }

  const handleReset = () => {
    setDraftLine(null)
    setLines([])
  }

  const handleSubmit = async () => {
    setSubmitting(true)
    setError('')
    try {
      await submitCalibration(jobId, lines)
      onDone()
    } catch (err) {
      setError(err.message)
    } finally {
      setSubmitting(false)
    }
  }

  const renderLine = (line, index, isDraft = false) => {
    const label = isDraft ? LINE_LABELS[nextKey] : `${index + 1}. ${LINE_LABELS[line.key]}`
    const labelX = (line.x1 + line.x2) / 2
    const labelY = (line.y1 + line.y2) / 2

    return (
      <g key={isDraft ? 'draft' : line.key}>
        <line
          x1={line.x1}
          y1={line.y1}
          x2={line.x2}
          y2={line.y2}
          className={`calibration-line ${isDraft ? 'draft' : ''}`}
        />
        <text x={labelX} y={labelY - 8} className="calibration-line-label">
          {label}
        </text>
      </g>
    )
  }

  return (
    <div className="calibration-view">
      <div className="calibration-header">
        <h2>{label || 'Manual Pool Calibration'}</h2>
        <p>
          Draw each straight pool reference line in order. You only need to trace the visible
          segment and the calibration step will extend it mathematically.
        </p>
      </div>

      <div className="calibration-layout">
        <div className="calibration-stage panel">
          <div className="panel-header">
            <span><span className="dot" />Calibration Frame</span>
            <span>{lines.length}/{LINE_KEYS.length} lines</span>
          </div>
          <div className="calibration-frame-wrap">
            <img
              ref={imageRef}
              src={frameUrl}
              alt="Calibration frame"
              className="calibration-frame"
              onLoad={(event) => {
                setNaturalSize({
                  width: event.currentTarget.naturalWidth || 1,
                  height: event.currentTarget.naturalHeight || 1,
                })
              }}
            />
            <div
              className="calibration-draw-layer"
              onMouseDown={handlePointerDown}
              onMouseMove={handlePointerMove}
              onMouseUp={handlePointerUp}
              onMouseLeave={handlePointerUp}
            >
              <svg className="calibration-lines-svg" viewBox={`0 0 ${naturalSize.width} ${naturalSize.height}`} preserveAspectRatio="none">
                {lines.map((line, index) => renderLine(line, index))}
                {draftLine && renderLine(draftLine, lines.length, true)}
              </svg>
            </div>
          </div>
        </div>

        <div className="calibration-sidebar panel">
          <div className="panel-header">
            <span><span className="dot" />Line Order</span>
            <span>{nextKey ? LINE_LABELS[nextKey] : 'ready'}</span>
          </div>
          <div className="panel-body calibration-sidebar-body">
            <ol className="calibration-list">
              {LINE_KEYS.map((key, index) => (
                <li key={key} className={index < lines.length ? 'done' : index === lines.length ? 'active' : ''}>
                  {LINE_LABELS[key]}
                </li>
              ))}
            </ol>

            {error && <div className="processing-error">{error}</div>}

            <div className="calibration-actions">
              <button className="nav-link" onClick={handleUndo} disabled={!lines.length || submitting}>
                Undo
              </button>
              <button className="nav-link" onClick={handleReset} disabled={(!lines.length && !draftLine) || submitting}>
                Reset
              </button>
              <button
                className="btn-primary"
                onClick={handleSubmit}
                disabled={lines.length !== LINE_KEYS.length || submitting}
              >
                {submitting ? 'Starting...' : 'Start Analysis'}
              </button>
              <button className="nav-link" onClick={onBack} disabled={submitting}>
                Back to Home
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
