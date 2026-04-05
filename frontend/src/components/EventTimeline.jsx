import { useEffect, useRef } from 'react'
import { formatTime, EVENT_COLORS } from '../utils/formatTime'

const TYPE_LABELS = {
  turnover: 'Turnover',
  goal: 'Goal',
  shot: 'Shot',
  man_up: 'Man Up',
  man_down: 'Man Down',
  exclusion: 'Exclusion',
  counter_attack: 'Counter Attack',
  press_trigger: 'Press Trigger',
}

export default function EventTimeline({ events, currentTime, onSeek }) {
  const listRef = useRef(null)
  const activeRef = useRef(null)

  // Auto-scroll to closest event
  useEffect(() => {
    if (activeRef.current) {
      activeRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    }
  }, [currentTime])

  if (!events?.length) {
    return (
      <div className="event-timeline-panel panel">
        <div className="panel-header">
          <span><span className="dot" />Events</span>
        </div>
        <div className="panel-body">
          <div className="empty-state">No events detected</div>
        </div>
      </div>
    )
  }

  // Find closest event index
  let closestIdx = 0
  let closestDist = Infinity
  events.forEach((evt, i) => {
    const dist = Math.abs(evt.t_seconds - currentTime)
    if (dist < closestDist) {
      closestDist = dist
      closestIdx = i
    }
  })

  return (
    <div className="event-timeline-panel panel">
      <div className="panel-header">
        <span><span className="dot" />Events</span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.68rem', fontWeight: 400, textTransform: 'none', letterSpacing: 0 }}>
          {events.length} detected
        </span>
      </div>
      <div className="panel-body" ref={listRef}>
        {events.map((evt, i) => {
          const isActive = i === closestIdx && closestDist < 5
          return (
            <div
              key={i}
              ref={isActive ? activeRef : null}
              className={`event-item ${isActive ? 'highlight' : ''}`}
              onClick={() => onSeek(evt.t_seconds)}
            >
              <div
                className="event-dot"
                style={{ background: EVENT_COLORS[evt.type] || '#666' }}
              />
              <div className="event-content">
                <div
                  className="event-type"
                  style={{ color: EVENT_COLORS[evt.type] || '#888' }}
                >
                  {TYPE_LABELS[evt.type] || evt.type}
                </div>
                {evt.detail && <div className="event-detail">{evt.detail}</div>}
              </div>
              <span className="event-time">{formatTime(evt.t_seconds)}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
