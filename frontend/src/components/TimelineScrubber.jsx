import { useState, useCallback, useRef, useEffect } from 'react'
import { formatTime, EVENT_COLORS } from '../utils/formatTime'

export default function TimelineScrubber({
  currentTime,
  duration,
  events,
  isPlaying,
  onTimeChange,
  onPlayToggle,
}) {
  const trackRef = useRef(null)

  return (
    <div className="scrubber-container">
      <div className="scrubber-row">
        <button className="play-btn" onClick={onPlayToggle}>
          {isPlaying ? (
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
              <rect x="6" y="4" width="4" height="16" rx="1" />
              <rect x="14" y="4" width="4" height="16" rx="1" />
            </svg>
          ) : (
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
              <polygon points="6,4 20,12 6,20" />
            </svg>
          )}
        </button>

        <span className="scrubber-time">{formatTime(currentTime)}</span>

        <div className="scrubber-track" ref={trackRef}>
          {/* Event markers */}
          {events?.map((evt, i) => {
            const pct = duration > 0 ? (evt.t_seconds / duration) * 100 : 0
            return (
              <div
                key={i}
                className="event-marker"
                style={{
                  left: `${pct}%`,
                  background: EVENT_COLORS[evt.type] || '#666',
                }}
              />
            )
          })}
          <input
            type="range"
            min={0}
            max={duration || 0}
            step={0.5}
            value={currentTime}
            onChange={(e) => onTimeChange(parseFloat(e.target.value))}
          />
        </div>

        <span className="scrubber-time">{formatTime(duration)}</span>
      </div>
    </div>
  )
}
