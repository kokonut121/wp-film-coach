/**
 * Format seconds into MM:SS or H:MM:SS
 */
export function formatTime(seconds) {
  if (seconds == null || isNaN(seconds)) return '--:--'
  const s = Math.round(seconds)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  const mm = String(m).padStart(2, '0')
  const ss = String(sec).padStart(2, '0')
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`
}

/**
 * Map team color string from backend to a CSS color
 */
export function teamColor(colorStr, fallback = '#888') {
  const map = {
    blue_caps: '#3b82f6',
    white_caps: '#e2e8f0',
    red_caps: '#ef4444',
    black_caps: '#64748b',
    green_caps: '#22c55e',
    yellow_caps: '#eab308',
    dark_caps: '#475569',
  }
  return map[colorStr] || fallback
}

/**
 * Event type → color mapping
 */
export const EVENT_COLORS = {
  turnover: '#ef4444',
  goal: '#22c55e',
  shot: '#10b981',
  man_up: '#eab308',
  man_down: '#f59e0b',
  exclusion: '#f97316',
  counter_attack: '#a855f7',
  press_trigger: '#06b6d4',
}
