import { useState, useEffect, useRef, useCallback } from 'react'
import { fetchResults } from '../api'
import TimelineScrubber from './TimelineScrubber'
import PoolMap from './PoolMap'
import EventTimeline from './EventTimeline'
import MetricsPanel from './MetricsPanel'
import ChatPane from './ChatPane'
import { formatTime, teamColor } from '../utils/formatTime'

const PLAYBACK_SPEED = 1 // seconds of game time per real second
const TICK_MS = 100

export default function ResultsView({ jobId, label }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [currentTime, setCurrentTime] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const playRef = useRef(null)

  // Fetch results
  useEffect(() => {
    let cancelled = false
    setLoading(true)

    const load = jobId === '__demo__' ? Promise.resolve(generateMockData()) : fetchResults(jobId)

    load
      .then((d) => {
        if (!cancelled) {
          setData(d)
          setLoading(false)
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err.message)
          setLoading(false)
        }
      })
    return () => { cancelled = true }
  }, [jobId])

  // Playback timer
  useEffect(() => {
    if (!isPlaying || !data) return
    const duration = data.meta?.duration_s || 0
    playRef.current = setInterval(() => {
      setCurrentTime((t) => {
        const next = t + PLAYBACK_SPEED * (TICK_MS / 1000)
        if (next >= duration) {
          setIsPlaying(false)
          return duration
        }
        return next
      })
    }, TICK_MS)
    return () => clearInterval(playRef.current)
  }, [isPlaying, data])

  const handleTimeChange = useCallback((t) => {
    setCurrentTime(t)
    setIsPlaying(false)
  }, [])

  const handlePlayToggle = useCallback(() => {
    setIsPlaying((p) => !p)
  }, [])

  if (loading) {
    return (
      <div className="results-view">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flex: 1 }}>
          <div className="spinner" style={{ width: 24, height: 24, borderWidth: 3, borderColor: 'rgba(0,212,255,0.2)', borderTopColor: 'var(--cyan)' }} />
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="results-view">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flex: 1 }}>
          <div className="processing-error">{error}</div>
        </div>
      </div>
    )
  }

  const duration = data?.meta?.duration_s || 0
  const colorA = teamColor(data?.meta?.team_a_colour, '#3b82f6')
  const colorB = teamColor(data?.meta?.team_b_colour, '#e2e8f0')

  return (
    <div className="results-view">
      {/* Top info bar */}
      <div className="results-top">
        <div className="results-top-info">
          <h2>{label || data?.meta?.label || 'Game Analysis'}</h2>
          <div className="team-badge">
            <span className="team-dot" style={{ background: colorA }} />
            Team A
          </div>
          <div className="team-badge">
            <span className="team-dot" style={{ background: colorB }} />
            Team B
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <span className="stat-pill">{formatTime(duration)} duration</span>
          <span className="stat-pill">{data?.events?.length || 0} events</span>
          <span className="stat-pill">{data?.formations?.length || 0} formations</span>
        </div>
      </div>

      {/* Shared timeline scrubber */}
      <TimelineScrubber
        currentTime={currentTime}
        duration={duration}
        events={data?.events}
        isPlaying={isPlaying}
        onTimeChange={handleTimeChange}
        onPlayToggle={handlePlayToggle}
      />

      {/* Dashboard grid */}
      <div className="dashboard-grid">
        {/* Pool Map - spans both rows on the left */}
        <div className="pool-map-panel panel">
          <div className="panel-header">
            <span><span className="dot" />Tactical Map</span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.68rem', fontWeight: 400, textTransform: 'none', letterSpacing: 0, color: 'var(--cyan)' }}>
              {formatTime(currentTime)}
            </span>
          </div>
          <div className="panel-body">
            <PoolMap
              positions={data?.positions}
              formations={data?.formations}
              meta={data?.meta}
              currentTime={currentTime}
            />
          </div>
        </div>

        {/* Event Timeline - top right */}
        <EventTimeline
          events={data?.events}
          currentTime={currentTime}
          onSeek={handleTimeChange}
        />

        {/* Metrics / Chat - bottom right, tabbed */}
        <BottomRightPanel
          metrics={data?.metrics}
          formations={data?.formations}
          meta={data?.meta}
          currentTime={currentTime}
          jobId={jobId}
          report={data?.report}
        />
      </div>
    </div>
  )
}

/**
 * Bottom-right panel toggles between Metrics and Chat
 */
function BottomRightPanel({ metrics, formations, meta, currentTime, jobId, report }) {
  const [mode, setMode] = useState('metrics')

  return (
    <div className="panel" style={{ display: 'flex', flexDirection: 'column' }}>
      {/* Mini toggle */}
      <div style={{
        display: 'flex',
        borderBottom: '1px solid var(--border)',
        background: 'var(--bg-panel)',
      }}>
        <button
          className={`metrics-tab ${mode === 'metrics' ? 'active' : ''}`}
          onClick={() => setMode('metrics')}
          style={{ flex: 1 }}
        >
          Metrics
        </button>
        <button
          className={`metrics-tab ${mode === 'chat' ? 'active' : ''}`}
          onClick={() => setMode('chat')}
          style={{ flex: 1 }}
        >
          AI Chat
        </button>
      </div>

      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {mode === 'metrics' ? (
          <MetricsPanel
            metrics={metrics}
            formations={formations}
            meta={meta}
            currentTime={currentTime}
          />
        ) : (
          <ChatPane jobId={jobId} report={report} />
        )}
      </div>
    </div>
  )
}

/** Generate mock data for demo mode */
function generateMockData() {
  const duration = 480 // 8 minutes
  const positions = []
  const teamAPlayers = [1, 2, 3, 4, 5, 6]
  const teamBPlayers = [7, 8, 9, 10, 11, 12]

  for (let t = 0; t <= duration; t += 0.3) {
    const phase = t * 0.05
    for (const pid of teamAPlayers) {
      const baseX = 4 + (pid - 1) * 3.2
      const baseY = 2 + (pid - 1) * 1.6
      positions.push({
        t_seconds: Math.round(t * 10) / 10,
        player_id: pid,
        team: 'team_a',
        x_metres: Math.max(0.5, Math.min(24.5, baseX + Math.sin(phase + pid) * 2.5)),
        y_metres: Math.max(0.5, Math.min(12.5, baseY + Math.cos(phase + pid * 0.7) * 2)),
        h_stale: false,
      })
    }
    for (const pid of teamBPlayers) {
      const baseX = 8 + (pid - 7) * 3
      const baseY = 2.5 + (pid - 7) * 1.5
      positions.push({
        t_seconds: Math.round(t * 10) / 10,
        player_id: pid,
        team: 'team_b',
        x_metres: Math.max(0.5, Math.min(24.5, baseX + Math.cos(phase + pid * 0.8) * 2.5)),
        y_metres: Math.max(0.5, Math.min(12.5, baseY + Math.sin(phase + pid * 0.5) * 2)),
        h_stale: false,
      })
    }
  }

  return {
    meta: {
      duration_s: duration,
      fps: 30,
      team_a_colour: 'blue_caps',
      team_b_colour: 'white_caps',
      label: 'Demo Game vs UCLA',
    },
    positions,
    events: [
      { t_seconds: 12, type: 'turnover', detail: 'Team A lost possession near 5m line' },
      { t_seconds: 45, type: 'man_up', detail: 'Team A 6v5 advantage' },
      { t_seconds: 78, type: 'exclusion', detail: 'Player 9 excluded (20s)' },
      { t_seconds: 120, type: 'counter_attack', detail: 'Team B fast break after turnover' },
      { t_seconds: 156, type: 'turnover', detail: 'Team B turnover at half distance' },
      { t_seconds: 195, type: 'press_trigger', detail: 'Team A hull contracted 35%' },
      { t_seconds: 230, type: 'man_up', detail: 'Team B 6v5 advantage' },
      { t_seconds: 280, type: 'counter_attack', detail: 'Team A rapid advance from turnover' },
      { t_seconds: 320, type: 'turnover', detail: 'Loose ball at centre' },
      { t_seconds: 380, type: 'exclusion', detail: 'Player 3 excluded (20s)' },
      { t_seconds: 420, type: 'press_trigger', detail: 'Team B press near 2m' },
      { t_seconds: 460, type: 'turnover', detail: 'Final possession change' },
    ],
    formations: [
      { t_seconds: 0, team: 'team_a', formation: '3-3', confidence: 0.92 },
      { t_seconds: 60, team: 'team_a', formation: '4-2', confidence: 0.85 },
      { t_seconds: 150, team: 'team_a', formation: 'arc', confidence: 0.78 },
      { t_seconds: 240, team: 'team_a', formation: '3-3', confidence: 0.88 },
      { t_seconds: 360, team: 'team_a', formation: 'umbrella', confidence: 0.81 },
    ],
    metrics: {
      possession: {
        period_1: { team_a: 0.58, team_b: 0.42 },
        period_2: { team_a: 0.45, team_b: 0.55 },
        period_3: { team_a: 0.52, team_b: 0.48 },
        period_4: { team_a: 0.61, team_b: 0.39 },
      },
      hull_area: Array.from({ length: 50 }, (_, i) => ({
        t_seconds: i * (duration / 50),
        team_a: 30 + Math.sin(i * 0.3) * 15 + Math.random() * 5,
        team_b: 28 + Math.cos(i * 0.25) * 12 + Math.random() * 5,
      })),
      centroid_spread: Array.from({ length: 50 }, (_, i) => ({
        t_seconds: i * (duration / 50),
        team_a: 4 + Math.sin(i * 0.2) * 2 + Math.random(),
        team_b: 3.8 + Math.cos(i * 0.15) * 1.8 + Math.random(),
      })),
      heatmaps: {},
    },
    report:
      '# Tactical Analysis Report\n\n## Summary\nThis was a competitive match with balanced possession. Team A showed strong defensive structure using a 3-3 formation, transitioning to 4-2 during man-up situations.\n\n## Key Moments\n- **0:45** — Team A earned a man-up advantage and applied heavy pressure near the 2m line\n- **1:18** — Player 9 exclusion opened up space for Team B counter-attack\n- **3:15** — Press trigger from Team A forced a critical turnover\n\n## Tactical Patterns\n- Team A favored the **3-3 formation** (58% of possessions)\n- Team B relied on **counter-attacks** more effectively in the 2nd period\n- Hull area contraction correlated with press triggers in 3 of 4 instances\n\n## Recommendations\n1. Team A should vary formation transitions to avoid predictability\n2. Team B needs faster recovery after turnovers\n3. Both teams showed improvement in man-up conversion vs previous matches',
  }
}
