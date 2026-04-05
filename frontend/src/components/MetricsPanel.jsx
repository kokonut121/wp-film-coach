import { useState, useRef, useEffect } from 'react'
import * as d3 from 'd3'
import { POOL, poolToSvg } from '../utils/poolDimensions'
import { teamColor, formatTime } from '../utils/formatTime'

const TABS = [
  { key: 'possession', label: 'Possession' },
  { key: 'hull', label: 'Hull Area' },
  { key: 'spread', label: 'Spread' },
  { key: 'heatmap', label: 'Heatmap' },
  { key: 'formations', label: 'Formations' },
]

export default function MetricsPanel({ metrics, formations, meta, currentTime }) {
  const [activeTab, setActiveTab] = useState('possession')
  const chartRef = useRef(null)

  const colorA = teamColor(meta?.team_a_colour, '#3b82f6')
  const colorB = teamColor(meta?.team_b_colour, '#e2e8f0')

  useEffect(() => {
    if (!chartRef.current) return
    const container = chartRef.current
    const rect = container.getBoundingClientRect()
    const width = rect.width || 280
    const height = rect.height || 200

    // Clear previous
    d3.select(container).selectAll('*').remove()

    if (activeTab === 'possession') {
      renderPossession(container, metrics?.possession, width, height, colorA, colorB)
    } else if (activeTab === 'hull') {
      renderTimeSeries(container, metrics?.hull_area, 'Hull Area (m²)', width, height, colorA, colorB)
    } else if (activeTab === 'spread') {
      renderTimeSeries(container, metrics?.centroid_spread, 'Spread (m)', width, height, colorA, colorB)
    } else if (activeTab === 'heatmap') {
      renderHeatmap(container, metrics?.heatmaps, width, height)
    } else if (activeTab === 'formations') {
      renderFormations(container, formations, width, height, meta?.duration_s || 3600, colorA)
    }
  }, [activeTab, metrics, formations, meta, colorA, colorB, currentTime])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
      <div className="metrics-tabs">
        {TABS.map((tab) => (
          <button
            key={tab.key}
            className={`metrics-tab ${activeTab === tab.key ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.key)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div style={{ flex: 1, padding: '12px 14px', overflow: 'hidden' }}>
        <div ref={chartRef} className="metrics-chart" />
      </div>
    </div>
  )
}

function renderPossession(container, possession, width, height, colorA, colorB) {
  if (!possession || !Object.keys(possession).length) {
    d3.select(container).append('div').attr('class', 'chart-empty').text('No possession data')
    return
  }

  const periods = Object.entries(possession).map(([key, val]) => ({
    period: key.replace('period_', 'P'),
    a: val.team_a || 0,
    b: val.team_b || 0,
  }))

  const margin = { top: 20, right: 16, bottom: 30, left: 40 }
  const w = width - margin.left - margin.right
  const h = height - margin.top - margin.bottom

  const svg = d3.select(container).append('svg')
    .attr('width', width).attr('height', height)
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

  const x = d3.scaleBand().domain(periods.map(d => d.period)).range([0, w]).padding(0.35)
  const y = d3.scaleLinear().domain([0, 1]).range([h, 0])

  // Axes
  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${h})`)
    .call(d3.axisBottom(x).tickSize(0)).select('.domain').remove()
  g.append('g').attr('class', 'axis')
    .call(d3.axisLeft(y).ticks(4).tickFormat(d3.format('.0%')).tickSize(-w))
    .select('.domain').remove()
  g.selectAll('.axis .tick line').attr('stroke', 'rgba(255,255,255,0.05)')

  // Stacked bars
  periods.forEach(d => {
    g.append('rect')
      .attr('x', x(d.period)).attr('y', y(d.a))
      .attr('width', x.bandwidth()).attr('height', h - y(d.a))
      .attr('fill', colorA).attr('rx', 3).attr('opacity', 0.8)
    g.append('rect')
      .attr('x', x(d.period)).attr('y', y(d.a + d.b))
      .attr('width', x.bandwidth()).attr('height', h - y(d.b))
      .attr('fill', colorB).attr('rx', 3).attr('opacity', 0.6)
      .attr('transform', `translate(0,${y(d.a) - y(d.a + d.b)})`)
  })

  // Legend
  const leg = svg.append('g').attr('transform', `translate(${margin.left}, 6)`)
  leg.append('rect').attr('width', 8).attr('height', 8).attr('rx', 2).attr('fill', colorA)
  leg.append('text').attr('x', 12).attr('y', 8).attr('font-size', 8)
    .attr('fill', 'rgba(255,255,255,0.5)').attr('font-family', 'var(--font-mono)').text('Team A')
  leg.append('rect').attr('x', 60).attr('width', 8).attr('height', 8).attr('rx', 2).attr('fill', colorB)
  leg.append('text').attr('x', 72).attr('y', 8).attr('font-size', 8)
    .attr('fill', 'rgba(255,255,255,0.5)').attr('font-family', 'var(--font-mono)').text('Team B')
}

function renderTimeSeries(container, data, label, width, height, colorA, colorB) {
  if (!data?.length) {
    d3.select(container).append('div').attr('class', 'chart-empty').text(`No ${label.toLowerCase()} data`)
    return
  }

  const margin = { top: 14, right: 16, bottom: 30, left: 44 }
  const w = width - margin.left - margin.right
  const h = height - margin.top - margin.bottom

  const svg = d3.select(container).append('svg')
    .attr('width', width).attr('height', height)
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

  const x = d3.scaleLinear()
    .domain(d3.extent(data, d => d.t_seconds))
    .range([0, w])

  const allVals = data.flatMap(d => [d.team_a, d.team_b].filter(v => v != null))
  const y = d3.scaleLinear()
    .domain([0, d3.max(allVals) * 1.1])
    .range([h, 0])

  // Axes
  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${h})`)
    .call(d3.axisBottom(x).ticks(5).tickFormat(d => formatTime(d)).tickSize(0))
    .select('.domain').remove()
  g.append('g').attr('class', 'axis')
    .call(d3.axisLeft(y).ticks(4).tickSize(-w))
    .select('.domain').remove()
  g.selectAll('.axis .tick line').attr('stroke', 'rgba(255,255,255,0.05)')

  // Team A area
  const areaA = d3.area()
    .x(d => x(d.t_seconds))
    .y0(h)
    .y1(d => y(d.team_a))
    .curve(d3.curveMonotoneX)

  g.append('path').datum(data)
    .attr('d', areaA)
    .attr('fill', colorA).attr('opacity', 0.15)

  g.append('path').datum(data)
    .attr('d', d3.line().x(d => x(d.t_seconds)).y(d => y(d.team_a)).curve(d3.curveMonotoneX))
    .attr('fill', 'none').attr('stroke', colorA).attr('stroke-width', 1.5)

  // Team B area
  const areaB = d3.area()
    .x(d => x(d.t_seconds))
    .y0(h)
    .y1(d => y(d.team_b))
    .curve(d3.curveMonotoneX)

  g.append('path').datum(data)
    .attr('d', areaB)
    .attr('fill', colorB).attr('opacity', 0.1)

  g.append('path').datum(data)
    .attr('d', d3.line().x(d => x(d.t_seconds)).y(d => y(d.team_b)).curve(d3.curveMonotoneX))
    .attr('fill', 'none').attr('stroke', colorB).attr('stroke-width', 1.5)

  // Y label
  g.append('text')
    .attr('transform', 'rotate(-90)')
    .attr('y', -34).attr('x', -h / 2)
    .attr('text-anchor', 'middle')
    .attr('font-family', 'var(--font-mono)')
    .attr('font-size', 8)
    .attr('fill', 'rgba(255,255,255,0.3)')
    .text(label)
}

function renderHeatmap(container, heatmaps, width, height) {
  if (!heatmaps || !Object.keys(heatmaps).length) {
    d3.select(container).append('div').attr('class', 'chart-empty').text('No heatmap data')
    return
  }

  // Aggregate all player heatmaps into one
  const firstKey = Object.keys(heatmaps)[0]
  const grid = heatmaps[firstKey]
  if (!grid?.length) {
    d3.select(container).append('div').attr('class', 'chart-empty').text('No heatmap data')
    return
  }

  // Sum all player heatmaps
  const rows = grid.length
  const cols = grid[0]?.length || 0
  const combined = Array.from({ length: rows }, () => new Float32Array(cols))

  for (const pid of Object.keys(heatmaps)) {
    const h = heatmaps[pid]
    if (!h?.length) continue
    for (let r = 0; r < Math.min(rows, h.length); r++) {
      for (let c = 0; c < Math.min(cols, (h[r]?.length || 0)); c++) {
        combined[r][c] += h[r][c]
      }
    }
  }

  const maxVal = Math.max(1, ...combined.flat())

  const margin = { top: 10, right: 10, bottom: 10, left: 10 }
  const w = width - margin.left - margin.right
  const h2 = height - margin.top - margin.bottom

  const cellW = w / cols
  const cellH = h2 / rows

  const svg = d3.select(container).append('svg')
    .attr('width', width).attr('height', height)
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

  // Pool outline
  g.append('rect')
    .attr('width', w).attr('height', h2)
    .attr('rx', 3).attr('fill', '#071525').attr('stroke', 'rgba(0,212,255,0.15)')

  const colorScale = d3.scaleSequential(d3.interpolateInferno).domain([0, maxVal])

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (combined[r][c] > 0) {
        g.append('rect')
          .attr('x', c * cellW).attr('y', r * cellH)
          .attr('width', cellW + 0.5).attr('height', cellH + 0.5)
          .attr('fill', colorScale(combined[r][c]))
          .attr('opacity', 0.8)
      }
    }
  }
}

function renderFormations(container, formations, width, height, duration, colorA) {
  if (!formations?.length) {
    d3.select(container).append('div').attr('class', 'chart-empty').text('No formation data')
    return
  }

  const margin = { top: 14, right: 16, bottom: 30, left: 56 }
  const w = width - margin.left - margin.right
  const h = height - margin.top - margin.bottom

  const svg = d3.select(container).append('svg')
    .attr('width', width).attr('height', height)
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

  const formationNames = [...new Set(formations.map(f => f.formation))]

  const x = d3.scaleLinear().domain([0, duration]).range([0, w])
  const y = d3.scaleBand().domain(formationNames).range([0, h]).padding(0.25)

  // Axes
  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${h})`)
    .call(d3.axisBottom(x).ticks(5).tickFormat(d => formatTime(d)).tickSize(0))
    .select('.domain').remove()
  g.append('g').attr('class', 'axis')
    .call(d3.axisLeft(y).tickSize(0))
    .select('.domain').remove()

  // Bars for each formation segment
  for (let i = 0; i < formations.length; i++) {
    const f = formations[i]
    const nextT = formations[i + 1]?.t_seconds || f.t_seconds + 10
    g.append('rect')
      .attr('x', x(f.t_seconds))
      .attr('y', y(f.formation))
      .attr('width', Math.max(2, x(nextT) - x(f.t_seconds)))
      .attr('height', y.bandwidth())
      .attr('rx', 2)
      .attr('fill', colorA)
      .attr('opacity', f.confidence || 0.6)
  }
}
