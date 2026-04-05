import { useRef, useEffect, useMemo } from 'react'
import * as d3 from 'd3'
import { POOL, poolToSvg } from '../utils/poolDimensions'
import { teamColor } from '../utils/formatTime'

const TRAIL_SECONDS = 3
const SVG_W = 700
const SVG_H = 380
const PAD = 36

export default function PoolMap({ positions, formations, meta, currentTime }) {
  const svgRef = useRef(null)

  // Get positions at current time (nearest frame within 0.5s)
  const { currentPlayers, trails, currentFormation } = useMemo(() => {
    if (!positions?.length) return { currentPlayers: [], trails: {}, currentFormation: null }

    // Find players at current time (within tolerance)
    const tol = 0.2
    const current = positions.filter(
      (p) => Math.abs(p.t_seconds - currentTime) < tol
    )

    // Build trails (last TRAIL_SECONDS of positions per player)
    const trailMap = {}
    const tMin = currentTime - TRAIL_SECONDS
    for (const p of positions) {
      if (p.t_seconds >= tMin && p.t_seconds <= currentTime) {
        if (!trailMap[p.player_id]) trailMap[p.player_id] = []
        trailMap[p.player_id].push(p)
      }
    }

    // Find current formation
    let formation = null
    if (formations?.length) {
      for (let i = formations.length - 1; i >= 0; i--) {
        if (formations[i].t_seconds <= currentTime) {
          formation = formations[i]
          break
        }
      }
    }

    return { currentPlayers: current, trails: trailMap, currentFormation: formation }
  }, [positions, formations, currentTime])

  const colorA = teamColor(meta?.team_a_colour, '#3b82f6')
  const colorB = teamColor(meta?.team_b_colour, '#e2e8f0')

  useEffect(() => {
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    // Defs for gradients and glows
    const defs = svg.append('defs')

    // Pool water gradient
    const waterGrad = defs.append('linearGradient')
      .attr('id', 'poolWater')
      .attr('x1', '0%').attr('y1', '0%')
      .attr('x2', '100%').attr('y2', '100%')
    waterGrad.append('stop').attr('offset', '0%').attr('stop-color', '#0c2d48')
    waterGrad.append('stop').attr('offset', '50%').attr('stop-color', '#0a1f35')
    waterGrad.append('stop').attr('offset', '100%').attr('stop-color', '#071525')

    // Glow filters
    const glowA = defs.append('filter').attr('id', 'glowA')
    glowA.append('feGaussianBlur').attr('stdDeviation', '3').attr('result', 'blur')
    glowA.append('feMerge').selectAll('feMergeNode')
      .data(['blur', 'SourceGraphic']).join('feMergeNode')
      .attr('in', d => d)

    const glowB = defs.append('filter').attr('id', 'glowB')
    glowB.append('feGaussianBlur').attr('stdDeviation', '3').attr('result', 'blur')
    glowB.append('feMerge').selectAll('feMergeNode')
      .data(['blur', 'SourceGraphic']).join('feMergeNode')
      .attr('in', d => d)

    const glowBall = defs.append('filter').attr('id', 'glowBall')
    glowBall.append('feGaussianBlur').attr('stdDeviation', '4').attr('result', 'blur')
    glowBall.append('feMerge').selectAll('feMergeNode')
      .data(['blur', 'SourceGraphic']).join('feMergeNode')
      .attr('in', d => d)

    // Pool background
    svg.append('rect')
      .attr('x', PAD).attr('y', PAD)
      .attr('width', SVG_W - PAD * 2).attr('height', SVG_H - PAD * 2)
      .attr('rx', 4).attr('ry', 4)
      .attr('fill', 'url(#poolWater)')
      .attr('stroke', 'rgba(0,212,255,0.15)')
      .attr('stroke-width', 1.5)

    // Pool lane lines
    const lineGroup = svg.append('g')
    for (const line of POOL.lines) {
      const { sx } = poolToSvg(line.x, 0, SVG_W, SVG_H, PAD)
      lineGroup.append('line')
        .attr('x1', sx).attr('y1', PAD)
        .attr('x2', sx).attr('y2', SVG_H - PAD)
        .attr('stroke', line.color)
        .attr('stroke-width', line.label === 'Goal' ? 2 : 1)
        .attr('stroke-dasharray', line.label === 'Goal' ? 'none' : '4 3')

      // Line labels at top
      lineGroup.append('text')
        .attr('x', sx).attr('y', PAD - 6)
        .attr('text-anchor', 'middle')
        .attr('font-family', 'var(--font-mono)')
        .attr('font-size', 8)
        .attr('fill', 'rgba(255,255,255,0.25)')
        .text(line.label)
    }

    // Goal areas (small rectangles at each end)
    const goalW = 12
    const goalH = 60
    const goalY = SVG_H / 2 - goalH / 2
    svg.append('rect')
      .attr('x', PAD - goalW).attr('y', goalY)
      .attr('width', goalW).attr('height', goalH)
      .attr('rx', 2)
      .attr('fill', 'none').attr('stroke', 'rgba(255,255,255,0.3)')
      .attr('stroke-width', 1.5)
    svg.append('rect')
      .attr('x', SVG_W - PAD).attr('y', goalY)
      .attr('width', goalW).attr('height', goalH)
      .attr('rx', 2)
      .attr('fill', 'none').attr('stroke', 'rgba(255,255,255,0.3)')
      .attr('stroke-width', 1.5)

    // Metre labels along bottom
    for (let m = 0; m <= 25; m += 5) {
      const { sx } = poolToSvg(m, 0, SVG_W, SVG_H, PAD)
      svg.append('text')
        .attr('x', sx).attr('y', SVG_H - PAD + 16)
        .attr('text-anchor', 'middle')
        .attr('font-family', 'var(--font-mono)')
        .attr('font-size', 8)
        .attr('fill', 'rgba(255,255,255,0.2)')
        .text(`${m}m`)
    }

    // Player trails
    const trailGroup = svg.append('g')
    const lineGen = d3.line()
      .x(d => poolToSvg(d.x_metres, d.y_metres, SVG_W, SVG_H, PAD).sx)
      .y(d => poolToSvg(d.x_metres, d.y_metres, SVG_W, SVG_H, PAD).sy)
      .curve(d3.curveCatmullRom)

    for (const [pid, pts] of Object.entries(trails)) {
      if (pts.length < 2) continue
      const sorted = pts.sort((a, b) => a.t_seconds - b.t_seconds)
      const color = sorted[0].team === 'team_a' ? colorA : colorB
      trailGroup.append('path')
        .attr('d', lineGen(sorted))
        .attr('fill', 'none')
        .attr('stroke', color)
        .attr('stroke-width', 2)
        .attr('stroke-linecap', 'round')
        .attr('opacity', 0.3)
    }

    // Player markers
    const playerGroup = svg.append('g')
    for (const p of currentPlayers) {
      const { sx, sy } = poolToSvg(p.x_metres, p.y_metres, SVG_W, SVG_H, PAD)
      const isA = p.team === 'team_a'
      const color = isA ? colorA : colorB
      const filterId = isA ? 'glowA' : 'glowB'

      // Outer glow ring
      playerGroup.append('circle')
        .attr('cx', sx).attr('cy', sy).attr('r', 10)
        .attr('fill', color)
        .attr('opacity', 0.15)
        .attr('filter', `url(#${filterId})`)

      // Player circle
      playerGroup.append('circle')
        .attr('cx', sx).attr('cy', sy).attr('r', 6)
        .attr('fill', color)
        .attr('stroke', 'rgba(255,255,255,0.5)')
        .attr('stroke-width', 1)
        .attr('filter', `url(#${filterId})`)

      // Player ID label
      playerGroup.append('text')
        .attr('x', sx).attr('y', sy - 10)
        .attr('text-anchor', 'middle')
        .attr('font-family', 'var(--font-mono)')
        .attr('font-size', 7)
        .attr('fill', 'rgba(255,255,255,0.6)')
        .text(p.player_id)
    }

    // Formation label
    if (currentFormation) {
      svg.append('text')
        .attr('x', SVG_W / 2).attr('y', SVG_H - PAD + 30)
        .attr('text-anchor', 'middle')
        .attr('font-family', 'var(--font-display)')
        .attr('font-size', 11)
        .attr('font-weight', 600)
        .attr('fill', 'rgba(0,212,255,0.4)')
        .text(`${currentFormation.team === 'team_a' ? 'A' : 'B'}: ${currentFormation.formation}`)
    }

  }, [currentPlayers, trails, currentFormation, colorA, colorB])

  return (
    <svg
      ref={svgRef}
      className="pool-svg"
      viewBox={`0 0 ${SVG_W} ${SVG_H}`}
      preserveAspectRatio="xMidYMid meet"
    />
  )
}
