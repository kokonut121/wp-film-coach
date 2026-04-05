// FINA standard water polo pool dimensions
export const POOL = {
  width: 25,   // metres (goal line to goal line)
  height: 13,  // metres (side to side)
  lines: [
    { x: 0, label: 'Goal', color: 'rgba(255,255,255,0.6)' },
    { x: 2, label: '2m', color: 'rgba(255,60,60,0.5)' },
    { x: 5, label: '5m', color: 'rgba(255,220,60,0.5)' },
    { x: 12.5, label: 'Half', color: 'rgba(255,255,255,0.4)' },
    { x: 20, label: '5m', color: 'rgba(255,220,60,0.5)' },
    { x: 23, label: '2m', color: 'rgba(255,60,60,0.5)' },
    { x: 25, label: 'Goal', color: 'rgba(255,255,255,0.6)' },
  ],
}

// Map pool coordinates to SVG pixel coordinates
export function poolToSvg(x, y, svgWidth, svgHeight, padding = 30) {
  const drawW = svgWidth - padding * 2
  const drawH = svgHeight - padding * 2
  return {
    sx: padding + (x / POOL.width) * drawW,
    sy: padding + (y / POOL.height) * drawH,
  }
}
