// A map of the capture, seen from above. It answers the one question the arrow cannot:
// which side have I already done? Shots are dots, uncovered targets are hollow rings,
// and you are the triangle.

import { coveredTargets } from './guidance.js'

const COLOURS = {
  covered: '#4ade80',
  pending: '#4b5563',
  shot: 'rgba(74, 222, 128, 0.85)',
  viewer: '#e8ecf2',
  subject: '#fbbf24'
}

export function drawRadar (canvas, { plan, shots, viewer }) {
  const context = canvas.getContext('2d')
  const size = canvas.width
  context.clearRect(0, 0, size, size)
  if (!viewer) return

  // Fit whatever exists into the square, so the map is readable from the first shot to
  // the last without ever changing scale under the eye more than it has to.
  const points = [[viewer.position[0], viewer.position[2]]]
  if (plan) {
    points.push([plan.centre[0], plan.centre[2]])
    for (const target of plan.targets) points.push([target.position[0], target.position[2]])
  }
  for (const shot of shots) points.push([shot.position[0], shot.position[2]])

  const xs = points.map(point => point[0])
  const ys = points.map(point => point[1])
  const spanX = Math.max(...xs) - Math.min(...xs)
  const spanY = Math.max(...ys) - Math.min(...ys)
  const span = Math.max(spanX, spanY, 1.5)
  const midX = (Math.max(...xs) + Math.min(...xs)) / 2
  const midY = (Math.max(...ys) + Math.min(...ys)) / 2
  const scale = (size - 28) / span

  const project = ([x, z]) => [size / 2 + (x - midX) * scale, size / 2 + (z - midY) * scale]

  if (plan) {
    const covered = coveredTargets(plan, shots)
    plan.targets.forEach((target, index) => {
      const [x, y] = project([target.position[0], target.position[2]])
      context.beginPath()
      context.arc(x, y, 4, 0, Math.PI * 2)
      context.fillStyle = covered[index] ? COLOURS.covered : 'transparent'
      context.strokeStyle = covered[index] ? COLOURS.covered : COLOURS.pending
      context.lineWidth = 1.5
      covered[index] ? context.fill() : context.stroke()
    })

    const [cx, cy] = project([plan.centre[0], plan.centre[2]])
    context.beginPath()
    context.arc(cx, cy, 6, 0, Math.PI * 2)
    context.strokeStyle = COLOURS.subject
    context.lineWidth = 2
    context.stroke()
  }

  context.fillStyle = COLOURS.shot
  for (const shot of shots) {
    const [x, y] = project([shot.position[0], shot.position[2]])
    context.fillRect(x - 1.5, y - 1.5, 3, 3)
  }

  // The triangle points where the phone is looking, flattened onto the floor.
  const [vx, vy] = project([viewer.position[0], viewer.position[2]])
  const heading = Math.atan2(viewer.forward[0], viewer.forward[2])
  context.save()
  context.translate(vx, vy)
  context.rotate(-heading)
  context.beginPath()
  context.moveTo(0, -9)
  context.lineTo(6, 7)
  context.lineTo(-6, 7)
  context.closePath()
  context.fillStyle = COLOURS.viewer
  context.fill()
  context.restore()
}
