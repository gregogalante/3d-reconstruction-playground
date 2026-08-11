// A phone that is not there: fabricated poses and fabricated frames, wearing the same
// interface as a real XR session.
//
// It exists so the parts that decide the quality of a dataset — the capture gate, the
// coverage model, the blur rejection, the upload queue — can be exercised on a desktop.
// The walker orbits a subject in front of where you started, drifting up and down so
// both guidance rings get covered, and every eighth frame comes out blurred so the
// rejection has something to reject.

const RADIUS = 1.6
const DEGREES_PER_SECOND = 26
const BLUR_EVERY = 8

export function createSimulator () {
  const subject = [0, 1.2, -RADIUS]
  // A clock of its own rather than the wall one: `stepBy` can then replay a whole
  // capture in a few milliseconds, which is how this gets tested without a phone.
  let elapsed = 0
  let last = null
  let stopped = false
  let grabs = 0
  let frameHandler = null
  const listeners = []

  const poseAt = (time) => {
    const azimuth = (time * DEGREES_PER_SECOND / 1000) * Math.PI / 180
    // elevation wanders slowly across both rings, so the walker does not spend the whole
    // simulation covering the same band
    const elevation = (16 + 14 * Math.sin(time / 7000)) * Math.PI / 180
    const position = [
      subject[0] + RADIUS * Math.cos(elevation) * Math.sin(azimuth),
      subject[1] + RADIUS * Math.sin(elevation),
      subject[2] + RADIUS * Math.cos(elevation) * Math.cos(azimuth)
    ]
    return { position, orientation: lookAt(position, subject) }
  }

  return {
    hasCamera: true,

    onEnd (handler) {
      listeners.push(handler)
    },

    end () {
      stopped = true
      for (const handler of listeners) handler()
    },

    run (onFrame) {
      frameHandler = onFrame
      const step = (now) => {
        if (stopped) return
        window.requestAnimationFrame(step)
        elapsed += last === null ? 16 : Math.min(100, now - last)
        last = now
        this.emit()
      }
      window.requestAnimationFrame(step)
    },

    // One frame, `ms` of simulated time later. Driving the simulation by hand is the
    // only way to exercise it where requestAnimationFrame never fires: a hidden tab, a
    // headless browser, an automated check.
    stepBy (ms = 33) {
      elapsed += ms
      this.emit()
    },

    emit () {
      if (!frameHandler) return
      const { position, orientation } = poseAt(elapsed)
      frameHandler({ position, orientation, matrix: null, emulated: true }, { simulated: true })
    },

    hitDistance () {
      return { position: subject }
    },

    intrinsics () {
      return { width: 960, height: 720, fx: 720, fy: 720, cx: 480, cy: 360, projection: null }
    },

    grab (canvas) {
      grabs++
      canvas.width = 960
      canvas.height = 720
      const context = canvas.getContext('2d')

      context.filter = 'none'
      const sky = context.createLinearGradient(0, 0, 0, canvas.height)
      sky.addColorStop(0, '#2b3a55')
      sky.addColorStop(1, '#8a6f4e')
      context.fillStyle = sky
      context.fillRect(0, 0, canvas.width, canvas.height)

      // Texture, so the sharpness measure has corners to lose when the frame is blurred
      const seed = Math.floor(elapsed / 40)
      for (let i = 0; i < 900; i++) {
        const x = (Math.sin(seed + i * 12.9898) * 43758.5453 % 1 + 1) % 1
        const y = (Math.sin(seed + i * 78.233) * 43758.5453 % 1 + 1) % 1
        context.fillStyle = i % 3 ? '#d9c9a3' : '#3c4a63'
        context.fillRect(x * canvas.width, y * canvas.height, 6, 6)
      }

      // Every eighth frame is a smeared one: the blur floor should throw these away
      if (grabs % BLUR_EVERY === 0) {
        const copy = context.getImageData(0, 0, canvas.width, canvas.height)
        context.putImageData(copy, 0, 0)
        context.filter = 'blur(6px)'
        context.drawImage(canvas, 0, 0)
        context.filter = 'none'
      }

      context.fillStyle = '#e8ecf2'
      context.font = '28px monospace'
      context.fillText(`simulated frame ${grabs}`, 24, 44)
      return canvas
    }
  }
}

function lookAt (from, target) {
  const forward = normalize([target[0] - from[0], target[1] - from[1], target[2] - from[2]])
  // A camera looks down its own -Z, so its +Z is the reverse of the viewing direction
  const z = [-forward[0], -forward[1], -forward[2]]
  const x = normalize(cross([0, 1, 0], z))
  const y = cross(z, x)
  return quaternionFromAxes(x, y, z)
}

function cross (a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

function normalize (a) {
  const size = Math.hypot(a[0], a[1], a[2]) || 1
  return [a[0] / size, a[1] / size, a[2] / size]
}

function quaternionFromAxes (x, y, z) {
  const trace = x[0] + y[1] + z[2]
  if (trace > 0) {
    const s = Math.sqrt(trace + 1) * 2
    return { x: (y[2] - z[1]) / s, y: (z[0] - x[2]) / s, z: (x[1] - y[0]) / s, w: 0.25 * s }
  }
  if (x[0] > y[1] && x[0] > z[2]) {
    const s = Math.sqrt(1 + x[0] - y[1] - z[2]) * 2
    return { x: 0.25 * s, y: (y[0] + x[1]) / s, z: (z[0] + x[2]) / s, w: (y[2] - z[1]) / s }
  }
  if (y[1] > z[2]) {
    const s = Math.sqrt(1 + y[1] - x[0] - z[2]) * 2
    return { x: (y[0] + x[1]) / s, y: 0.25 * s, z: (z[1] + y[2]) / s, w: (z[0] - x[2]) / s }
  }
  const s = Math.sqrt(1 + z[2] - x[0] - y[1]) * 2
  return { x: (z[0] + x[2]) / s, y: (z[1] + y[2]) / s, z: 0.25 * s, w: (x[1] - y[0]) / s }
}
