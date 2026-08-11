// The WebXR side: an immersive AR session for the pose, raw camera access for the pixels.
//
// On Android this is ARCore underneath, which is what makes the guidance possible at all:
// a plain camera page knows nothing about where it is. Two things are asked for and both
// may be refused, so every caller has to cope with a session that tracks but cannot hand
// over frames.

const SHARPNESS_WIDTH = 320 // the width pipeline.py scores frames at, kept the same here

export async function capabilities () {
  const report = { webxr: false, ar: false, secure: window.isSecureContext }
  if (!navigator.xr) return report
  report.webxr = true
  try {
    report.ar = await navigator.xr.isSessionSupported('immersive-ar')
  } catch {
    report.ar = false
  }
  return report
}

export async function startSession ({ overlay }) {
  const canvas = document.createElement('canvas')
  const gl = canvas.getContext('webgl2', { xrCompatible: true, alpha: true })
  if (!gl) throw new Error('This browser has no WebGL2, which WebXR needs')

  // camera-access is optional on purpose: without it the session still tracks, and a
  // guided capture that cannot photograph is worth saying out loud rather than failing.
  const session = await navigator.xr.requestSession('immersive-ar', {
    optionalFeatures: ['camera-access', 'dom-overlay', 'hit-test', 'local-floor'],
    domOverlay: { root: overlay }
  })

  await gl.makeXRCompatible()
  session.updateRenderState({ baseLayer: new window.XRWebGLLayer(session, gl) })

  // local-floor puts the origin on the ground, which makes the radar readable; local is
  // the fallback and only shifts the origin, every distance in the guidance is relative.
  let space
  try {
    space = await session.requestReferenceSpace('local-floor')
  } catch {
    space = await session.requestReferenceSpace('local')
  }

  let hitTest = null
  try {
    const viewerSpace = await session.requestReferenceSpace('viewer')
    hitTest = await session.requestHitTestSource({ space: viewerSpace })
  } catch {
    hitTest = null // no plane detection: the subject distance falls back to a guess
  }

  const binding = new window.XRWebGLBinding(session, gl)
  const framebuffer = gl.createFramebuffer()
  return new Session({ session, gl, space, binding, framebuffer, hitTest })
}

class Session {
  constructor (parts) {
    Object.assign(this, parts)
    this.hasCamera = false
  }

  onEnd (handler) {
    this.session.addEventListener('end', handler)
  }

  end () {
    try {
      this.session.end()
    } catch {
      // already gone, which is the state the caller wanted anyway
    }
  }

  // The frame loop hands back a plain object per frame so the app never has to hold a
  // reference to an XRFrame, which is only valid inside its own callback.
  run (onFrame) {
    const step = (time, frame) => {
      this.session.requestAnimationFrame(step)
      const pose = frame.getViewerPose(this.space)
      if (!pose) return onFrame(null, frame)

      const layer = this.session.renderState.baseLayer
      this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, layer.framebuffer)
      // the camera feed is composited behind us, so the layer is cleared to transparent
      this.gl.clearColor(0, 0, 0, 0)
      this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT)

      const view = pose.views[0]
      this.view = view
      this.hasCamera = Boolean(view.camera)
      onFrame({
        position: [pose.transform.position.x, pose.transform.position.y, pose.transform.position.z],
        orientation: pose.transform.orientation,
        matrix: Array.from(pose.transform.matrix),
        emulated: pose.emulatedPosition
      }, frame)
    }
    this.session.requestAnimationFrame(step)
  }

  // Distance to whatever the middle of the screen is pointing at, when the device can
  // tell. Beats asking the person to estimate the radius of their own orbit.
  hitDistance (frame) {
    if (!this.hitTest) return null
    const results = frame.getHitTestResults(this.hitTest)
    if (!results.length) return null
    const pose = results[0].getPose(this.space)
    if (!pose) return null
    return { position: [pose.transform.position.x, pose.transform.position.y, pose.transform.position.z] }
  }

  intrinsics () {
    const camera = this.view && this.view.camera
    if (!camera) return null
    // Derived from the projection matrix against the camera image size, which assumes
    // Chrome hands out an image aligned with the view frustum. The raw matrix travels
    // with it so a later conversion can be checked rather than trusted.
    const p = this.view.projectionMatrix
    return {
      width: camera.width,
      height: camera.height,
      fx: p[0] * camera.width / 2,
      fy: p[5] * camera.height / 2,
      cx: camera.width * (1 - p[8]) / 2,
      cy: camera.height * (1 + p[9]) / 2,
      projection: Array.from(p)
    }
  }

  // Copy the camera image out of the GPU into a canvas. Only valid inside the frame
  // callback that produced the view, hence the synchronous read.
  grab (scratch) {
    const camera = this.view && this.view.camera
    if (!camera) return null

    const texture = this.binding.getCameraImage(camera)
    if (!texture) return null

    const { gl } = this
    const { width, height } = camera
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer)
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0)
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.session.renderState.baseLayer.framebuffer)
      return null
    }
    const pixels = new Uint8Array(width * height * 4)
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels)
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.session.renderState.baseLayer.framebuffer)

    return drawFlipped(scratch, pixels, width, height)
  }
}

// GL reads rows from the bottom up and a canvas expects them from the top down, so the
// copy walks the rows backwards. One pass, no second canvas.
function drawFlipped (canvas, pixels, width, height) {
  canvas.width = width
  canvas.height = height
  const context = canvas.getContext('2d')
  const image = context.createImageData(width, height)
  const row = width * 4
  for (let y = 0; y < height; y++) {
    image.data.set(pixels.subarray((height - 1 - y) * row, (height - y) * row), y * row)
  }
  context.putImageData(image, 0, 0)
  return canvas
}

// Variance of the Laplacian, the measure pipeline.py uses to pick frames out of a video.
// Same shape of number, so a threshold learned in one place means something in the other.
export function sharpnessOf (canvas, work) {
  const width = SHARPNESS_WIDTH
  const height = Math.round(width * canvas.height / canvas.width)
  work.width = width
  work.height = height
  const context = work.getContext('2d', { willReadFrequently: true })
  context.drawImage(canvas, 0, 0, width, height)
  const { data } = context.getImageData(0, 0, width, height)

  const grey = new Float32Array(width * height)
  for (let i = 0; i < grey.length; i++) {
    grey[i] = 0.114 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.299 * data[i * 4 + 2]
  }

  let sum = 0
  let squares = 0
  let count = 0
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x
      const value = grey[i - width] + grey[i + width] + grey[i - 1] + grey[i + 1] - 4 * grey[i]
      sum += value
      squares += value * value
      count++
    }
  }
  const mean = sum / count
  return squares / count - mean * mean
}
