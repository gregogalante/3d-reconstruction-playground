// Wiring: capabilities, screens, the frame loop and the two degraded paths below it.
//
// There are three ways this page can capture, and they all funnel into the same `shoot`:
//   - WebXR with camera access, the one worth having, guided by the ARCore pose
//   - a plain camera through getUserMedia, blind, when WebXR is missing
//   - a simulator (?simulate=1) that fabricates poses and frames, so the guidance can be
//     exercised on a desktop without a phone in the loop
// Keeping them behind one shutter is what stops the fallbacks from rotting unnoticed.

import * as api from './lib/api.js'
import { capabilities, startSession, sharpnessOf } from './lib/xr.js'
import { advise, planOrbit, recordShot, coveredTargets } from './lib/guidance.js'
import { drawRadar, drawCoverageRing, drawShotProgress } from './lib/radar.js'
import { forwardOf, angleBetween } from './lib/vec.js'
import { createSimulator } from './lib/simulate.js'

const el = id => document.getElementById(id)
const SIMULATE = new window.URLSearchParams(window.location.search).has('simulate')

// A frame darker or flatter than this is a pocket, a lens cap or a smear.
const ABSOLUTE_SHARPNESS = 8
// Blur is relative: a white wall scores low and is still in focus, so the floor follows
// what this capture has been seeing rather than an absolute number.
const RELATIVE_SHARPNESS = 0.55
const SHARPNESS_MEMORY = 12
// How long a JPEG encode is given before the frame is written off, see `shoot`.
const ENCODE_TIMEOUT = 1500
// Longest side of an uploaded frame. pipeline.py resizes everything to
// IMAGE_MAX_DIMENSION (1024) anyway, so encoding the full 1920 spends the shutter's time
// on pixels that get thrown away — and the shutter's time is what opens holes in a
// capture. This keeps some headroom over the cap and still halves the work.
const UPLOAD_MAX_DIMENSION = 1440
// Corners per frame under which a scene has nothing for the matcher. The server's own
// number, kept here to decide when to say so; see FEATURES_STARVED in capture_server.py.
const FEATURES_STARVED = 750

const state = {
  mode: 'orbit',
  dataset: '',
  server: null,
  uploader: null,
  device: null,
  plan: null,
  shots: [],
  scores: [],
  lastCapture: 0,
  lastRadar: 0,
  hit: null,
  depth: null,
  rejected: 0,
  video: null,
  interval: null,
  milestone: 0,
  fov: null,
  features: []
}

const scratch = el('scratch')
const work = document.createElement('canvas')
const reduced = document.createElement('canvas')

// ----------------------------------------------------------------------------
// SETUP
// ----------------------------------------------------------------------------

function show (id) {
  for (const section of document.querySelectorAll('.screen, .hud')) section.hidden = section.id !== id
}

async function checkCapabilities () {
  const list = el('capabilities')
  const report = await capabilities()
  const rows = [
    ['Secure connection', report.secure, 'without it the browser hands out neither camera nor AR'],
    ['WebXR', report.webxr, 'Chrome on Android, with Google Play Services for AR installed'],
    ['AR tracking', report.ar, 'ARCore: this is what guides the capture']
  ]
  list.innerHTML = ''
  for (const [name, ok, why] of rows) {
    const item = document.createElement('li')
    item.className = ok ? 'yes' : 'no'
    item.innerHTML = `<span><b>${name}</b> — ${ok ? 'ready' : why}</span>`
    list.appendChild(item)
  }

  const start = el('start')
  start.disabled = false
  if (SIMULATE) {
    start.textContent = 'Start simulated capture'
  } else if (report.ar) {
    start.textContent = 'Start guided capture'
  } else {
    start.textContent = 'Start without guidance'
    const note = document.createElement('li')
    note.className = 'no'
    // On Android the usual cause is one missing app, and saying so beats a shrug
    note.innerHTML = /android/i.test(navigator.userAgent)
      ? '<span>Install <a href="https://play.google.com/store/apps/details?id=com.google.ar.core">Google Play Services for AR</a> to get the guidance, or capture without it</span>'
      : '<span>Falling back to a plain camera: shots are yours to place</span>'
    list.appendChild(note)
  }
  state.device = { userAgent: navigator.userAgent, ...report }
}

async function warnAboutDataset () {
  const name = el('dataset').value.trim()
  const note = el('dataset-note')
  if (!name) return (note.textContent = '')
  try {
    const datasets = await api.listDatasets()
    const existing = datasets.find(dataset => dataset.name === name)
    note.textContent = existing
      ? `${name} already holds ${existing.photos} photos — this capture is added to them`
      : `New dataset, frames land in storage/datasets/${name}/train/`
  } catch {
    note.textContent = ''
  }
}

// ----------------------------------------------------------------------------
// CAPTURE
// ----------------------------------------------------------------------------

// How blurred a frame may be before it is thrown away — and it gets more forgiving the
// longer the chain has been waiting for one.
//
// Rejecting a frame is only free while the next one is coming. On the second real
// capture the breaks were not fast swings outrunning the shutter: on the six pairs that
// ended up sharing no view, 2.84 seconds passed between frames against 0.55 when things
// went well, at ordinary turning speed. Something refused every frame through the turn,
// and by the time one was accepted the view had moved 34 to 80 degrees. A frame blurred
// by the same turn would have joined those two ends together; the hole cannot be joined
// by anything.
function sharpnessFloor (pendingTurn, limit) {
  if (state.scores.length < 5) return ABSOLUTE_SHARPNESS
  const sorted = [...state.scores].sort((a, b) => a - b)
  const median = sorted[Math.floor(sorted.length / 2)]
  const floor = Math.max(ABSOLUTE_SHARPNESS, median * RELATIVE_SHARPNESS)

  // Past the overlap limit the frame is holding the chain together, so the bar drops to
  // the floor of "is this an image at all" and lets it through.
  if (!limit || !pendingTurn) return floor
  const urgency = Math.min(1, Math.max(0, (pendingTurn - limit) / limit))
  return floor * (1 - urgency) + ABSOLUTE_SHARPNESS * urgency
}

async function shoot (canvas, viewer, extra = {}) {
  // Encoding a frame takes longer than a frame lasts, and the gate that opened this shot
  // stays open until the shot is recorded: without both of these the loop fires twice
  // from the same viewpoint and the dataset gets a duplicate for every capture.
  if (state.capturing) return false
  state.capturing = true
  state.lastCapture = window.performance.now()

  try {
    const score = sharpnessOf(canvas, work)
    const floor = sharpnessFloor(extra.pending_turn, extra.turn_limit)
    if (score < floor) {
      state.rejected++
      flash('Too blurry — hold steadier')
      return false
    }

    state.scores.push(score)
    if (state.scores.length > SHARPNESS_MEMORY) state.scores.shift()

    // toBlob hands the encode to another thread and there is no guarantee it comes back:
    // a browser that defers it — a backgrounded tab is enough — would leave this await
    // pending for good, and with it the guard above, so the capture would stop dead
    // without saying anything. Losing one frame to a stalled encoder is the better half
    // of that trade.
    const source = shrink(canvas)
    const blob = await Promise.race([
      new Promise(resolve => source.toBlob(resolve, 'image/jpeg', 0.92)),
      new Promise(resolve => window.setTimeout(() => resolve(null), ENCODE_TIMEOUT))
    ])
    if (!blob) {
      flash('The encoder stalled — frame skipped')
      return false
    }

    state.uploader.push(blob, {
      mode: state.mode,
      sharpness_client: Math.round(score * 10) / 10,
      subject: state.plan ? state.plan.centre : null,
      orbit_radius: state.plan ? state.plan.radius : null,
      ...extra
    })

    if (viewer) state.shots.push(recordShot(state.plan, viewer))
    el('hud-count').textContent = state.shots.length || state.uploader.counts.uploaded + state.uploader.counts.pending
    showThumbnail(blob)
    buzz(30)
    return true
  } finally {
    state.capturing = false
  }
}

// A copy no larger than the pipeline will keep, drawn once into a canvas that is reused.
function shrink (canvas) {
  const scale = UPLOAD_MAX_DIMENSION / Math.max(canvas.width, canvas.height)
  if (scale >= 1) return canvas
  reduced.width = Math.round(canvas.width * scale)
  reduced.height = Math.round(canvas.height * scale)
  reduced.getContext('2d').drawImage(canvas, 0, 0, reduced.width, reduced.height)
  return reduced
}

function showThumbnail (blob) {
  const thumb = el('thumb')
  if (thumb.src) window.URL.revokeObjectURL(thumb.src)
  thumb.src = window.URL.createObjectURL(blob)
  thumb.hidden = false
  el('undo').hidden = false
}

// Two things write to the same band: a flash, which is a one off, and the guidance,
// which asserts a warning for as long as it holds. The deadline is what keeps them from
// erasing each other — a flash owns the band until it expires, the guidance after that.
let flashUntil = 0
function flash (message) {
  const warning = el('warning')
  warning.textContent = message
  warning.hidden = false
  flashUntil = window.performance.now() + 1600
}

function showWarning (message) {
  const warning = el('warning')
  if (window.performance.now() < flashUntil) return
  warning.hidden = !message
  if (message) warning.textContent = message
}

// ----------------------------------------------------------------------------
// FEEDBACK YOU DO NOT HAVE TO LOOK AT
// ----------------------------------------------------------------------------

// Walking around a thing, you are watching the thing, not the phone, so a shot announces
// itself through the vibration motor: one buzz a frame, a triple at each quarter of the
// orbit. Silent on purpose — a capture happens in rooms with other people in them.
const MILESTONES = [0.25, 0.5, 0.75]

function buzz (pattern) {
  if (navigator.vibrate) navigator.vibrate(pattern)
}

// The other way a frame is useless, and the one nothing on the phone can see: a white
// wall is in perfect focus and has nothing to match. The server counts the corners it
// finds and this says so, once, while there is still time to aim elsewhere.
function noteTexture (count) {
  if (typeof count !== 'number') return
  state.features.push(count)
  if (state.features.length > SHARPNESS_MEMORY) state.features.shift()
  if (state.features.length < 5 || state.texturedWarned) return

  const sorted = [...state.features].sort((a, b) => a - b)
  if (sorted[Math.floor(sorted.length / 2)] >= FEATURES_STARVED) return
  state.texturedWarned = true
  flash('Almost nothing to match here — take in more of the room')
}

// ----------------------------------------------------------------------------
// GUIDED SESSION
// ----------------------------------------------------------------------------

async function runGuided () {
  // The overlay has to be in the DOM and visible before the session asks for it: it is
  // the dom-overlay root, the only markup the compositor will show over the camera.
  show('hud')
  const overlay = el('hud')
  overlay.addEventListener('beforexrselect', event => event.preventDefault())

  const session = SIMULATE ? createSimulator() : await startSession({ overlay })
  state.xr = session
  session.onEnd(() => finish({ ended: true }))
  // Driving the simulation by hand from the console is how it gets checked where
  // requestAnimationFrame never runs, so the handle is published in that mode only.
  if (SIMULATE) window.captureDebug = { session, state, shoot }

  el('subject-prompt').hidden = state.mode !== 'orbit'
  el('radar').hidden = state.mode === 'orbit'

  session.run((pose, frame) => {
    if (!pose) return
    const viewer = { position: pose.position, forward: forwardOf(pose.orientation) }
    state.viewer = viewer
    state.pose = pose

    // The hit test keeps running in walk mode: how far the wall in front is decides how
    // far a step may be, and a step measured in metres means nothing without it.
    if (state.mode === 'walk' || !state.plan) {
      state.hit = session.hitDistance ? session.hitDistance(frame) : null
      state.depth = state.hit
        ? Math.hypot(...state.hit.position.map((value, axis) => value - viewer.position[axis]))
        : null
    }
    if (state.mode === 'orbit' && !state.plan) {
      const prompt = el('subject-prompt').querySelector('p')
      prompt.textContent = state.hit
        ? `Subject ${distanceTo(state.hit.position, viewer.position)} away. Set it and start orbiting.`
        : 'Point at the middle of your subject from where you want to orbit, then set it.'
    }

    // The lens is the thing the overlap rule is written against, and only the session
    // knows how wide it is: a phone in portrait sees a third of what a guess assumes.
    if (state.fov == null && session.intrinsics) {
      const intrinsics = session.intrinsics()
      if (intrinsics) state.fov = 2 * Math.atan(intrinsics.width / (2 * intrinsics.fx)) * 180 / Math.PI
    }

    const advice = advise({
      mode: state.mode,
      plan: state.plan,
      viewer,
      shots: state.shots,
      fov: state.fov,
      depth: state.depth,
      seconds: (window.performance.now() - state.lastCapture) / 1000
    })
    paint(advice, session)

    if (advice.capture && el('auto').checked) {
      const canvas = session.grab(scratch)
      if (canvas) shoot(canvas, viewer, meta(session, advice, viewer))
    }
  })
}

function distanceTo (a, b) {
  const metres = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])
  return metres < 1 ? `${Math.round(metres * 100)} cm` : `${metres.toFixed(1)} m`
}

function meta (session, advice, viewer) {
  const last = state.shots[state.shots.length - 1]
  return {
    tracking: SIMULATE ? 'simulated' : 'webxr',
    pose: state.pose,
    fov: state.fov,
    intrinsics: scaleIntrinsics(session.intrinsics ? session.intrinsics() : null),
    target: advice && advice.target ? { azimuth: advice.target.azimuth, elevation: advice.target.elevation } : null,
    // how far the view had already swung from the last frame that was kept: the blur bar
    // is relaxed against this, and it is worth having in the manifest afterwards
    pending_turn: last && viewer ? Math.round(angleBetween(viewer.forward, last.forward) * 10) / 10 : 0,
    turn_limit: advice && advice.limits ? Math.round(advice.limits.turn * 10) / 10 : null,
    rejected_before: state.rejected
  }
}

// The intrinsics describe the camera image, so shrinking that image moves them too. The
// raw projection matrix travels alongside untouched, being independent of the size.
function scaleIntrinsics (intrinsics) {
  if (!intrinsics) return null
  const scale = Math.min(1, UPLOAD_MAX_DIMENSION / Math.max(intrinsics.width, intrinsics.height))
  if (scale >= 1) return intrinsics
  return {
    ...intrinsics,
    width: Math.round(intrinsics.width * scale),
    height: Math.round(intrinsics.height * scale),
    fx: intrinsics.fx * scale,
    fy: intrinsics.fy * scale,
    cx: intrinsics.cx * scale,
    cy: intrinsics.cy * scale,
    uploaded_scale: scale
  }
}

function paint (advice, session) {
  el('guidance').textContent = advice.text
  showWarning(advice.warning)

  const arrow = el('arrow')
  if (advice.arrow) {
    arrow.hidden = false
    arrow.style.transform = `rotate(${advice.arrow.angle}deg)`
    arrow.classList.toggle('behind', advice.arrow.behind)
  } else {
    arrow.hidden = true
  }

  el('shutter').classList.toggle('armed', advice.capture)
  if (state.plan) {
    el('hud-count').textContent = `${state.shots.length} · ${Math.round(advice.progress * 100)}%`
  }

  // Orbiting, the number that matters is how far you are from the subject. Walking, it is
  // how far the wall in front is, since that is what sets the length of a step — the
  // distance since the last frame is already the dial's job.
  const distance = el('hud-distance')
  const metres = state.mode === 'orbit' ? advice.distance : state.depth
  distance.hidden = metres == null
  if (metres != null) {
    distance.textContent = `${metres.toFixed(1)} m`
    distance.style.color = ['close', 'far'].includes(advice.cue) ? 'var(--warn)' : ''
  }

  announceMilestone(advice.progress)

  const now = window.performance.now()
  if (now - state.lastRadar > 100) {
    state.lastRadar = now
    // The dial says something in both modes: coverage of the orbit where there is one,
    // and how close the next frame is where there is not. The trail beside it is the map
    // a walk has instead of a ring of targets.
    if (state.mode === 'orbit') {
      drawCoverageRing(el('ring'), { plan: state.plan, shots: state.shots, viewer: state.viewer })
    } else {
      drawShotProgress(el('ring'), { readiness: advice.readiness || 0, warning: Boolean(advice.warning) })
      drawRadar(el('radar'), { plan: state.plan, shots: state.shots, viewer: state.viewer })
    }
  }

  if (session && !session.hasCamera && !SIMULATE && !state.warnedCamera) {
    state.warnedCamera = true
    flash('This session tracks but will not hand over frames: no camera-access')
  }
}

// Crossing a quarter of the orbit is worth knowing about without reading anything, and
// it is the only moment a capture has that feels like progress.
function announceMilestone (progress) {
  if (!progress) return
  const reached = MILESTONES.filter(fraction => progress >= fraction).length
  if (reached <= state.milestone) return
  state.milestone = reached
  buzz([25, 70, 25])
}

function setSubject () {
  const viewer = state.viewer
  if (!viewer) return
  // The hit test knows the real distance; without one, an arm's length plus a bit is the
  // least wrong guess for a thing you are about to walk around.
  const radius = state.hit
    ? Math.hypot(...state.hit.position.map((value, axis) => value - viewer.position[axis]))
    : 1.5
  state.plan = planOrbit(viewer, Math.max(0.4, radius))
  el('subject-prompt').hidden = true
}

// ----------------------------------------------------------------------------
// PLAIN CAMERA FALLBACK
// ----------------------------------------------------------------------------

async function runPlain () {
  show('plain')
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1080 } },
    audio: false
  })
  const video = el('video')
  video.srcObject = stream
  state.video = video
  await video.play()

  el('plain-guidance').textContent =
    'No AR here: take a shot every step or two, keep the subject in view, never spin on the spot.'
}

async function shootPlain () {
  const video = state.video
  if (!video || !video.videoWidth) return
  scratch.width = video.videoWidth
  scratch.height = video.videoHeight
  scratch.getContext('2d').drawImage(video, 0, 0)
  const taken = await shoot(scratch, null, { tracking: 'none' })
  if (taken) el('plain-count').textContent = state.uploader.counts.uploaded + state.uploader.counts.pending
}

// ----------------------------------------------------------------------------
// SESSION LIFECYCLE
// ----------------------------------------------------------------------------

// A capture is minutes of walking with the phone held up and nothing touching the glass,
// which is exactly how a screen decides to sleep. An immersive session keeps itself
// awake; the plain fallback does not, so ask. Failure here is not worth a message.
async function keepAwake () {
  try {
    state.wakeLock = await navigator.wakeLock.request('screen')
  } catch {
    state.wakeLock = null
  }
}

async function start () {
  const dataset = el('dataset').value.trim()
  if (!dataset) return fail('Give the dataset a name first')

  el('start').disabled = true
  try {
    state.dataset = dataset
    state.server = await api.openSession({
      dataset,
      mode: state.mode,
      device: state.device,
      tracking: SIMULATE ? 'simulated' : (state.device && state.device.ar ? 'webxr' : 'none')
    })
    state.uploader = api.createUploader(state.server.session, {
      onChange: counts => {
        const queue = el('hud-queue')
        queue.hidden = counts.pending === 0 && counts.failed === 0
        queue.textContent = counts.failed ? `${counts.pending}↑ ${counts.failed}✗` : `${counts.pending}↑`
      },
      onStored: stored => noteTexture(stored.features)
    })
    el('hud-dataset').textContent = state.server.dataset
    keepAwake()

    if (SIMULATE || (state.device && state.device.ar)) {
      await runGuided()
    } else {
      await runPlain()
    }
  } catch (error) {
    el('start').disabled = false
    fail(error.message || String(error))
  }
}

function fail (message) {
  const box = el('setup-error')
  box.textContent = message
  box.hidden = false
  show('setup')
}

// Better to be told now than after a pipeline run: the two ways a capture comes back
// unusable are too few frames and one side of the subject done twice.
function verdict (frames) {
  const covered = state.plan
    ? coveredTargets(state.plan, state.shots).filter(Boolean).length / state.plan.targets.length
    : null
  if (frames < 12) return `${frames} frames is thin — expect a partial reconstruction, or none`
  if (covered !== null && covered < 0.6) {
    return `Only ${Math.round(covered * 100)}% of the way around: the far side will be missing`
  }
  if (covered !== null && covered < 0.95) {
    return `${Math.round(covered * 100)}% covered — good, with gaps where you did not walk`
  }
  return 'Looks like a complete capture'
}

// The expensive mistake is walking away from a capture that cannot reconstruct, and
// finding out after a pipeline run. So a thin one is argued with, once, while the
// subject is still in front of you — unless the session itself is ending, where there is
// nothing left to go back to.
function tooThinToLeave () {
  if (state.shots.length < 12) {
    return `Only ${state.shots.length} frames so far. That is unlikely to reconstruct.`
  }
  if (!state.plan) return null
  const covered = coveredTargets(state.plan, state.shots).filter(Boolean).length / state.plan.targets.length
  return covered < 0.6
    ? `Only ${Math.round(covered * 100)}% of the way around. The far side will be missing.`
    : null
}

function askBeforeFinishing (message) {
  el('confirm-text').textContent = message
  el('confirm-finish').hidden = false
  buzz([40, 80, 40])
  return new Promise(resolve => {
    el('keep-going').onclick = () => {
      el('confirm-finish').hidden = true
      resolve(false)
    }
    el('finish-anyway').onclick = () => {
      el('confirm-finish').hidden = true
      resolve(true)
    }
  })
}

async function finish ({ ended = false, forced = false } = {}) {
  if (!state.server) return

  const thin = ended || forced ? null : tooThinToLeave()
  if (thin && !await askBeforeFinishing(thin)) return

  const server = state.server
  state.server = null

  if (state.interval) window.clearInterval(state.interval)
  if (state.video && state.video.srcObject) {
    // let the camera light go out the moment the capture is over
    for (const track of state.video.srcObject.getTracks()) track.stop()
  }
  if (state.xr && !ended) state.xr.end()
  if (state.wakeLock) state.wakeLock.release().catch(() => {})

  el('guidance').textContent = 'Uploading the last frames…'
  await state.uploader.drain()
  const summary = await api.finishSession(server.session, { rejected: state.rejected })

  // the server has the poses and the frames, so its reading beats the one the phone can
  // make from shot count alone
  el('summary-verdict').textContent = summary.diagnosis && summary.diagnosis.verdict
    ? summary.diagnosis.verdict
    : verdict(summary.frames)

  const stats = el('summary-stats')
  stats.innerHTML = ''
  const counts = state.uploader.counts
  const rows = [
    ['Dataset', summary.dataset],
    ['Frames kept', summary.frames],
    ['Rejected as blurry', state.rejected],
    ['Failed to upload', counts.failed],
    ['Sharpness, median', summary.sharpness.median ?? '—'],
    ['Texture, median corners', summary.diagnosis?.features?.median ?? '—'],
    ['Frames sharing no view', summary.diagnosis?.turns?.sharing_nothing ?? '—']
  ]
  for (const [name, value] of rows) {
    const term = document.createElement('dt')
    term.textContent = name
    const definition = document.createElement('dd')
    definition.textContent = value
    stats.append(term, definition)
  }
  el('summary-command').textContent = summary.command
  show('summary')
}

// ----------------------------------------------------------------------------
// EVENTS
// ----------------------------------------------------------------------------

el('mode').addEventListener('click', event => {
  const button = event.target.closest('button')
  if (!button) return
  state.mode = button.dataset.mode
  for (const other of el('mode').children) other.classList.toggle('selected', other === button)
})

el('dataset').addEventListener('change', warnAboutDataset)
el('dataset').addEventListener('blur', warnAboutDataset)
el('start').addEventListener('click', start)
el('set-subject').addEventListener('click', setSubject)
el('finish').addEventListener('click', () => finish())
el('plain-finish').addEventListener('click', () => finish())
el('plain-shutter').addEventListener('click', shootPlain)
el('shutter').addEventListener('click', () => {
  if (!state.xr) return
  const canvas = state.xr.grab(scratch)
  if (canvas) shoot(canvas, state.viewer, meta(state.xr, null))
})
el('undo').addEventListener('click', async () => {
  try {
    await api.undoLast(state.server.session)
    state.shots.pop()
    el('thumb').hidden = true
    el('undo').hidden = true
  } catch (error) {
    flash(error.message)
  }
})
el('plain-auto').addEventListener('change', event => {
  window.clearInterval(state.interval)
  if (event.target.checked) state.interval = window.setInterval(shootPlain, 2000)
})
el('again').addEventListener('click', () => window.location.reload())

// Walking out of a capture by swiping back loses the frames still in the queue, and the
// session stays open on the server holding a half written manifest.
window.addEventListener('beforeunload', event => {
  if (!state.server) return
  event.preventDefault()
  event.returnValue = ''
})

checkCapabilities()
