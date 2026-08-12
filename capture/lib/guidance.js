// What to tell the person holding the phone, and when to take the shot.
//
// The rules encode the way a capture fails. A dataset dies of three things: shots taken
// from one spot while turning (a panorama has no parallax, so nothing triangulates),
// gaps too wide for two views to share features, and blur. So the capture gate is not a
// timer, it is a geometric one — take a frame when the viewpoint has actually changed —
// and the advice is about where to put your feet next.
//
// No DOM in here on purpose: this is the part worth testing, and it runs in node.

import { sub, add, scale, dot, cross, distance, normalize, length, angleBetween } from './vec.js'

export const ORBIT = {
  targetTolerance: 18, // how close a shot has to pass for a guidance target to count
  offAxisLimit: 38, // degrees the subject may sit off the middle of the frame
  minRadiusRatio: 0.55, // fraction of the set radius before you are too close
  maxRadiusRatio: 2.2,
  minSeconds: 0.3
}

export const WALK = {
  spinTravel: 0.12, // turning this much without moving is the panorama failure
  minSeconds: 0.2 // only there to stop one motion producing two identical frames
}

// How much of the lens two neighbouring shots have to share. Dense matching wants
// something like three quarters, so the turn between them is capped at a quarter of the
// field of view.
const OVERLAP_TURN = 0.25
// And the sideways step is capped against how far the subject is: the old photogrammetry
// guideline of a baseline no longer than an eighth of the depth.
const BASELINE_TO_DEPTH = 8
const DEFAULT_FOV = 60
const DEFAULT_DEPTH = 1.5

const clamp = (value, low, high) => Math.min(high, Math.max(low, value))

// The thresholds are not constants because the lens is not one. A phone in portrait sees
// 34 degrees across, where a fixed 25 degree rule leaves a quarter of the frame shared —
// measured on a real capture, 35 of 67 neighbouring pairs turned more than half the lens
// and 7 shared nothing at all, and COLMAP registered 16 of 68 frames.
export function limitsFor ({ fov = DEFAULT_FOV, depth = null } = {}) {
  return {
    turn: clamp(fov * OVERLAP_TURN, 4, 15),
    baseline: clamp((depth || DEFAULT_DEPTH) / BASELINE_TO_DEPTH, 0.06, 0.5),
    fov
  }
}

const UP = [0, 1, 0]

export function elevationOf (direction) {
  return Math.asin(Math.min(1, Math.max(-1, direction[1]))) * 180 / Math.PI
}

// Where a target sits relative to the person: ahead of them, to their right, above them.
// Screen directions come out of this and nothing else, so the arrow cannot disagree with
// the words next to it.
export function relativeTo (viewer, position) {
  const right = normalize(cross(viewer.forward, UP))
  const to = sub(position, viewer.position)
  return {
    ahead: dot(to, viewer.forward),
    lateral: dot(to, right),
    vertical: dot(to, UP),
    distance: length(to)
  }
}

export function planOrbit (viewer, radius, { rings = [8, 28], perRing = 12 } = {}) {
  // The subject sits where you are looking, at the distance you chose; the rings are the
  // orbits you should walk. Two heights, because one ring reconstructs a band and leaves
  // the top of the object as a guess.
  const centre = add(viewer.position, scale(normalize(viewer.forward), radius))
  const targets = []
  for (const elevation of rings) {
    for (let step = 0; step < perRing; step++) {
      const azimuth = (360 / perRing) * step
      const phi = elevation * Math.PI / 180
      const theta = azimuth * Math.PI / 180
      const direction = [
        Math.cos(phi) * Math.cos(theta),
        Math.sin(phi),
        Math.cos(phi) * Math.sin(theta)
      ]
      targets.push({ direction, position: add(centre, scale(direction, radius)), elevation, azimuth })
    }
  }
  return { centre, radius, targets }
}

export function directionFromCentre (plan, position) {
  return normalize(sub(position, plan.centre))
}

export function coveredTargets (plan, shots, tolerance = ORBIT.targetTolerance) {
  return plan.targets.map(target =>
    shots.some(shot => angleBetween(shot.direction, target.direction) <= tolerance))
}

function orbitAdvice (state) {
  const { plan, viewer, shots, seconds } = state
  const toCentre = sub(plan.centre, viewer.position)
  const radius = length(toCentre)
  // walking a step around the subject turns the camera by the same angle, so the orbit
  // step is the overlap limit and nothing else
  const limits = limitsFor({ fov: state.fov, depth: radius })
  const direction = directionFromCentre(plan, viewer.position)
  const offAxis = angleBetween(viewer.forward, toCentre)

  const separation = shots.length
    ? Math.min(...shots.map(shot => angleBetween(shot.direction, direction)))
    : 180

  const covered = coveredTargets(plan, shots)
  const remaining = plan.targets.filter((_, index) => !covered[index])
  // The next target is the closest one you have not covered, measured around the subject
  // rather than through it: walking the orbit is the motion that fills a model.
  const target = remaining.length
    ? remaining.reduce((best, candidate) =>
      angleBetween(candidate.direction, direction) < angleBetween(best.direction, direction) ? candidate : best)
    : null
  const relative = target ? relativeTo(viewer, target.position) : null

  const advice = {
    capture: false,
    cue: 'walk',
    limits,
    progress: covered.filter(Boolean).length / plan.targets.length,
    target,
    arrow: relative ? { angle: Math.atan2(relative.lateral, relative.ahead) * 180 / Math.PI, behind: relative.ahead < 0 } : null,
    text: '',
    warning: null
  }

  advice.distance = radius

  if (offAxis > ORBIT.offAxisLimit) {
    advice.cue = 'aim'
    advice.text = 'Point back at the subject'
    return advice
  }
  if (radius < plan.radius * ORBIT.minRadiusRatio) {
    advice.cue = 'close'
    advice.text = 'Too close — step back'
    return advice
  }
  if (radius > plan.radius * ORBIT.maxRadiusRatio) {
    advice.cue = 'far'
    advice.text = 'Too far — close in'
    return advice
  }
  if (!target) {
    advice.cue = 'complete'
    advice.text = 'Every angle covered. Finish, or keep filling in.'
    advice.capture = separation >= limits.turn && seconds >= ORBIT.minSeconds
    return advice
  }

  if (separation >= limits.turn && seconds >= ORBIT.minSeconds) {
    advice.capture = true
    advice.cue = 'walk'
    advice.text = 'Keep walking around it'
    return advice
  }

  const sideways = Math.abs(relative.lateral) > Math.abs(relative.vertical)
  if (sideways) {
    advice.cue = relative.ahead < 0 ? 'behind' : (relative.lateral > 0 ? 'right' : 'left')
    advice.text = relative.ahead < 0
      ? 'Turn around and keep going'
      : `Keep going ${relative.lateral > 0 ? 'right' : 'left'} around it`
  } else {
    advice.cue = relative.vertical > 0 ? 'up' : 'down'
    advice.text = relative.vertical > 0 ? 'Raise the phone and orbit higher' : 'Lower the phone a little'
  }
  return advice
}

function walkAdvice (state) {
  const { viewer, shots, seconds } = state
  const limits = limitsFor({ fov: state.fov, depth: state.depth })
  if (!shots.length) {
    return { capture: seconds >= WALK.minSeconds, cue: 'start', progress: 0, readiness: 1, text: 'Start walking', warning: null, arrow: null, target: null, limits }
  }

  const last = shots[shots.length - 1]
  const travel = distance(viewer.position, last.position)
  const turn = angleBetween(viewer.forward, last.forward)
  // How close the next shot is, as a fraction: whichever of walking and turning gets
  // there first. The dial around the shutter draws this, so a walk has something to read
  // where an orbit has its coverage.
  const readiness = Math.min(1, Math.max(travel / limits.baseline, turn / limits.turn))
  const advice = { capture: false, cue: 'walk', progress: 0, readiness, target: null, arrow: null, text: '', warning: null, distance: travel, limits }

  // Turning a lot from a spot you have not left is exactly how you end up with a
  // panorama, which is the one capture no amount of matching can rescue.
  if (turn >= limits.turn && travel < WALK.spinTravel) {
    advice.warning = 'You are turning on the spot — walk sideways instead'
    advice.cue = 'spin'
  }

  // A turn earns a shot on its own, with no distance asked for. Refusing one because the
  // person had not also walked is what tore the first real capture apart: they swung the
  // phone through 60 degrees, nothing was taken, and the frames either side of the swing
  // had nothing in common for COLMAP to join them by.
  if ((travel >= limits.baseline || turn >= limits.turn) && seconds >= WALK.minSeconds) {
    advice.capture = true
    advice.text = 'Keep going'
    return advice
  }

  const left = Math.max(0, limits.baseline - travel)
  advice.text = `Walk ${Math.round(left * 100)} cm more`
  return advice
}

export function advise (state) {
  if (state.mode === 'orbit') {
    return state.plan ? orbitAdvice(state) : {
      capture: false, cue: 'subject', progress: 0, target: null, arrow: null, warning: null,
      limits: limitsFor({ fov: state.fov }),
      text: 'Point at the middle of your subject and set it'
    }
  }
  return walkAdvice(state)
}

// A shot as the guidance wants to remember it: where it was taken from, where it looked,
// and — for an orbit — where that puts it around the subject.
export function recordShot (plan, viewer) {
  return {
    position: viewer.position,
    forward: viewer.forward,
    direction: plan ? directionFromCentre(plan, viewer.position) : normalize(viewer.forward)
  }
}
