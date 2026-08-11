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
  angularStep: 12, // degrees around the subject between two shots
  targetTolerance: 18, // how close a shot has to pass for a guidance target to count
  offAxisLimit: 38, // degrees the subject may sit off the middle of the frame
  minRadiusRatio: 0.55, // fraction of the set radius before you are too close
  maxRadiusRatio: 2.2,
  minSeconds: 0.4
}

export const WALK = {
  baseline: 0.3, // metres of travel that earn a shot
  rotationBaseline: 25, // degrees of turn that also earn one...
  rotationMinTravel: 0.1, // ...as long as you moved at least this far while turning
  spinTravel: 0.12, // turning this much without moving is the panorama failure
  minSeconds: 0.6
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
    progress: covered.filter(Boolean).length / plan.targets.length,
    target,
    arrow: relative ? { angle: Math.atan2(relative.lateral, relative.ahead) * 180 / Math.PI, behind: relative.ahead < 0 } : null,
    text: '',
    warning: null
  }

  if (offAxis > ORBIT.offAxisLimit) {
    advice.text = 'Point back at the subject'
    return advice
  }
  if (radius < plan.radius * ORBIT.minRadiusRatio) {
    advice.text = 'Too close — step back'
    return advice
  }
  if (radius > plan.radius * ORBIT.maxRadiusRatio) {
    advice.text = 'Too far — close in'
    return advice
  }
  if (!target) {
    advice.text = 'Every angle covered. Finish, or keep filling in.'
    advice.capture = separation >= ORBIT.angularStep && seconds >= ORBIT.minSeconds
    return advice
  }

  if (separation >= ORBIT.angularStep && seconds >= ORBIT.minSeconds) {
    advice.capture = true
    advice.text = 'Keep walking around it'
    return advice
  }

  const sideways = Math.abs(relative.lateral) > Math.abs(relative.vertical)
  if (sideways) {
    advice.text = relative.ahead < 0
      ? 'Turn around and keep going'
      : `Keep going ${relative.lateral > 0 ? 'right' : 'left'} around it`
  } else {
    advice.text = relative.vertical > 0 ? 'Raise the phone and orbit higher' : 'Lower the phone a little'
  }
  return advice
}

function walkAdvice (state) {
  const { viewer, shots, seconds } = state
  if (!shots.length) {
    return { capture: seconds >= WALK.minSeconds, progress: 0, text: 'Start walking', warning: null, arrow: null, target: null }
  }

  const last = shots[shots.length - 1]
  const travel = distance(viewer.position, last.position)
  const turn = angleBetween(viewer.forward, last.forward)
  const advice = { capture: false, progress: 0, target: null, arrow: null, text: '', warning: null }

  // Turning a lot from a spot you have not left is exactly how you end up with a
  // panorama, which is the one capture no amount of matching can rescue.
  if (turn >= WALK.rotationBaseline && travel < WALK.spinTravel) {
    advice.warning = 'You are turning on the spot — walk sideways instead'
  }

  const earned = travel >= WALK.baseline || (turn >= WALK.rotationBaseline && travel >= WALK.rotationMinTravel)
  if (earned && seconds >= WALK.minSeconds) {
    advice.capture = true
    advice.text = 'Keep going'
    return advice
  }

  const left = Math.max(0, WALK.baseline - travel)
  advice.text = `Walk ${Math.round(left * 100)} cm more`
  return advice
}

export function advise (state) {
  if (state.mode === 'orbit') {
    return state.plan ? orbitAdvice(state) : {
      capture: false, progress: 0, target: null, arrow: null, warning: null,
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
