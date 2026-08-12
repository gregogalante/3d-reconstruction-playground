// node --test capture/lib
//
// The guidance is the only part of the capture with rules worth arguing about, and the
// only part that runs without a phone. These pin the rules down: when a shot is earned,
// when the person is told to move, and the panorama warning that saves a dataset.

import test from 'node:test'
import assert from 'node:assert/strict'

import { planOrbit, advise, recordShot, coveredTargets, limitsFor } from './guidance.js'
import { angleBetween, distance } from './vec.js'

const viewerAt = (position, lookingAt = [0, 0, 0]) => {
  const forward = [lookingAt[0] - position[0], lookingAt[1] - position[1], lookingAt[2] - position[2]]
  const size = Math.hypot(...forward) || 1
  return { position, forward: forward.map(value => value / size) }
}

// A walker standing `radius` from the subject, `azimuth` degrees around it.
const onOrbit = (plan, azimuth, elevation = 0) => {
  const phi = elevation * Math.PI / 180
  const theta = azimuth * Math.PI / 180
  const offset = [
    Math.cos(phi) * Math.cos(theta) * plan.radius,
    Math.sin(phi) * plan.radius,
    Math.cos(phi) * Math.sin(theta) * plan.radius
  ]
  return viewerAt(plan.centre.map((value, axis) => value + offset[axis]), plan.centre)
}

test('the plan puts the subject where you are looking, at the distance you chose', () => {
  const plan = planOrbit(viewerAt([0, 1.5, 0], [0, 1.5, -2]), 1.6)
  assert.equal(plan.targets.length, 24)
  assert.ok(distance(plan.centre, [0, 1.5, -1.6]) < 1e-6)
  for (const target of plan.targets) {
    assert.ok(Math.abs(distance(target.position, plan.centre) - 1.6) < 1e-6)
  }
})

test('a shot is earned by moving around the subject, not by standing still', () => {
  const plan = planOrbit(viewerAt([0, 1.5, 0], [0, 1.5, -2]), 1.6)
  const shots = []

  const first = advise({ mode: 'orbit', plan, viewer: onOrbit(plan, 0), shots, seconds: 5 })
  assert.equal(first.capture, true, 'the first shot has nothing to be too close to')
  shots.push(recordShot(plan, onOrbit(plan, 0)))

  const standing = advise({ mode: 'orbit', plan, viewer: onOrbit(plan, 0), shots, seconds: 5 })
  assert.equal(standing.capture, false, 'the same spot is the same shot')

  const step = limitsFor({ depth: 1.6 }).turn
  const nudged = advise({ mode: 'orbit', plan, viewer: onOrbit(plan, step - 3), shots, seconds: 5 })
  assert.equal(nudged.capture, false, 'half a step is not a new viewpoint')

  const moved = advise({ mode: 'orbit', plan, viewer: onOrbit(plan, step + 2), shots, seconds: 5 })
  assert.equal(moved.capture, true)
})

test('a shot is never earned before the shutter has settled', () => {
  const plan = planOrbit(viewerAt([0, 1.5, 0], [0, 1.5, -2]), 1.6)
  const shots = [recordShot(plan, onOrbit(plan, 0))]
  const tooSoon = advise({ mode: 'orbit', plan, viewer: onOrbit(plan, 40), shots, seconds: 0.1 })
  assert.equal(tooSoon.capture, false)
})

test('standing in the wrong place is named, and stops the shutter', () => {
  const plan = planOrbit(viewerAt([0, 1.5, 0], [0, 1.5, -2]), 1.6)
  const shots = []

  const close = { ...onOrbit(plan, 0) }
  close.position = plan.centre.map((value, axis) => value + (close.position[axis] - plan.centre[axis]) * 0.3)
  const tooClose = advise({ mode: 'orbit', plan, viewer: { ...close, forward: onOrbit(plan, 0).forward }, shots, seconds: 5 })
  assert.match(tooClose.text, /step back/i)
  assert.equal(tooClose.capture, false)

  const far = { ...onOrbit(plan, 0) }
  far.position = plan.centre.map((value, axis) => value + (far.position[axis] - plan.centre[axis]) * 3)
  const tooFar = advise({ mode: 'orbit', plan, viewer: { ...far, forward: onOrbit(plan, 0).forward }, shots, seconds: 5 })
  assert.match(tooFar.text, /close in/i)
  assert.equal(tooFar.capture, false)

  const away = onOrbit(plan, 0)
  const looking = advise({ mode: 'orbit', plan, viewer: { position: away.position, forward: [1, 0, 0] }, shots, seconds: 5 })
  assert.match(looking.text, /point back/i)
  assert.equal(looking.capture, false)
})

test('coverage counts a target once a shot passes close enough around the subject', () => {
  const plan = planOrbit(viewerAt([0, 1.5, 0], [0, 1.5, -2]), 1.6)
  assert.equal(coveredTargets(plan, []).filter(Boolean).length, 0)

  const shots = [recordShot(plan, onOrbit(plan, plan.targets[0].azimuth, plan.targets[0].elevation))]
  const covered = coveredTargets(plan, shots)
  assert.equal(covered[0], true)
  assert.ok(covered.filter(Boolean).length < plan.targets.length, 'one shot is not a capture')
})

test('walking the whole orbit covers it, which is what the progress bar claims', () => {
  const plan = planOrbit(viewerAt([0, 1.5, 0], [0, 1.5, -2]), 1.6)
  const shots = []
  for (const elevation of [8, 28]) {
    for (let azimuth = 0; azimuth < 360; azimuth += 10) {
      shots.push(recordShot(plan, onOrbit(plan, azimuth, elevation)))
    }
  }
  assert.equal(coveredTargets(plan, shots).every(Boolean), true)
  assert.equal(advise({ mode: 'orbit', plan, viewer: onOrbit(plan, 0, 8), shots, seconds: 5 }).progress, 1)
})

test('the arrow and the words agree on which way to walk', () => {
  const plan = planOrbit(viewerAt([0, 1.5, 0], [0, 1.5, -2]), 1.6)
  // one shot, so the walker is told to move on to the next uncovered target
  const shots = [recordShot(plan, onOrbit(plan, 0, 8))]
  const advice = advise({ mode: 'orbit', plan, viewer: onOrbit(plan, 2, 8), shots, seconds: 5 })
  assert.ok(advice.arrow, 'there is a target to point at')
  if (/right/.test(advice.text)) assert.ok(advice.arrow.angle > 0, 'right means clockwise on screen')
  if (/left/.test(advice.text)) assert.ok(advice.arrow.angle < 0, 'left means anticlockwise on screen')
})

test('walking earns a shot by distance, and turning earns one on its own', () => {
  const shots = [{ position: [0, 1.5, 0], forward: [0, 0, -1], direction: [0, 0, -1] }]
  const limits = limitsFor({ fov: 34, depth: 1.5 })

  const still = advise({ mode: 'walk', fov: 34, depth: 1.5, viewer: viewerAt([0.02, 1.5, 0], [0.02, 1.5, -1]), shots, seconds: 5 })
  assert.equal(still.capture, false)
  assert.match(still.text, /cm more/)

  const walked = advise({ mode: 'walk', fov: 34, depth: 1.5, viewer: viewerAt([limits.baseline + 0.02, 1.5, 0], [limits.baseline + 0.02, 1.5, -1]), shots, seconds: 5 })
  assert.equal(walked.capture, true)

  // A turn past the overlap limit takes the shot even from the same spot: the frames
  // either side of a swing have to have something in common, or the model splits there.
  const spun = advise({ mode: 'walk', fov: 34, depth: 1.5, viewer: { position: [0.02, 1.5, 0], forward: [1, 0, 0] }, shots, seconds: 5 })
  assert.equal(spun.capture, true, 'the chain matters more than the parallax')
  assert.match(spun.warning, /turning on the spot/i, 'and it still says what is wrong')

  // the same turn, having walked while doing it, is a shot without the complaint
  const arced = advise({ mode: 'walk', fov: 34, depth: 1.5, viewer: { position: [0.2, 1.5, 0], forward: [1, 0, 0] }, shots, seconds: 5 })
  assert.equal(arced.capture, true)
  assert.equal(arced.warning, null)
})

test('a walk always knows how close the next frame is', () => {
  const shots = [{ position: [0, 1.5, 0], forward: [0, 0, -1], direction: [0, 0, -1] }]
  const still = advise({ mode: 'walk', fov: 34, depth: 1.5, viewer: viewerAt([0, 1.5, 0], [0, 1.5, -1]), shots, seconds: 5 })
  assert.equal(still.readiness, 0, 'nothing moved, nothing earned')

  const limits = limitsFor({ fov: 34, depth: 1.5 })
  const halfway = advise({ mode: 'walk', fov: 34, depth: 1.5, viewer: viewerAt([limits.baseline / 2, 1.5, 0], [limits.baseline / 2, 1.5, -1]), shots, seconds: 5 })
  assert.ok(Math.abs(halfway.readiness - 0.5) < 0.02)

  const there = advise({ mode: 'walk', fov: 34, depth: 1.5, viewer: viewerAt([limits.baseline * 2, 1.5, 0], [limits.baseline * 2, 1.5, -1]), shots, seconds: 5 })
  assert.equal(there.readiness, 1, 'the dial fills and stops, it does not wrap')
  assert.equal(there.capture, true)
})

test('the thresholds follow the lens and the distance, not a constant', () => {
  // a phone in portrait sees 34 degrees across, a wide lens twice that: the same rule
  // has to mean a much smaller step on the narrow one
  assert.ok(limitsFor({ fov: 34 }).turn < limitsFor({ fov: 70 }).turn)
  assert.ok(Math.abs(limitsFor({ fov: 34 }).turn - 8.5) < 0.01)
  // and the step grows with how far away the subject is
  assert.ok(limitsFor({ depth: 0.8 }).baseline < limitsFor({ depth: 4 }).baseline)
  // clamped at both ends, so a bad depth reading cannot ask for a 3 metre stride
  assert.equal(limitsFor({ depth: 40 }).baseline, 0.5)
  assert.equal(limitsFor({ depth: 0.1 }).baseline, 0.06)
})

test('the very first shot of a walk is free, there is nothing to compare it with', () => {
  const advice = advise({ mode: 'walk', viewer: viewerAt([0, 1.5, 0], [0, 1.5, -1]), shots: [], seconds: 5 })
  assert.equal(advice.capture, true)
})

test('an orbit with no plan asks for the subject instead of guessing one', () => {
  const advice = advise({ mode: 'orbit', plan: null, viewer: viewerAt([0, 1.5, 0]), shots: [], seconds: 5 })
  assert.equal(advice.capture, false)
  assert.match(advice.text, /subject/i)
})

test('angles are measured the way the thresholds are quoted', () => {
  assert.ok(Math.abs(angleBetween([1, 0, 0], [0, 1, 0]) - 90) < 1e-9)
  assert.ok(Math.abs(angleBetween([1, 0, 0], [1, 0, 0])) < 1e-6)
  assert.ok(Math.abs(angleBetween([1, 0, 0], [-1, 0, 0]) - 180) < 1e-9)
})
