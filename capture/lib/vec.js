// The three vector operations the guidance needs, kept apart from it so the guidance
// reads as intent. Everything here is plain arrays of three numbers, in the WebXR frame:
// right handed, +Y up, the camera looking down -Z.

export function sub (a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

export function add (a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

export function scale (a, k) {
  return [a[0] * k, a[1] * k, a[2] * k]
}

export function dot (a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

export function cross (a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]
  ]
}

export function length (a) {
  return Math.sqrt(dot(a, a))
}

export function distance (a, b) {
  return length(sub(a, b))
}

export function normalize (a) {
  const size = length(a)
  return size < 1e-9 ? [0, 0, 0] : scale(a, 1 / size)
}

// Degrees, because every threshold in the guidance is quoted in degrees and converting
// at each comparison is how sign and unit mistakes get in.
export function angleBetween (a, b) {
  const cosine = dot(normalize(a), normalize(b))
  return Math.acos(Math.min(1, Math.max(-1, cosine))) * 180 / Math.PI
}

export function rotateByQuaternion (q, v) {
  // v + 2 * qv x (qv x v + w * v), the usual expansion that avoids building a matrix
  const qv = [q.x, q.y, q.z]
  const t = cross(qv, add(cross(qv, v), scale(v, q.w)))
  return add(v, scale(t, 2))
}

export function forwardOf (orientation) {
  // A camera in WebXR looks down its own -Z, so its viewing direction in the world is
  // that axis carried over by the pose orientation.
  return rotateByQuaternion(orientation, [0, 0, -1])
}
