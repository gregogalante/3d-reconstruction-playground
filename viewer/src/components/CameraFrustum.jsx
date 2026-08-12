import { useMemo } from 'react';
import * as THREE from 'three';

export default function CameraFrustum({ fovx = 45, fovy = 35, scale = 0.3, color = '#00ff00' }) {
  const geometry = useMemo(() => {
    const hw = Math.tan((fovx * Math.PI / 180) / 2) * scale;
    const hh = Math.tan((fovy * Math.PI / 180) / 2) * scale;

    // Camera looks down -Z in local space
    const o = [0, 0, 0];
    const tl = [-hw, hh, -scale];
    const tr = [hw, hh, -scale];
    const br = [hw, -hh, -scale];
    const bl = [-hw, -hh, -scale];

    const vertices = new Float32Array([
      // Lines from origin to corners
      ...o, ...tl, ...o, ...tr, ...o, ...br, ...o, ...bl,
      // Near plane rectangle
      ...tl, ...tr, ...tr, ...br, ...br, ...bl, ...bl, ...tl,
    ]);

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    return geom;
  }, [fovx, fovy, scale]);

  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color={color} />
    </lineSegments>
  );
}
