import { useMemo } from 'react';
import * as THREE from 'three';
import CameraFrustum from './CameraFrustum';

export default function RelocationMarker({ relocation }) {
  const matrix = useMemo(() => {
    if (!relocation.camera_to_world) return new THREE.Matrix4();
    const m = new THREE.Matrix4();
    const c = relocation.camera_to_world;
    m.set(
      c[0][0], c[0][1], c[0][2], c[0][3],
      c[1][0], c[1][1], c[1][2], c[1][3],
      c[2][0], c[2][1], c[2][2], c[2][3],
      c[3][0], c[3][1], c[3][2], c[3][3],
    );
    return m;
  }, [relocation]);

  return (
    <group matrixAutoUpdate={false} matrix={matrix}>
      <CameraFrustum fovx={relocation.fovx ?? 50} fovy={relocation.fovy ?? 38} scale={0.4} color="#ff4444" />
    </group>
  );
}
