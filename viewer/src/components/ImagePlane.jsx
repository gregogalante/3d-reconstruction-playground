import { useMemo } from 'react';
import { useLoader } from '@react-three/fiber';
import * as THREE from 'three';
import CameraFrustum from './CameraFrustum';
import { imageUrl } from '../api';

export default function ImagePlane({ camera, dataset }) {
  const texture = useLoader(THREE.TextureLoader, imageUrl(dataset, camera.image_name));

  const matrix = useMemo(() => {
    const m = new THREE.Matrix4();
    const c = camera.camera_to_world;
    m.set(
      c[0][0], c[0][1], c[0][2], c[0][3],
      c[1][0], c[1][1], c[1][2], c[1][3],
      c[2][0], c[2][1], c[2][2], c[2][3],
      c[3][0], c[3][1], c[3][2], c[3][3],
    );
    return m;
  }, [camera]);

  const scale = 0.3;
  const hw = Math.tan((camera.fovx * Math.PI / 180) / 2) * scale;
  const hh = Math.tan((camera.fovy * Math.PI / 180) / 2) * scale;

  return (
    <group matrixAutoUpdate={false} matrix={matrix}>
      <CameraFrustum fovx={camera.fovx} fovy={camera.fovy} scale={scale} color="#00ff00" />
      <mesh position={[0, 0, -scale]}>
        <planeGeometry args={[hw * 2, hh * 2]} />
        <meshBasicMaterial
          map={texture}
          side={THREE.DoubleSide}
          transparent
          opacity={0.85}
        />
      </mesh>
    </group>
  );
}
