import { Suspense, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Splat } from '@react-three/drei';
import PointCloud from './PointCloud';
import ImagePlane from './ImagePlane';
import RelocationMarker from './RelocationMarker';
import { cloudUrl, splatUrl } from '../api';

// the dense cloud has orders of magnitude more points than the sparse one, so its
// points must be smaller to keep surfaces readable
const POINT_SIZE = { sparse: 0.03, dense: 0.008 };

export default function Viewer3D({ dataset, cloud, cameras, activeImages, relocations }) {
  const activeCameras = useMemo(
    () => cameras.filter(c => activeImages.has(c.image_name)),
    [cameras, activeImages]
  );

  return (
    <Canvas
      camera={{ position: [0, 5, 10], fov: 60, near: 0.01, far: 1000 }}
      className="!absolute inset-0"
    >
      <ambientLight intensity={1} />
      <OrbitControls makeDefault />
      <axesHelper args={[2]} />

      <group rotation={[-Math.PI / 2, 0, 0]}>
        {cloud === 'splat' ? (
          <Suspense fallback={null}>
            <Splat src={splatUrl(dataset)} key={`${dataset}-splat`} />
          </Suspense>
        ) : cloud ? (
          <Suspense fallback={null}>
            <PointCloud
              url={cloudUrl(dataset, cloud)}
              size={POINT_SIZE[cloud]}
              key={`${dataset}-${cloud}`}
            />
          </Suspense>
        ) : null}

        {activeCameras.map(cam => (
          <Suspense key={cam.image_name} fallback={null}>
            <ImagePlane camera={cam} dataset={dataset} />
          </Suspense>
        ))}

        {relocations.map(rel => (
          <RelocationMarker key={rel.name} relocation={rel} />
        ))}
      </group>
    </Canvas>
  );
}
