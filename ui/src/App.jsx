import { useState, useEffect, useCallback } from 'react';
import { fetchDatasets, fetchCameras, fetchClouds, fetchRelocations, relocationOverlayUrl } from './api';
import Sidebar from './components/Sidebar';
import Viewer3D from './components/Viewer3D';
import ImageModal from './components/ImageModal';

function App() {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [cameras, setCameras] = useState([]);
  const [clouds, setClouds] = useState({});
  const [selectedCloud, setSelectedCloud] = useState(null);
  const [activeImages, setActiveImages] = useState(new Set());
  const [relocations, setRelocations] = useState([]);
  const [showRelocations, setShowRelocations] = useState(true);
  const [overlay, setOverlay] = useState(null);

  useEffect(() => {
    fetchDatasets().then(setDatasets);
    fetchRelocations().then(setRelocations);
  }, []);

  useEffect(() => {
    if (!selectedDataset) {
      setCameras([]);
      setClouds({});
      setSelectedCloud(null);
      setActiveImages(new Set());
      return;
    }
    fetchCameras(selectedDataset).then(setCameras);
    fetchClouds(selectedDataset).then(available => {
      setClouds(available);
      // show whatever the pipeline has already produced, densest first
      const order = ['splat', 'dense', 'sparse'];
      setSelectedCloud(order.find(kind => available[kind]?.available) || null);
    });
    setActiveImages(new Set());
  }, [selectedDataset]);

  const toggleImage = useCallback((imageName) => {
    setActiveImages(prev => {
      const next = new Set(prev);
      if (next.has(imageName)) next.delete(imageName);
      else next.add(imageName);
      return next;
    });
  }, []);

  const selectAll = useCallback(() => {
    setActiveImages(new Set(cameras.map(c => c.image_name)));
  }, [cameras]);

  const clearAll = useCallback(() => {
    setActiveImages(new Set());
  }, []);

  // a fresh relocation shows its overlay straight away, it is what tells you it worked
  const handleRelocated = useCallback(async relocation => {
    setRelocations(await fetchRelocations());
    setOverlay(relocation);
  }, []);

  const filteredRelocations = relocations.filter(
    r => r.dataset_name === selectedDataset && r.success
  );

  return (
    <div className="flex h-screen bg-gray-950 text-gray-100">
      <Sidebar
        datasets={datasets}
        selectedDataset={selectedDataset}
        onSelectDataset={setSelectedDataset}
        clouds={clouds}
        selectedCloud={selectedCloud}
        onSelectCloud={setSelectedCloud}
        cameras={cameras}
        activeImages={activeImages}
        onToggleImage={toggleImage}
        onSelectAll={selectAll}
        onClearAll={clearAll}
        relocations={filteredRelocations}
        showRelocations={showRelocations}
        onToggleRelocations={() => setShowRelocations(v => !v)}
        onRelocated={handleRelocated}
        onOpenOverlay={setOverlay}
      />
      <div className="flex-1 relative">
        {selectedDataset ? (
          <Viewer3D
            dataset={selectedDataset}
            cloud={selectedCloud}
            cameras={cameras}
            activeImages={activeImages}
            relocations={showRelocations ? filteredRelocations : []}
          />
        ) : (
          <div className="flex items-center justify-center h-full text-gray-500">
            Select a dataset to begin
          </div>
        )}
      </div>

      {overlay && (
        <ImageModal
          src={relocationOverlayUrl(overlay.folder, overlay.name)}
          title={overlay.name}
          onClose={() => setOverlay(null)}
        />
      )}
    </div>
  );
}

export default App;
