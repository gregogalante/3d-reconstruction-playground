import { useState } from 'react';
import ImageModal from './ImageModal';
import { relocationOverlayUrl } from '../api';

export default function RelocationList({ relocations, show, onToggleShow }) {
  const [opened, setOpened] = useState(null);

  return (
    <div>
      <label className="flex items-center gap-2 mb-2 text-xs text-gray-400 cursor-pointer">
        <input
          type="checkbox"
          checked={show}
          onChange={onToggleShow}
          className="rounded"
        />
        Show in viewer
      </label>
      <div className="space-y-0.5">
        {relocations.map(rel => (
          <div key={rel.name} className="px-2 py-1 rounded text-xs text-gray-400">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-red-500 flex-shrink-0" />
              <span className="truncate">{rel.name}</span>
              <span className="ml-auto text-gray-600 flex-shrink-0">
                {rel.num_inliers}/{rel.num_correspondences}
                {rel.reprojection_error != null && ` · ${rel.reprojection_error}px`}
              </span>
            </div>
            {/* the overlay is the only way to tell a good pose from a plausible one */}
            {rel.has_overlay && (
              <button
                type="button"
                onClick={() => setOpened(rel)}
                className="mt-1 ml-4 text-gray-600 hover:text-gray-300 underline"
              >
                verification overlay
              </button>
            )}
          </div>
        ))}
      </div>

      {opened && (
        <ImageModal
          src={relocationOverlayUrl(opened.folder, opened.name)}
          title={opened.name}
          onClose={() => setOpened(null)}
        />
      )}
    </div>
  );
}
