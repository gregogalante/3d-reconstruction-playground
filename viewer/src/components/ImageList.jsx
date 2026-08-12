import { imageUrl } from '../api';

export default function ImageList({
  cameras, dataset, activeImages, onToggle, onSelectAll, onClearAll,
}) {
  return (
    <div>
      <div className="flex gap-1 mb-2">
        <button
          onClick={onSelectAll}
          className="text-xs px-2 py-1 rounded bg-gray-800 hover:bg-gray-700 text-gray-300"
        >
          All
        </button>
        <button
          onClick={onClearAll}
          className="text-xs px-2 py-1 rounded bg-gray-800 hover:bg-gray-700 text-gray-300"
        >
          None
        </button>
      </div>
      <div className="space-y-0.5 max-h-64 overflow-y-auto">
        {cameras.map(cam => {
          const active = activeImages.has(cam.image_name);
          return (
            <button
              key={cam.image_name}
              onClick={() => onToggle(cam.image_name)}
              className={`w-full flex items-center gap-2 px-2 py-1 rounded text-xs transition-colors ${
                active ? 'bg-green-900/50 text-green-300' : 'hover:bg-gray-800 text-gray-400'
              }`}
            >
              <img
                src={imageUrl(dataset, cam.image_name)}
                alt={cam.image_name}
                loading="lazy"
                className="w-8 h-6 object-cover rounded flex-shrink-0"
              />
              <span className="truncate">{cam.image_name}</span>
              <span className={`ml-auto w-2 h-2 rounded-full flex-shrink-0 ${
                active ? 'bg-green-400' : 'bg-gray-700'
              }`} />
            </button>
          );
        })}
      </div>
    </div>
  );
}
