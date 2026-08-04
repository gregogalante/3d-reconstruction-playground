const LABELS = {
  sparse: 'Sparse (SfM)',
  dense: 'Dense (MVS)',
  splat: 'Gaussian splat',
};

function formatCount (points) {
  if (points >= 1e6) return `${(points / 1e6).toFixed(1)}M`;
  if (points >= 1e3) return `${Math.round(points / 1e3)}k`;
  return `${points}`;
}

export default function CloudList ({ clouds, selected, onSelect }) {
  return (
    <div className="space-y-0.5">
      {Object.keys(LABELS).map(kind => {
        const cloud = clouds[kind] || { available: false };
        const active = selected === kind;
        return (
          <button
            key={kind}
            onClick={() => cloud.available && onSelect(kind)}
            disabled={!cloud.available}
            className={`w-full flex items-center gap-2 px-2 py-1.5 rounded text-sm transition-colors ${
              active
                ? 'bg-blue-600 text-white'
                : cloud.available
                  ? 'hover:bg-gray-800 text-gray-300'
                  : 'text-gray-600 cursor-not-allowed'
            }`}
          >
            <span className="truncate">{LABELS[kind]}</span>
            <span className={`ml-auto text-xs ${active ? 'text-blue-100' : 'text-gray-600'}`}>
              {cloud.available ? formatCount(cloud.points) : 'not built'}
            </span>
          </button>
        );
      })}
    </div>
  );
}
