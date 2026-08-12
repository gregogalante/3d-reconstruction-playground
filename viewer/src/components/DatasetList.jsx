export default function DatasetList({ datasets, selected, onSelect }) {
  return (
    <div className="space-y-0.5">
      {datasets.map(name => (
        <button
          key={name}
          onClick={() => onSelect(name)}
          className={`w-full text-left px-2 py-1.5 rounded text-sm transition-colors ${
            selected === name
              ? 'bg-blue-600 text-white'
              : 'hover:bg-gray-800 text-gray-300'
          }`}
        >
          {name}
        </button>
      ))}
    </div>
  );
}
