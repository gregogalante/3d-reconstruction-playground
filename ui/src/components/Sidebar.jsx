import DatasetList from './DatasetList';
import CloudList from './CloudList';
import ImageList from './ImageList';
import RelocationList from './RelocationList';

export default function Sidebar({
  datasets, selectedDataset, onSelectDataset,
  clouds, selectedCloud, onSelectCloud,
  cameras, activeImages, onToggleImage, onSelectAll, onClearAll,
  relocations, showRelocations, onToggleRelocations,
}) {
  return (
    <div className="w-80 bg-gray-900 border-r border-gray-800 flex flex-col overflow-hidden">
      <div className="p-3 border-b border-gray-800">
        <h1 className="text-sm font-bold tracking-wide uppercase text-gray-400">COLMAP Viewer</h1>
      </div>

      <div className="flex-1 overflow-y-auto">
        <Section title="Datasets">
          <DatasetList
            datasets={datasets}
            selected={selectedDataset}
            onSelect={onSelectDataset}
          />
        </Section>

        {selectedDataset && (
          <>
            <Section title="Point clouds">
              <CloudList
                clouds={clouds}
                selected={selectedCloud}
                onSelect={onSelectCloud}
              />
            </Section>

            <Section title={`Images (${activeImages.size}/${cameras.length})`}>
              <ImageList
                cameras={cameras}
                dataset={selectedDataset}
                activeImages={activeImages}
                onToggle={onToggleImage}
                onSelectAll={onSelectAll}
                onClearAll={onClearAll}
              />
            </Section>

            {relocations.length > 0 && (
              <Section title="Relocations">
                <RelocationList
                  relocations={relocations}
                  show={showRelocations}
                  onToggleShow={onToggleRelocations}
                />
              </Section>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function Section({ title, children }) {
  return (
    <div className="border-b border-gray-800">
      <div className="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-gray-500">
        {title}
      </div>
      <div className="px-2 pb-2">{children}</div>
    </div>
  );
}
