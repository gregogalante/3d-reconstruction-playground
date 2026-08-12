import { useEffect } from 'react';

export default function ImageModal({ src, title, onClose }) {
  useEffect(() => {
    const onKey = e => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 bg-black/85 flex flex-col items-center justify-center p-6 cursor-zoom-out"
      onClick={onClose}
    >
      <div className="mb-2 text-xs text-gray-400 flex gap-3">
        <span className="truncate">{title}</span>
        <span className="text-gray-600">click anywhere or press esc to close</span>
      </div>
      {/* the overlay is a wide two panel image, fitting it whole is what makes it readable */}
      <img src={src} alt={title} className="max-w-full max-h-[85vh] object-contain" />
    </div>
  );
}
