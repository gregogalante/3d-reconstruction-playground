import { useRef, useState } from 'react';
import { relocateImage } from '../api';

export default function RelocateForm({ dataset, onRelocated }) {
  const input = useRef(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState(null);

  async function upload(event) {
    const file = event.target.files?.[0];
    // the input is reset right away, so the same photo can be sent again after a failure
    event.target.value = '';
    if (!file) return;
    setError(null);
    setPending(true);
    try {
      onRelocated(await relocateImage(dataset, file));
    } catch (err) {
      setError(err.message);
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="space-y-1">
      <input ref={input} type="file" accept="image/jpeg,image/png" onChange={upload} className="hidden" />
      <button
        type="button"
        disabled={pending}
        onClick={() => input.current.click()}
        className="w-full px-2 py-1.5 rounded text-xs bg-gray-800 hover:bg-gray-700 disabled:opacity-50 disabled:hover:bg-gray-800"
      >
        {pending ? `Locating in ${dataset}...` : 'Locate a photo'}
      </button>
      {error && <div className="px-2 text-xs text-red-400">{error}</div>}
    </div>
  );
}
