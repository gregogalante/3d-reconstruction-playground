export async function fetchDatasets() {
  const res = await fetch('/api/datasets');
  const data = await res.json();
  return data.datasets;
}

export async function fetchCameras(dataset) {
  const res = await fetch(`/api/datasets/${dataset}/cameras`);
  const data = await res.json();
  return data.cameras;
}

export async function fetchRelocations() {
  const res = await fetch('/api/relocations');
  const data = await res.json();
  return data.relocations;
}

export function imageUrl(dataset, filename) {
  return `/api/datasets/${dataset}/images/${filename}`;
}

export async function fetchClouds(dataset) {
  const res = await fetch(`/api/datasets/${dataset}/clouds`);
  const data = await res.json();
  return data.clouds;
}

export function cloudUrl(dataset, kind) {
  return `/api/datasets/${dataset}/clouds/${kind}.ply`;
}

export function splatUrl(dataset) {
  return `/api/datasets/${dataset}/splat.splat`;
}

export function relocationImageUrl(folder, name) {
  return `/api/relocations/${folder}/${name}/image`;
}

export function relocationOverlayUrl(folder, name) {
  return `/api/relocations/${folder}/${name}/overlay`;
}

export async function relocateImage(dataset, file) {
  const body = new FormData();
  body.append('file', file);
  const res = await fetch(`/api/datasets/${dataset}/relocate`, { method: 'POST', body });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'Relocation failed');
  return data;
}
