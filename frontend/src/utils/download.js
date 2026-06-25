/**
 * Trigger a browser download of a JSON payload.
 * @param {object} data  — the object to serialise
 * @param {string} filename — desired filename (without extension)
 */
export function downloadJSON(data, filename) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${filename}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Normalise an array of numbers to [0, 1].
 * @param {number[]} values
 * @returns {number[]}
 */
export function normalise(values) {
  const max = Math.max(...values, 1e-9);
  const min = Math.min(...values, 0);
  const range = max - min || 1;
  return values.map((v) => (v - min) / range);
}
