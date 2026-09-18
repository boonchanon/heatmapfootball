/* Shared, model-free occupancy math used by the page and Node regression checks. */
function selectSamples(data, options) {
  return data.samples.filter(p => p[2] >= options.start && p[2] < options.end &&
    (options.track === 'all' || String(p[4]) === options.track) &&
    (!options.excludeTruncated || !p[5]));
}
function occupancyGrid(samples, fps, sigma = 2) {
  const width = 105, height = 68;
  const raw = new Float64Array(width * height);
  for (const p of samples) {
    if (!Number.isFinite(p[0]) || !Number.isFinite(p[1]) || Math.abs(p[0]) > 52.5 || Math.abs(p[1]) > 34) continue;
    const x = Math.min(width - 1, Math.floor(p[0] + 52.5));
    const y = Math.min(height - 1, Math.floor(p[1] + 34));
    raw[y * width + x] += 1 / fps;
  }
  if (sigma <= 0) return raw;
  const result = new Float64Array(raw.length), kernel = [];
  const radius = Math.ceil(sigma * 3);
  for (let dy = -radius; dy <= radius; dy++) for (let dx = -radius; dx <= radius; dx++) {
    kernel.push([dx, dy, Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))]);
  }
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const value = raw[y * width + x];
    if (!value) continue;
    let weight = 0;
    for (const [dx, dy, w] of kernel) if (x + dx >= 0 && x + dx < width && y + dy >= 0 && y + dy < height) weight += w;
    for (const [dx, dy, w] of kernel) if (x + dx >= 0 && x + dx < width && y + dy >= 0 && y + dy < height) {
      result[(y + dy) * width + x + dx] += value * w / weight;
    }
  }
  return result;
}
if (typeof module !== 'undefined') module.exports = {selectSamples, occupancyGrid};
