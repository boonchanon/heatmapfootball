# Offline heatmap

Open `outputs/match_20s_heatmap/index.html` in Edge or Chrome. The page embeds its
data and needs no server or network connection. Choose all tracks or one ID,
set a time interval, adjust display blur, and save the canvas as PNG.

The initial artifact uses the improved 20-second tracking export: 5,911 finite
in-field samples, with 769 outside/missing points excluded. Border-truncated or
unknown-quality boxes are excluded by default. Each sample contributes 1/FPS
seconds to a 1x1 m cell in the 105x68 m canonical field. No missing observations
are interpolated. Time filtering uses [start, end).

The optional Gaussian display blur conserves total accumulated time, including
at field edges. Its peak is labeled in accumulated seconds per cell. Color scale
is relative to the current selection: colors alone cannot compare different
tracks or intervals. Aggregate track-seconds can exceed video duration.

This is camera-visible occupancy, not a complete player heatmap or a distance
estimate. IDs can fragment, referees can be present, and projection and identity
accuracy remain unverified. Team/jersey inference is not part of the source MVP.

Build another page into a fresh output directory:

```powershell
.venv/Scripts/python.exe tools/build_heatmap.py --tracking outputs/match_20s_improved_final/tracking.json --output outputs/match_20s_heatmap
.venv/Scripts/python.exe -m pytest tests -q
node tests/test_heatmap_math.cjs
```

Validation: 43 Python tests pass; JavaScript checks cover occupancy conservation,
boundary handling, time/ID/truncation filters. Headless Edge checked the actual
page's track/time controls, invalid and empty intervals, and PNG encoding.
`preview.png` and `browser_validation.json` in the output directory record the
browser check.
