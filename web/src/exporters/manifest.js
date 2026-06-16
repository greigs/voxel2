// Tile manifest (CSV) and paint color-code legend (JSON + rows for the UI).

import { paletteRgb } from "../voxParser.js";
import { dirName } from "../tiles.js";

export function legendRows(layers) {
  const legend = layers.colorLegend || new Map();
  const rows = [];
  for (const code of [...legend.keys()].sort((a, b) => a - b)) {
    const paletteIndex = legend.get(code);
    const rgb = paletteRgb(layers.paletteByIndex, paletteIndex);
    const hex = "#" + rgb.map((c) => c.toString(16).padStart(2, "0")).join("").toUpperCase();
    rows.push({ color_code: code, palette_index: paletteIndex, rgb, hex });
  }
  return rows;
}

export function legendJson(layers) {
  return JSON.stringify({ colors: legendRows(layers) }, null, 2);
}

export function manifestCsv(tiles) {
  const lines = ["number,voxel_x,voxel_y,voxel_z,face,color_code"];
  for (const t of tiles) {
    lines.push([t.number, t.voxel[0], t.voxel[1], t.voxel[2], dirName(t.axis, t.sign), t.colorCode].join(","));
  }
  return lines.join("\r\n") + "\r\n";
}
