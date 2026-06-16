// App entry point: wire the upload + params form, run the entirely client-side pipeline,
// drive the three.js viewer, and build download blobs.

import { parseVox } from "./voxParser.js";
import { computeLayers } from "./voxelOps.js";
import { generateFaceTiles } from "./tiles.js";
import * as labels from "./labels.js";
import { baseStl } from "./exporters/stl.js";
import { exportTiles3mf } from "./exporters/threemf.js";
import { manifestCsv, legendJson, legendRows } from "./exporters/manifest.js";
import { Viewer } from "./viewer.js";

const FONT_URL = `${import.meta.env.BASE_URL}fonts/DejaVuSans-Bold.ttf`;

const $ = (id) => document.getElementById(id);

const state = {
  arrayBuffer: null,
  fileName: null,
  viewer: null,
  tiles: null,
  layers: null,
};

function setStatus(msg, isError = false) {
  const el = $("status");
  el.textContent = msg || "";
  el.classList.toggle("error", isError);
}

function readParams() {
  const num = (id) => parseFloat($(id).value);
  const intOrNull = (id) => ($(id).value === "" ? null : parseInt($(id).value, 10));
  return {
    scaleFactor: num("scale_factor"),
    erosionVoxels: parseInt($("erosion_voxels").value, 10),
    pegSizeVoxels: intOrNull("peg_size_voxels"),
    pegDepthVoxels: parseInt($("peg_depth_voxels").value, 10),
    voxelSizeMm: num("voxel_size_mm"),
    embossDepthMm: num("emboss_depth_mm"),
    pegClearanceMm: num("peg_clearance_mm"),
    pegDepthClearanceMm: num("peg_depth_clearance_mm"),
    tileClearanceMm: num("tile_clearance_mm"),
    withLabels: $("with_labels").checked,
  };
}

// ---- file selection ---------------------------------------------------------------

function onFile(file) {
  if (!file) return;
  state.fileName = file.name;
  $("file-label").textContent = file.name;
  const reader = new FileReader();
  reader.onload = () => {
    state.arrayBuffer = reader.result;
    $("generate").disabled = false;
    setStatus("Loaded. Set parameters and click Generate.");
  };
  reader.onerror = () => setStatus("Could not read file.", true);
  reader.readAsArrayBuffer(file);
}

function wireFileInput() {
  const drop = $("drop");
  const input = $("file");
  input.addEventListener("change", () => onFile(input.files[0]));
  ["dragenter", "dragover"].forEach((e) =>
    drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((e) =>
    drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.remove("dragover"); }));
  drop.addEventListener("drop", (ev) => onFile(ev.dataTransfer.files[0]));
}

// ---- downloads --------------------------------------------------------------------

function clearDownloads() {
  $("downloads").innerHTML = "";
}

function addDownload(name, blob, label, meta, primary = false) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  if (primary) a.classList.add("primary");
  a.innerHTML = `<span>${label}</span><span class="meta">${meta}</span>`;
  $("downloads").appendChild(a);
}

function kb(blob) {
  const n = blob.size;
  return n > 1024 * 1024 ? `${(n / 1024 / 1024).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`;
}

// ---- legend -----------------------------------------------------------------------

function renderLegend(layers) {
  const el = $("legend");
  el.innerHTML = "";
  const rows = legendRows(layers);
  if (!rows.length) return;
  const title = document.createElement("div");
  title.className = "row";
  title.innerHTML = "<strong>Paint color codes</strong>";
  el.appendChild(title);
  for (const r of rows) {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `<span class="sw" style="background:${r.hex}"></span> code ${r.color_code} - ${r.hex}`;
    el.appendChild(row);
  }
}

// ---- pipeline ---------------------------------------------------------------------

async function generate() {
  if (!state.arrayBuffer) return;
  $("generate").disabled = true;
  const p = readParams();
  const baseName = (state.fileName || "model").replace(/\.vox$/i, "");

  try {
    setStatus("Parsing .vox...");
    await nextFrame();
    const parsed = parseVox(state.arrayBuffer);
    if (!parsed.solid.any()) throw new Error("The .vox model is empty.");

    setStatus("Computing base + peg holes...");
    await nextFrame();
    const layers = computeLayers(parsed, p);

    if (p.withLabels) {
      setStatus("Loading font...");
      await labels.loadFont(FONT_URL);
    }

    setStatus("Generating tiles...");
    await nextFrame();
    const tiles = generateFaceTiles(layers, {
      withLabels: p.withLabels,
      embossDepthMm: p.embossDepthMm,
      pegClearanceMm: p.pegClearanceMm,
      pegDepthClearanceMm: p.pegDepthClearanceMm,
      tileClearanceMm: p.tileClearanceMm,
    });
    if (!tiles.length) throw new Error("No exposed faces; nothing to generate.");

    state.tiles = tiles;
    state.layers = layers;

    setStatus("Building downloads...");
    await nextFrame();
    clearDownloads();

    const threemf = await exportTiles3mf(tiles);
    if (threemf) addDownload(`${baseName}_bambu.3mf`, threemf, "Bambu / Orca 3MF", kb(threemf), true);

    if (layers.baseAligned.any()) {
      const base = baseStl(layers.baseAligned, layers.voxelSizeMm);
      addDownload(`${baseName}_base.stl`, base, "Base model STL", kb(base));
    }

    const csv = new Blob([manifestCsv(tiles)], { type: "text/csv" });
    addDownload(`${baseName}_manifest.csv`, csv, "Tile manifest CSV", kb(csv));

    if (p.withLabels && legendRows(layers).length) {
      const legend = new Blob([legendJson(layers)], { type: "application/json" });
      addDownload(`${baseName}_color_legend.json`, legend, "Color legend JSON", kb(legend));
    }

    $("downloads-card").hidden = false;

    // Viewer.
    if (!state.viewer) state.viewer = new Viewer($("viewport"));
    state.viewer.setScene(tiles, layers);
    state.viewer.setExplode(parseInt($("explode").value, 10) / 100);
    state.viewer.setBaseVisible($("show_base").checked);
    state.viewer.setNumbersVisible($("show_numbers").checked);
    state.viewer.setEdgesVisible($("show_edges").checked);
    renderLegend(layers);
    $("view-card").hidden = false;

    setStatus(`Done: ${tiles.length} tiles.`);
  } catch (err) {
    console.error(err);
    setStatus(err.message || String(err), true);
  } finally {
    $("generate").disabled = false;
  }
}

function nextFrame() {
  return new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
}

// ---- view controls ----------------------------------------------------------------

function wireViewControls() {
  $("explode").addEventListener("input", () => {
    if (state.viewer) state.viewer.setExplode(parseInt($("explode").value, 10) / 100);
  });
  $("show_base").addEventListener("change", () => {
    if (state.viewer) state.viewer.setBaseVisible($("show_base").checked);
  });
  $("show_numbers").addEventListener("change", () => {
    if (state.viewer) state.viewer.setNumbersVisible($("show_numbers").checked);
  });
  $("show_edges").addEventListener("change", () => {
    if (state.viewer) state.viewer.setEdgesVisible($("show_edges").checked);
  });
}

wireFileInput();
wireViewControls();
$("generate").addEventListener("click", generate);
