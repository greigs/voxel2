// Export skin tiles as a Bambu Studio / OrcaSlicer ready 3MF. Port of export_bambu_3mf.py.
//
// Each tile is one grouped object made of two parts: the body (extruder 1) and the number
// inlay (extruder 2). Tiles are laid FLAT (colored outer face on the plate, peg up) on a
// grid so the two-color face prints without supports and stays grouped with its number.

import JSZip from "jszip";
import { alignVectors } from "../mesh.js";

const CONTENT_TYPES = `<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
 <Default Extension="png" ContentType="image/png"/>
</Types>
`;

const RELS = `<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
`;

// Build the <mesh> block as a single string. Pushing one array entry per vertex/triangle
// would create tens of millions of tiny strings for large models and run the tab out of
// memory, so accumulate locally and emit one chunk.
function meshXml(indexed) {
  const parts = ["   <mesh>\n    <vertices>\n"];
  for (const [x, y, z] of indexed.vertices) {
    parts.push("     <vertex x=\"", x.toFixed(4), "\" y=\"", y.toFixed(4), "\" z=\"", z.toFixed(4), "\"/>\n");
  }
  parts.push("    </vertices>\n    <triangles>\n");
  for (const [a, b, c] of indexed.faces) {
    parts.push("     <triangle v1=\"", a, "\" v2=\"", b, "\" v3=\"", c, "\"/>\n");
  }
  parts.push("    </triangles>\n   </mesh>\n");
  return parts.join("");
}

/**
 * Build a Bambu/Orca 3MF Blob from tiles. Async (zips with JSZip).
 */
export async function exportTiles3mf(tiles, opts = {}) {
  const { plateWidthMm = 250.0, marginMm = 2.0, bodyExtruder = 1, numberExtruder = 2 } = opts;
  if (!tiles.length) return null;

  const targetDown = [0, 0, -1];

  // Grid spacing from the (unrotated) tile footprints. Computed without cloning so we
  // never hold every rotated tile in memory at once (large models would OOM the tab).
  let cell = 0;
  for (const t of tiles) {
    const e = t.mesh.extents();
    cell = Math.max(cell, e[0], e[1]);
  }
  cell += marginMm;
  const perRow = Math.max(1, Math.floor(plateWidthMm / cell));

  const model = [
    '<?xml version="1.0" encoding="UTF-8"?>\n',
    '<model unit="millimeter" xml:lang="en-US" ' +
      'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" ' +
      'xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06">\n',
    ' <metadata name="Application">voxel-paint-by-numbers</metadata>\n',
    ' <resources>\n',
  ];
  const build = [' <build>\n'];
  const cfg = ['<?xml version="1.0" encoding="UTF-8"?>\n<config>\n'];

  let nextId = 2;
  tiles.forEach((t, i) => {
    const col = i % perRow;
    const row = Math.floor(i / perRow);
    const cx = col * cell + marginMm;
    const cy = row * cell + marginMm;

    const n = [0, 0, 0];
    n[t.axis] = t.sign;
    const R = alignVectors(n, targetDown);
    const body = t.mesh.clone().applyRotation(R);
    // Second-extruder part = front number inlay + back number/arrow inlay (same color),
    // merged into one part before rotation.
    let number = null;
    if (t.numberMesh) number = t.numberMesh.clone();
    if (t.backMesh) {
      if (number) number.append(t.backMesh);
      else number = t.backMesh.clone();
    }
    if (number) number.applyRotation(R);

    const meshes = [body];
    if (number) meshes.push(number);
    const lo = [Infinity, Infinity, Infinity];
    for (const m of meshes) {
      const b = m.bounds();
      for (let k = 0; k < 3; k++) lo[k] = Math.min(lo[k], b.min[k]);
    }
    const shift = [cx - lo[0], cy - lo[1], -lo[2]];
    for (const m of meshes) m.translate(shift);

    const bodyId = nextId++;
    model.push(`  <object id="${bodyId}" type="model">\n`);
    model.push(meshXml(body.toIndexed()));
    model.push("  </object>\n");

    let numId = null;
    if (number) {
      numId = nextId++;
      model.push(`  <object id="${numId}" type="model">\n`);
      model.push(meshXml(number.toIndexed()));
      model.push("  </object>\n");
    }

    const groupId = nextId++;
    model.push(`  <object id="${groupId}" type="model">\n   <components>\n`);
    model.push(`    <component objectid="${bodyId}"/>\n`);
    if (numId != null) model.push(`    <component objectid="${numId}"/>\n`);
    model.push("   </components>\n  </object>\n");

    build.push(`  <item objectid="${groupId}" transform="1 0 0 0 1 0 0 0 1 0 0 0"/>\n`);

    const tag = String(t.number).padStart(4, "0");
    cfg.push(`  <object id="${groupId}">\n`);
    cfg.push(`   <metadata key="name" value="tile_${tag}"/>\n`);
    cfg.push(`   <part id="${bodyId}" subtype="normal_part">\n`);
    cfg.push(`    <metadata key="name" value="tile_${tag}_body"/>\n`);
    cfg.push(`    <metadata key="extruder" value="${bodyExtruder}"/>\n`);
    cfg.push("   </part>\n");
    if (numId != null) {
      cfg.push(`   <part id="${numId}" subtype="normal_part">\n`);
      cfg.push(`    <metadata key="name" value="tile_${tag}_number c${t.colorCode}"/>\n`);
      cfg.push(`    <metadata key="extruder" value="${numberExtruder}"/>\n`);
      cfg.push("   </part>\n");
    }
    cfg.push("  </object>\n");
  });

  model.push(" </resources>\n");
  build.push(" </build>\n");
  model.push(build.join(""));
  model.push("</model>\n");
  cfg.push("</config>\n");

  const zip = new JSZip();
  zip.file("[Content_Types].xml", CONTENT_TYPES);
  zip.file("_rels/.rels", RELS);
  zip.file("3D/3dmodel.model", model.join(""));
  zip.file("Metadata/model_settings.config", cfg.join(""));
  return zip.generateAsync({ type: "blob", compression: "DEFLATE" });
}
