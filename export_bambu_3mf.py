"""Export skin tiles as a Bambu Studio / OrcaSlicer ready 3MF.

Each tile becomes a single grouped object made of two parts:
  * the tile body  -> extruder/filament 1 (your base color), and
  * the number inlay -> extruder/filament 2 (your text color).

Because body + number are one grouped object, they stay together when you Arrange
the plate or move tiles around. Every tile is laid FLAT with its colored outer face
on the build plate (the peg points up), so the two-color face prints cleanly with no
supports, and the assembly/color numbers stay locked to their tile.

The file follows the Bambu/Orca 3MF layout: geometry in ``3D/3dmodel.model`` (grouped
via <components>) and per-part filament in ``Metadata/model_settings.config`` using the
documented ``<part ...><metadata key="extruder" .../></part>`` structure.
"""

import os
import zipfile

import numpy as np
import trimesh


CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
 <Default Extension="png" ContentType="image/png"/>
</Types>
"""

RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"""


def _mesh_xml(mesh, out):
    out.append("   <mesh>\n    <vertices>\n")
    v = np.asarray(mesh.vertices, dtype=float)
    for x, y, z in v:
        out.append(f'     <vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>\n')
    out.append("    </vertices>\n    <triangles>\n")
    f = np.asarray(mesh.faces, dtype=int)
    for a, b, c in f:
        out.append(f'     <triangle v1="{a}" v2="{b}" v3="{c}"/>\n')
    out.append("    </triangles>\n   </mesh>\n")


def export_tiles_3mf(tile_list, path, plate_width_mm=250.0, margin_mm=2.0,
                     body_extruder=1, number_extruder=2):
    """Write ``tile_list`` to a Bambu/Orca 3MF at ``path`` (tiles laid flat, grouped)."""
    if not tile_list:
        print("No tiles to export to 3MF.")
        return False

    target_down = np.array([0.0, 0.0, -1.0])  # outer face onto the plate

    # Grid layout: wrap tiles across rows up to plate_width_mm.
    cell = 0.0
    for t in tile_list:
        cell = max(cell, float(t.mesh.extents[:2].max()))
    cell += margin_mm
    per_row = max(1, int(plate_width_mm // cell))

    model = [
        '<?xml version="1.0" encoding="UTF-8"?>\n',
        '<model unit="millimeter" xml:lang="en-US" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
        'xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06">\n',
        ' <metadata name="Application">voxel2-tiler</metadata>\n',
        ' <resources>\n',
    ]
    build = [' <build>\n']
    cfg = ['<?xml version="1.0" encoding="UTF-8"?>\n<config>\n']

    next_id = 2
    for i, t in enumerate(tile_list):
        n = np.zeros(3)
        n[t.axis] = float(t.sign)
        R = trimesh.geometry.align_vectors(n, target_down)

        col = i % per_row
        row = i // per_row
        cx = col * cell + margin_mm
        cy = row * cell + margin_mm

        body = t.mesh.copy()
        body.apply_transform(R)
        meshes = [body]
        num = None
        if t.number_mesh is not None:
            num = t.number_mesh.copy()
            num.apply_transform(R)
            meshes.append(num)

        # Drop onto the plate (z>=0) and shift into the grid cell.
        lo = np.min([m.bounds[0] for m in meshes], axis=0)
        shift = np.array([cx - lo[0], cy - lo[1], -lo[2]])
        for m in meshes:
            m.apply_translation(shift)

        body_id = next_id; next_id += 1
        model.append(f'  <object id="{body_id}" type="model">\n')
        _mesh_xml(body, model)
        model.append('  </object>\n')

        num_id = None
        if num is not None:
            num_id = next_id; next_id += 1
            model.append(f'  <object id="{num_id}" type="model">\n')
            _mesh_xml(num, model)
            model.append('  </object>\n')

        group_id = next_id; next_id += 1
        model.append(f'  <object id="{group_id}" type="model">\n   <components>\n')
        model.append(f'    <component objectid="{body_id}"/>\n')
        if num_id is not None:
            model.append(f'    <component objectid="{num_id}"/>\n')
        model.append('   </components>\n  </object>\n')

        build.append(f'  <item objectid="{group_id}" '
                     'transform="1 0 0 0 1 0 0 0 1 0 0 0"/>\n')

        cfg.append(f'  <object id="{group_id}">\n')
        cfg.append(f'   <metadata key="name" value="tile_{t.number:04d}"/>\n')
        cfg.append(f'   <part id="{body_id}" subtype="normal_part">\n')
        cfg.append(f'    <metadata key="name" value="tile_{t.number:04d}_body"/>\n')
        cfg.append(f'    <metadata key="extruder" value="{body_extruder}"/>\n')
        cfg.append('   </part>\n')
        if num_id is not None:
            cfg.append(f'   <part id="{num_id}" subtype="normal_part">\n')
            cfg.append(f'    <metadata key="name" value="tile_{t.number:04d}_number c{t.color_code}"/>\n')
            cfg.append(f'    <metadata key="extruder" value="{number_extruder}"/>\n')
            cfg.append('   </part>\n')
        cfg.append('  </object>\n')

    model.append(' </resources>\n')
    build.append(' </build>\n')
    model.append("".join(build))
    model.append('</model>\n')
    cfg.append('</config>\n')

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", RELS)
        z.writestr("3D/3dmodel.model", "".join(model))
        z.writestr("Metadata/model_settings.config", "".join(cfg))

    print(f"Wrote Bambu 3MF to '{path}' ({len(tile_list)} tiles, flat, "
          f"body=extruder {body_extruder}, numbers=extruder {number_extruder}).")
    return True
