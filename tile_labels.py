"""Flat, two-color number inlays for skin tiles.

Each tile gets two numbers written on its flat outer face:
  * an assembly id (unique per tile) so you know where each printed tile goes, and
  * a paint color-code so you know which color to paint that tile.

The numbers are produced as a thin flat body that is coplanar with the tile's outer
surface (it occupies the top ``emboss_depth`` mm and its outer face is flush with the
tile face - never recessed, never proud). The matching cavity is cut from the tile body
so the two parts butt together with no overlap, ready for a two-filament AMS print
(tile = base filament, numbers = text filament).

Glyph outlines come from matplotlib's bundled DejaVu Sans (no system font needed),
are turned into shapely polygons (even-odd fill, so holes in 0/6/8 work), extruded
with trimesh, and cut from the tile with the manifold boolean backend.
"""

from functools import lru_cache

import numpy as np
import trimesh


@lru_cache(maxsize=1)
def _font_prop():
    from matplotlib.font_manager import FontProperties
    return FontProperties(family="DejaVu Sans", weight="bold")


@lru_cache(maxsize=512)
def _glyph_polygon(text):
    """Return a shapely (Multi)Polygon for ``text`` at nominal size 1.0, or None."""
    from matplotlib.textpath import TextPath
    from shapely.geometry import Polygon

    tp = TextPath((0, 0), text, size=1.0, prop=_font_prop())
    poly = None
    for ring in tp.to_polygons(closed_only=True):
        if len(ring) < 3:
            continue
        p = Polygon(ring).buffer(0)
        poly = p if poly is None else poly.symmetric_difference(p)
    return poly


def _fit(poly, target_w, target_h, cx, cy):
    """Scale ``poly`` uniformly to fit a target box and recenter it at (cx, cy)."""
    from shapely.affinity import scale, translate

    minx, miny, maxx, maxy = poly.bounds
    w, h = maxx - minx, maxy - miny
    if w <= 0 or h <= 0:
        return poly
    s = min(target_w / w, target_h / h)
    poly = scale(poly, xfact=s, yfact=s, origin=(0, 0))
    minx, miny, maxx, maxy = poly.bounds
    return translate(poly, xoff=cx - (minx + maxx) / 2.0, yoff=cy - (miny + maxy) / 2.0)


def label_polygon(lines, face_size, margin_frac=0.16):
    """Build a centered, multi-line text polygon (in face u,v mm) for ``lines``.

    The block is fit inside a centered square of side ``face_size * (1 - 2*margin)``.
    Returns a shapely (Multi)Polygon, or None if nothing renders.
    """
    from shapely.ops import unary_union

    lines = [str(s) for s in lines if str(s) != ""]
    if not lines:
        return None

    avail = face_size * (1.0 - 2.0 * margin_frac)
    n = len(lines)
    line_h = avail / n * 0.78
    gap = avail / n * 0.22
    total_h = n * line_h + (n - 1) * gap
    y_top = total_h / 2.0 - line_h / 2.0

    parts = []
    for i, ln in enumerate(lines):
        g = _glyph_polygon(ln)
        if g is None or g.is_empty:
            continue
        cy = y_top - i * (line_h + gap)
        parts.append(_fit(g, avail, line_h, 0.0, cy))
    if not parts:
        return None
    return unary_union(parts)


def _extrude(poly, height):
    """Extrude a shapely (Multi)Polygon to a trimesh solid of the given height."""
    geoms = list(poly.geoms) if poly.geom_type == "MultiPolygon" else [poly]
    meshes = []
    for g in geoms:
        if g.is_empty or g.area <= 0:
            continue
        meshes.append(trimesh.creation.extrude_polygon(g, height=height))
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


def face_transform(C, uhat, vhat, w, L):
    """4x4 transform mapping local (u, v, depth) -> world on a tile face.

    Local x->u tangent, y->v tangent, z->w (inward); origin at the outer-face center.
    """
    M = np.eye(4)
    M[:3, 0] = uhat
    M[:3, 1] = vhat
    M[:3, 2] = w
    M[:3, 3] = np.asarray(C, float) + np.asarray(uhat, float) * (L / 2.0) \
        + np.asarray(vhat, float) * (L / 2.0)
    return M


def make_label_meshes(poly, C, uhat, vhat, w, L, emboss_depth, eps=0.05):
    """Return ``(number_mesh, cut_mesh)`` for a face given its label polygon.

    ``number_mesh`` is the flush inlay (outer face at depth 0, extends inward to
    ``emboss_depth``). ``cut_mesh`` is the same shape extended ``eps`` past the surface
    so a boolean difference cleanly removes the cavity from the tile body.
    """
    if poly is None or poly.is_empty:
        return None, None
    number = _extrude(poly, emboss_depth)
    cut = _extrude(poly, emboss_depth + eps)
    if number is None or cut is None:
        return None, None
    cut.apply_translation([0.0, 0.0, -eps])  # poke slightly out of the surface

    M = face_transform(C, uhat, vhat, w, L)
    number.apply_transform(M)
    cut.apply_transform(M)
    return number, cut


def apply_label(tile_mesh, poly, C, uhat, vhat, w, L, emboss_depth):
    """Cut the label cavity from ``tile_mesh`` and return ``(body, number_mesh)``.

    On any boolean failure, falls back to the original tile with no number so a single
    bad glyph never aborts a whole run.
    """
    number, cut = make_label_meshes(poly, C, uhat, vhat, w, L, emboss_depth)
    if number is None:
        return tile_mesh, None
    try:
        body = tile_mesh.difference(cut)
        if body is None or body.is_empty or not body.is_watertight:
            return tile_mesh, number
        return body, number
    except Exception:
        return tile_mesh, number
