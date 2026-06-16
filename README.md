# Voxel Skin Tiler

Turn a MagicaVoxel `.vox` model into flat, mitered, multi-color **skin tiles** that snap onto a printed base model - one tile per exposed voxel face, each with a registration peg and an optional flush two-color assembly number / paint color-code inlay.

The tool (`web/`) runs **entirely client-side** (no server) and is hosted on GitHub Pages. Upload a `.vox`, set parameters, preview in 3D with an explode slider, and download a Bambu/Orca `.3mf`, STLs, a tile manifest, and a color legend.

The repo includes a few sample `.vox` models (`test.vox`, `pug.vox`, `scene.vox`) you can load to try it out.

## Web tool

Live site: `https://greigs.github.io/voxel2/` (published by the GitHub Actions workflow on push).

It is a static Vite app (vanilla JS + three.js):

1. parse `.vox` -> scale -> erode (tile thickness `T`) -> carve peg holes -> crop (`src/voxParser.js`, `src/voxelOps.js`)
2. generate one analytic tile per exposed face with 45-degree miters + a central peg (`src/tiles.js`)

   The peg can carry **crush ribs** - thin triangular ridges on each peg face that locally interfere with the base hole and deform on insertion, so the fit stays snug and consistent across print orientations. Controls (Parameters panel): `Peg ribs (per face)` (default 1, set 0 to disable), `Peg rib height (mm)` (default 0.2), `Peg rib width (mm)` (default 1). Net crush per rib = `rib height - peg clearance` (defaults give 0.1 mm). If snug orientations are too tight, raise peg clearance; if loose ones still slip, raise rib height.
3. add flush two-color number inlays with no 3D boolean - the flat top is rebuilt as `square - text` plus a shallow text pocket, and the number fills it exactly (`src/labels.js`, using opentype.js + polygon-clipping + earcut)
4. export a grouped, two-extruder Bambu/Orca `.3mf`, plus STLs / manifest CSV / legend JSON (`src/exporters/`)
5. preview everything in three.js with an explode slider, base toggle, and color legend (`src/viewer.js`)

### Run locally

```bash
cd web
npm install
npm run dev      # open the printed localhost URL
npm run build    # production build into web/dist
node test_node.mjs   # geometry sanity check (all tiles closed 2-manifolds)
```

### Deploy to GitHub Pages

`/.github/workflows/deploy-pages.yml` builds `web/` and publishes `web/dist` on every push that touches `web/`. In the repo, enable **Settings -> Pages -> Build and deployment -> Source: GitHub Actions**. The workflow sets Vite's `base` to `/<repo-name>/` automatically; for local builds you can override with `VITE_BASE`.

## License

This project is unlicensed. (Or specify your license if you have one).
