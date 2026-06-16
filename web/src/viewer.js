// Interactive three.js viewer: colored tile bodies + number inlays, an optional base
// model, an explode slider (each tile slides along its own face normal, matching the
// fixed view_tiles.py behavior), and base/number toggles.

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import { paletteRgb } from "./voxParser.js";
import { gridSurfaceMesh } from "./exporters/stl.js";

// White number-border thickness in world units (mm). Real, zoom-scaling line width.
const NUMBER_OUTLINE_WIDTH_MM = 0.08;

function geomFromMesh(mesh) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(Float32Array.from(mesh.positions), 3));
  g.computeVertexNormals();
  return g;
}

export class Viewer {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0d1117);

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100000);
    this.camera.up.set(0, 0, 1);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;

    // Bright, even illumination so white tiles read as white (a physically-based
    // material only reaches the light intensity that hits it). Hemisphere gives flat
    // fill; the directionals add form.
    const hemi = new THREE.HemisphereLight(0xffffff, 0x444a52, 2.0);
    this.scene.add(hemi);
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(1, 1, 2);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.8);
    fill.position.set(-1.5, -1, 0.5);
    this.scene.add(fill);
    const back = new THREE.DirectionalLight(0xffffff, 0.5);
    back.position.set(0, 0, -1);
    this.scene.add(back);

    this.tileGroups = []; // { group, normal }
    this.numberMeshes = [];
    this.edgeSegments = [];
    this.edgeMat = new THREE.LineBasicMaterial({ color: 0x14181d });
    this.edgesVisible = true;
    // Preview-only thin white border around number glyphs, for legibility on dark tiles.
    // Fat-line material so the width is real (world units) and controllable.
    this.numberOutlineMat = new LineMaterial({
      color: 0xffffff, linewidth: NUMBER_OUTLINE_WIDTH_MM, worldUnits: true,
      alphaToCoverage: true,
    });
    this.tilesRoot = new THREE.Group();
    this.scene.add(this.tilesRoot);
    this.baseRoot = new THREE.Group();
    this.scene.add(this.baseRoot);

    this.maxExplode = 10;

    this._resize = this._resize.bind(this);
    window.addEventListener("resize", this._resize);
    this._resize();
    this._animate();
  }

  clear() {
    const dispose = (obj) => {
      obj.traverse((c) => {
        if (c.geometry) c.geometry.dispose();
        if (c.material) c.material.dispose();
      });
    };
    dispose(this.tilesRoot);
    dispose(this.baseRoot);
    this.tilesRoot.clear();
    this.baseRoot.clear();
    this.tileGroups = [];
    this.numberMeshes = [];
    this.edgeSegments = [];
  }

  setScene(tiles, layers, numberColorHex = 0x12161c) {
    this.clear();

    // Unlit materials -> perfectly uniform, orientation-independent color (each tile
    // shows its exact paint color). Scene lights have no effect on these.
    const numberMat = new THREE.MeshBasicMaterial({ color: numberColorHex });

    // Real paint color per tile, from the voxel palette via the color-code legend. One
    // shared material per color keeps large models light.
    const legend = (layers && layers.colorLegend) || new Map();
    const matByCode = new Map();
    const materialForCode = (cc) => {
      let mat = matByCode.get(cc);
      if (mat) return mat;
      const color = new THREE.Color();
      if (legend.has(cc)) {
        const [r, g, b] = paletteRgb(layers.paletteByIndex, legend.get(cc));
        color.setRGB(r / 255, g / 255, b / 255, THREE.SRGBColorSpace);
      } else {
        color.setRGB(0.6, 0.6, 0.6, THREE.SRGBColorSpace);
      }
      mat = new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide });
      matByCode.set(cc, mat);
      return mat;
    };

    for (const t of tiles) {
      const group = new THREE.Group();
      const normal = new THREE.Vector3().setComponent(t.axis, t.sign);
      const bodyGeom = geomFromMesh(t.mesh);
      group.add(new THREE.Mesh(bodyGeom, materialForCode(t.colorCode)));
      // Thin dark outlines convey shape/edges while colors stay uniform (unlit). Built
      // from the clean tile (outlineMesh) so the number outlines are NOT included.
      const outlineGeom = geomFromMesh(t.outlineMesh || t.mesh);
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(outlineGeom, 20), this.edgeMat);
      outlineGeom.dispose();
      edges.visible = this.edgesVisible;
      group.add(edges);
      this.edgeSegments.push(edges);
      if (t.numberMesh) {
        const numGeom = geomFromMesh(t.numberMesh);
        const nm = new THREE.Mesh(numGeom, numberMat);
        group.add(nm);
        this.numberMeshes.push(nm);
        // Thin white border around the glyphs (the number's top outline), preview only.
        // Nudge it outward along the face normal so it stays proud of the coplanar number
        // surface and isn't hidden by z-fighting when viewed up close.
        const eg = new THREE.EdgesGeometry(numGeom, 20);
        const lsg = new LineSegmentsGeometry().fromEdgesGeometry(eg);
        eg.dispose();
        const outline = new LineSegments2(lsg, this.numberOutlineMat);
        outline.position.copy(normal).multiplyScalar(0.05);
        group.add(outline);
        this.numberMeshes.push(outline);
      }
      this.tilesRoot.add(group);
      this.tileGroups.push({ group, normal });
    }

    // Base model from the carved/eroded grid, in the same scaled-mm coordinate space.
    if (layers && layers.baseAligned && layers.baseAligned.any()) {
      const baseMesh = gridSurfaceMesh(layers.baseAligned, layers.voxelSizeMm);
      const mat = new THREE.MeshStandardMaterial({
        color: 0x8b949e, roughness: 0.9, metalness: 0.0, side: THREE.DoubleSide,
      });
      this.baseRoot.add(new THREE.Mesh(geomFromMesh(baseMesh), mat));
    }

    // Explode distance scaled to the model size.
    const L = layers ? layers.scaleFactorInt * layers.voxelSizeMm : 10;
    this.maxExplode = Math.max(L * 1.5, 5);

    this._frame();
  }

  setExplode(t01) {
    const d = t01 * this.maxExplode;
    for (const { group, normal } of this.tileGroups) {
      group.position.copy(normal).multiplyScalar(d);
    }
  }

  setBaseVisible(v) { this.baseRoot.visible = v; }
  setNumbersVisible(v) { for (const m of this.numberMeshes) m.visible = v; }
  setEdgesVisible(v) { this.edgesVisible = v; for (const e of this.edgeSegments) e.visible = v; }

  _frame() {
    const box = new THREE.Box3().setFromObject(this.tilesRoot);
    if (box.isEmpty()) return;
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const radius = Math.max(size.x, size.y, size.z) * 0.5 || 10;
    const dist = radius / Math.sin((this.camera.fov * Math.PI) / 180 / 2) * 1.6;
    this.controls.target.copy(center);
    this.camera.position.set(center.x + dist * 0.7, center.y - dist, center.z + dist * 0.7);
    this.camera.near = Math.max(0.1, dist / 1000);
    this.camera.far = dist * 1000;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  _resize() {
    const w = this.container.clientWidth || 1;
    const h = this.container.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    if (this.numberOutlineMat) this.numberOutlineMat.resolution.set(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  _animate() {
    requestAnimationFrame(() => this._animate());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
