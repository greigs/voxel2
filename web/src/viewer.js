// Interactive three.js viewer: colored tile bodies + number inlays, an optional base
// model, an explode slider (each tile slides along its own face normal, matching the
// fixed view_tiles.py behavior), and base/number toggles.

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { paletteRgb } from "./voxParser.js";
import { gridSurfaceMesh } from "./exporters/stl.js";

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

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const key = new THREE.DirectionalLight(0xffffff, 0.9);
    key.position.set(1, 1, 2);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.4);
    fill.position.set(-1.5, -1, 0.5);
    this.scene.add(fill);

    this.tileGroups = []; // { group, normal }
    this.numberMeshes = [];
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
  }

  setScene(tiles, layers, numberColorHex = 0x12161c) {
    this.clear();

    const numberMat = new THREE.MeshStandardMaterial({
      color: numberColorHex, roughness: 0.6, metalness: 0.0,
    });

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
      mat = new THREE.MeshStandardMaterial({
        color, roughness: 0.75, metalness: 0.0, side: THREE.DoubleSide,
      });
      matByCode.set(cc, mat);
      return mat;
    };

    for (const t of tiles) {
      const group = new THREE.Group();
      group.add(new THREE.Mesh(geomFromMesh(t.mesh), materialForCode(t.colorCode)));
      if (t.numberMesh) {
        const nm = new THREE.Mesh(geomFromMesh(t.numberMesh), numberMat);
        group.add(nm);
        this.numberMeshes.push(nm);
      }
      const normal = new THREE.Vector3();
      normal.setComponent(t.axis, t.sign);
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
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  _animate() {
    requestAnimationFrame(() => this._animate());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
