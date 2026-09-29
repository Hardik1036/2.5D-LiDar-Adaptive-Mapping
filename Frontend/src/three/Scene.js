import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

/** Visualization modes. Values are the display labels shown in Scene Controls. */
export const MODE_POINTS = "Raw LiDAR Point Cloud";
export const MODE_ADAPTIVE = "2.5D Adaptive Mapping";
export const MODE_SEMANTIC = "Semantic Hazard Intelligence";
export const MODE_PERCEPTION = "Dynamic Perception & Tracking";
export const MODES = [
  MODE_POINTS,
  MODE_ADAPTIVE,
  MODE_SEMANTIC,
  MODE_PERCEPTION,
];

export const DEFAULT_OPTIONS = {
  size: 1,
  minHeight: -5.0,
  maxHeight: 10.0,
  mode: MODE_ADAPTIVE,
  colorBy: "cost",
  ground: true,
  grid: true,
  axes: false,
  playbackSpeed: 1.0,
  filterMode: "all",
};

/**
 * Enforces DRDO-standard cost band colors.
 * 1. Primary Path: Semantic Cost emitted by backend
 * 2. Fallback Path: Absolute Height relative to ground plane
 * Prohibit elevated voxels/cells (Z > -1.2 m) from rendering in green (0x238636).
 */
export function getCostColor(cost, z) {
  const zNum = (z !== undefined && z !== null) ? Number(z) : -999.0;
  const c = Number(cost ?? 0);

  // Ceiling Gate: Returns above vehicle floor clearance (Z > -1.15m) cannot be green
  if (!isNaN(zNum) && zNum > -1.15 && c <= 50) {
    return 0xF85149; // Tactical Crimson Red: Lethal Obstacle
  }

  // Strict 3-Tier Defense Banding
  if (c <= 50) {
    return 0x238636; // Emerald Green: Safe Traversable Corridor (Cost 0-50)
  } else if (c <= 180) {
    return 0xD29922; // Solar Amber Yellow: Caution Gap (Cost 51-180)
  } else {
    return 0xF85149; // Tactical Crimson Red: Lethal Hazard / Curbs / Walls (Cost 181-255)
  }
}

const palette = [
  new THREE.Color(0x238636),
  new THREE.Color(0xD29922),
  new THREE.Color(0xF85149),
];
export const costBand = (cost) => (cost <= 50 ? 0 : cost <= 180 ? 1 : 2);

/** Semantic classes are derived from the tracked object's own class string.
 * No object category is invented that the telemetry frame does not report. */
export const classOf = (name = "") =>
  /vehicle|car|truck|bus/i.test(name)
    ? "vehicle"
    : /pedestrian|person|human/i.test(name)
      ? "pedestrian"
      : /animal|dog|cow|cattle/i.test(name)
        ? "animal"
        : "obstacle";
const semanticColor = {
  vehicle: new THREE.Color("#F85149"),
  pedestrian: new THREE.Color("#D29922"),
  animal: new THREE.Color("#D29922"),
  obstacle: new THREE.Color("#F85149"),
};
const footprintOf = (object) => {
  if (Array.isArray(object.dimensions) && object.dimensions.length >= 3) {
    return {
      w: object.dimensions[0],
      l: object.dimensions[1],
      h: object.dimensions[2],
    };
  }
  const k = classOf(object.class);
  
  return k === "vehicle"
    ? { w: 1.9, l: 4.2, h: 1.6 }
    : k === "animal"
      ? { w: 0.7, l: 1.2, h: 0.9 }
      : k === "pedestrian"
        ? { w: 0.6, l: 0.6, h: 1.75 }
        : { w: 0.9, l: 0.9, h: 0.9 };
};
/** Stable pseudo-random offsets so object point clusters do not flicker per frame. */
const jitter = (i, salt) => {
  const v = Math.sin(i * 12.9898 + salt * 78.233) * 43758.5453;
  return v - Math.floor(v) - 0.5;
};

/** Backend (x,y,z-up) -> Three (x,z-up,-y). Heading: radians CCW from +X.
 * All live/fallback frames use setFrame(). No sockets or fabricated stats here.
 */
export class LidarScene {
  constructor(container, onError) {
    this.container = container;
    this.options = { ...DEFAULT_OPTIONS };
    this.minHeight = DEFAULT_OPTIONS.minHeight;
    this.maxHeight = DEFAULT_OPTIONS.maxHeight;
    this.actors = new Map();
    this.dynamicTracks = this.actors;
    this.resources = new Set();
    this.frame = null;
    this.dirty = false;
    this.capacity = 0;
    this.disposed = false;
    this.vehicleMesh = null;
    this.vehicleGroup = null;
    this.egoGroup = null;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color("#050811");
    this.camera = new THREE.PerspectiveCamera(48, 1, 0.1, 500);
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: "high-performance",
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.domElement.setAttribute(
      "aria-label",
      "Interactive 3D telemetry scene. Drag to orbit, scroll to zoom, right-drag to pan.",
    );
    this.renderer.domElement.setAttribute("role", "img");
    container.appendChild(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.minDistance = 3;
    this.controls.maxDistance = 180;
    this.controls.maxPolarAngle = Math.PI / 2 - 0.01;
    this.userInteracted = false;
    this.hasFramedSweep = false;
    this.sweepBoundingBox = null;
    this.sweepCenter = null;
    this.sweepSize = null;
    this.controls.addEventListener("start", () => {
      this.userInteracted = true;
    });
    this.resetCamera();
    this.scene.add(new THREE.HemisphereLight(0xe0edff, 0x344254, 2));
    const light = new THREE.DirectionalLight(0xffffff, 2.5);
    light.position.set(10, 25, 10);
    this.scene.add(light);
    this.ground = new THREE.Mesh(
      this.track(new THREE.PlaneGeometry(240, 240)),
      this.track(
        new THREE.MeshBasicMaterial({
          color: "#080C14",
          side: THREE.DoubleSide,
        }),
      ),
    );
    this.ground.rotation.x = -Math.PI / 2;
    this.ground.position.y = -2.1;
    this.scene.add(this.ground);
    this.grid = new THREE.GridHelper(240, 120, 0x33445a, 0x192638);
    this.grid.position.y = -2.09;
    this.scene.add(this.grid);
    this.axes = new THREE.AxesHelper(5);
    this.axes.visible = false;
    this.scene.add(this.axes);
    this.tileGeometry = this.track(new THREE.BoxGeometry(1, 1, 1));
    this.tileMaterial = this.track(
      new THREE.MeshBasicMaterial({
        side: THREE.DoubleSide,
        transparent: false,
        opacity: 1.0,
      }),
    );
    // In-shader tile boundaries make adaptive resolution legible without extra draw calls.
    this.tileMaterial.onBeforeCompile = (shader) => {
      shader.vertexShader = shader.vertexShader
        .replace("#include <common>", "#include <common>\nvarying vec2 tileUv;")
        .replace("#include <uv_vertex>", "#include <uv_vertex>\ntileUv = uv;");
      shader.fragmentShader = shader.fragmentShader
        .replace("#include <common>", "#include <common>\nvarying vec2 tileUv;")
        .replace(
          "#include <color_fragment>",
          "#include <color_fragment>\nfloat edge = min(min(tileUv.x, tileUv.y), min(1.0-tileUv.x, 1.0-tileUv.y));\nif(edge < 0.012) diffuseColor.rgb *= 0.28;",
        );
    };
    this.pointMaterial = this.track(
      new THREE.PointsMaterial({
        size: 0.24,
        vertexColors: true,
        sizeAttenuation: true,
      }),
    );
    this.matrix = new THREE.Matrix4();
    this.color = new THREE.Color();
    this.colorA = new THREE.Color();
    this.colorB = new THREE.Color();
    this.box = this.track(new THREE.BoxGeometry(1, 1, 1));
    this.capsule = this.track(new THREE.CapsuleGeometry(0.19, 0.43, 4, 8));
    this.sphere = this.track(new THREE.SphereGeometry(0.19, 10, 8));
    this.actorMat = this.track(
      new THREE.MeshStandardMaterial({ color: "#c8d1dc", roughness: 0.65 }),
    );
    this.darkMat = this.track(
      new THREE.MeshStandardMaterial({ color: "#30363d", roughness: 0.7 }),
    );
    this.hazardMat = this.track(
      new THREE.MeshBasicMaterial({
        color: "#D29922",
        transparent: true,
        opacity: 0.35,
        depthWrite: false,
        side: THREE.DoubleSide,
      }),
    );
    const shape = new THREE.Shape();
    shape.moveTo(0, 0);
    shape.lineTo(-1, 1);
    shape.quadraticCurveTo(0, 1.2, 1, 1);
    shape.lineTo(0, 0);
    this.hazardGeo = this.track(new THREE.ShapeGeometry(shape));
    this.hazardGeo.rotateX(Math.PI / 2);
    this.rawPointCloudGeometry = this.track(new THREE.BufferGeometry());
    this.rawPointCloud = new THREE.Points(
      this.rawPointCloudGeometry,
      this.pointMaterial,
    );
    this.rawPointCloud.frustumCulled = false;
    this.rawPointCloud.visible = false;
    this.scene.add(this.rawPointCloud);
    this.points = this.rawPointCloud;
    this.ensureCapacity(12000);
    this.buildPerception();
    this.reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
    this.onLost = (e) => {
      e.preventDefault();
      this.renderer.setAnimationLoop(null);
      onError("WebGL context lost. Reload to restore the 3D view.");
    };
    this.renderer.domElement.addEventListener("webglcontextlost", this.onLost);
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(container);
    this.resize();
    this.renderer.setAnimationLoop((time) => this.render(time / 1000));
  }
  track(resource) {
    if (resource && typeof resource.dispose === "function") {
      this.resources.add(resource);
    }
    return resource;
  }
  cameraPreset(mode = this.options.mode) {
    // The perception mode opens closer to the platform for presentation use.
    if (mode === MODE_PERCEPTION) {
      return { position: [24, 17, 30], target: [0, 1.2, -2] };
    }

    if (this.sweepCenter && this.sweepSize) {
      const cx = this.sweepCenter.x;
      const cy = this.sweepCenter.y;
      const cz = this.sweepCenter.z;
      const spanX = Math.max(this.sweepSize.x, 20.0);
      const spanZ = Math.max(this.sweepSize.z, 20.0);
      const span = Math.max(spanX, spanZ);

      return {
        position: [cx + span * 0.55, cy + Math.max(span * 0.70, 24.0), cz + span * 0.75],
        target: [cx, cy, cz],
      };
    }

    return { position: [42, 48, 58], target: [0, 0, -8] };
  }
  resetCamera() {
    const preset = this.cameraPreset();
    this.tween = null;
    this.camera.position.set(...preset.position);
    this.controls.target.set(...preset.target);
    this.controls.update();
  }
  /** Smooth camera move between modes: the scene is never torn down or reloaded. */
  glideTo(preset, seconds = 0.7) {
    if (this.reduced?.matches) {
      this.camera.position.set(...preset.position);
      this.controls.target.set(...preset.target);
      this.controls.update();
      return;
    }
    this.tween = {
      from: this.camera.position.clone(),
      to: new THREE.Vector3(...preset.position),
      targetFrom: this.controls.target.clone(),
      targetTo: new THREE.Vector3(...preset.target),
      elapsed: 0,
      seconds,
    };
  }
  setControlsEnabled(enabled) {
    this.controls.enabled = enabled;
  }
  cameraState() {
    return {
      position: this.camera.position.clone(),
      target: this.controls.target.clone(),
    };
  }
  restoreCamera(state) {
    if (!state) return;
    this.camera.position.copy(state.position);
    this.controls.target.copy(state.target);
    this.controls.update();
  }
  resize() {
    const w = this.container.clientWidth,
      h = this.container.clientHeight;
    if (!w || !h) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }
  setFrame(frame) {
    this.nextFrame = frame;
    this.dirty = true;
  }
  setPose(pose) {
    this.updatePose(pose);
  }
  updatePose(pose) {
    if (!pose) return;
    const x = Number(pose.x ?? pose.position?.[0] ?? 0);
    const y = Number(pose.y ?? pose.position?.[1] ?? 0);
    const z = Number(pose.z ?? pose.position?.[2] ?? 0);
    const vx = Number(pose.vx ?? pose.velocity?.[0] ?? 0);
    const vy = Number(pose.vy ?? pose.velocity?.[1] ?? 0);
    const heading = Number(pose.ego_yaw ?? pose.heading ?? pose.yaw ?? 0);
    const velocity = Math.hypot(vx, vy);

    let targetYaw = heading;
    if (velocity >= 0.1) {
      targetYaw = Math.atan2(-vy || 0, vx);
    }

    // Align vehicle model front bumper to positive X axis (+X forward)
    const finalRotY = targetYaw + Math.PI / 2;

    if (this.vehicleMesh) {
      this.vehicleMesh.position.set(x, z, -y);
      this.vehicleMesh.rotation.y = finalRotY;
    }
    if (this.vehicleGroup && this.vehicleGroup !== this.vehicleMesh) {
      this.vehicleGroup.position.set(x, z, -y);
      this.vehicleGroup.rotation.y = finalRotY;
    }
    if (this.egoGroup && this.egoGroup !== this.vehicleMesh) {
      this.egoGroup.position.set(x, z, -y);
      this.egoGroup.rotation.y = finalRotY;
    }
    if (this.sector) {
      this.sector.position.set(x, 0.02, -y);
      this.sector.rotation.y = targetYaw;
      this.sector.rotation.z = 0; // Center ray projected forward along travel corridor
    }
    if (this.rim) {
      this.rim.position.set(x, 0.03, -y);
      this.rim.rotation.y = targetYaw;
      this.rim.rotation.z = 0;
    }
    this.dirty = true;
  }
  setOptions(options) {
    const previous = this.options.mode;
    this.options = { ...options };
    if (options.minHeight !== undefined) this.minHeight = options.minHeight;
    if (options.maxHeight !== undefined) this.maxHeight = options.maxHeight;
    if (previous !== options.mode) {
      const wasPerception = previous === MODE_PERCEPTION;
      const isPerception = options.mode === MODE_PERCEPTION;
      if (wasPerception !== isPerception)
        this.glideTo(this.cameraPreset(options.mode));
      this.dirty = true;
      if (this.frame) {
        this.updateFrame();
      }
    }
    this.renderer.setPixelRatio(
      Math.min(window.devicePixelRatio || 1, options.pixelRatio || 1.5),
    );
    this.resize();
    this.dirty = true;
  }
  ensureCapacity(n) {
    if (n <= this.capacity) return;
    this.capacity = Math.max(12000, 2 ** Math.ceil(Math.log2(n)));
    if (this.tiles) {
      this.scene.remove(this.tiles);
      this.tiles.dispose();
    }
    this.tiles = new THREE.InstancedMesh(
      this.tileGeometry,
      this.tileMaterial,
      this.capacity,
    );
    this.tiles.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.tiles.frustumCulled = false;
    this.tiles.count = 0;
    this.scene.add(this.tiles);
    this.instancedTerrainMesh = this.tiles;
    this.maxTerrainCells = this.capacity;
  }
  getColor(cell, zOverride) {
    const zMean = Number(cell.z ?? cell.z_mean ?? ((Number(cell.z_min ?? -1.68) + Number(cell.z_max ?? -1.68)) * 0.5));
    const zMin = Number(cell.z_min ?? zMean);
    const zMax = Number(cell.z_max ?? zMean);
    const deltaZ = Number(cell.delta_z ?? (zMax - zMin) ?? 0.0);
    const isGroundBand = (zMean >= -2.20 && zMean <= -1.25 && deltaZ <= 0.18 && zMax <= -1.15) || Boolean(cell.is_ground);

    const effectiveCost = isGroundBand ? 0 : Number(cell.cost ?? cell.semantic_cost ?? 0);
    const z = zOverride !== undefined ? zOverride : zMean;

    if (this.options.mode === MODE_ADAPTIVE || this.options.mode === MODE_SEMANTIC || this.options.colorBy === "cost") {
      return this.color.setHex(getCostColor(effectiveCost, z));
    }
    // Height coloring fallback: adhere to DRDO standards (prohibiting green in air)
    return this.color.setHex(getCostColor(undefined, z));
  }
  mesh(geometry, material, scale, position) {
    const m = new THREE.Mesh(geometry, material);
    m.scale.set(...scale);
    m.position.set(...position);
    return m;
  }
  addActor(object) {
    const root = new THREE.Group(),
      body = new THREE.Group();
    root.add(body);
    const vehicle = /vehicle|car|truck|bus/i.test(object.class),
      human = /pedestrian|person|human/i.test(object.class);
    const limbs = [];
    if (vehicle) {
      body.add(
        this.mesh(this.box, this.actorMat, [1.7, 0.65, 3.5], [0, 0.65, 0]),
        this.mesh(this.box, this.darkMat, [1.45, 0.6, 1.7], [0, 1.22, -0.25]),
      );
      for (const x of [-0.9, 0.9])
        for (const z of [-1.1, 1.1])
          body.add(
            this.mesh(this.box, this.darkMat, [0.25, 0.48, 0.6], [x, 0.35, z]),
          );
      body.add(
        this.mesh(this.box, this.actorMat, [1.2, 0.12, 0.06], [0, 0.8, 1.78]),
      );
    } else if (human) {
      body.add(
        this.mesh(this.capsule, this.actorMat, [1, 1, 1], [0, 1.1, 0]),
        this.mesh(this.sphere, this.actorMat, [1, 1, 1], [0, 1.65, 0]),
      );
      for (const [x, y, isArm] of [
        [-0.13, 0.77, false],
        [0.13, 0.77, false],
        [-0.32, 1.35, true],
        [0.32, 1.35, true],
      ]) {
        const pivot = new THREE.Group();
        pivot.position.set(x, y, 0);
        pivot.add(
          this.mesh(
            this.box,
            this.darkMat,
            [0.14, isArm ? 0.52 : 0.65, 0.16],
            [0, isArm ? -0.26 : -0.325, 0],
          ),
        );
        body.add(pivot);
        limbs.push(pivot);
      }
    } else
      body.add(
        this.mesh(this.box, this.actorMat, [0.7, 0.7, 0.7], [0, 0.35, 0]),
      );
    const hazard = new THREE.Mesh(this.hazardGeo, this.hazardMat);
    this.scene.add(root, hazard);
    const actor = { root, body, hazard, limbs, vehicle, human, object, hazardCone: hazard };
    root.hazardCone = hazard;
    this.actors.set(object.id, actor);
    return actor;
  }
  getOrCreateTrackAvatar(id, className) {
    let a = this.actors.get(id);
    if (!a) {
      a = this.addActor({ id, class: className });
    }
    if (a) {
      a.hazardCone = a.hazard;
      if (a.root) a.root.hazardCone = a.hazard;
      return a.root;
    }
    return null;
  }
  updateFrame() {
    const { frame, options: o } = this;
    if (!frame) return;
    const isPoints = o.mode === MODE_POINTS;
    const isAdaptive = o.mode === MODE_ADAPTIVE;
    const isSemantic = o.mode === MODE_SEMANTIC;
    const isPerception = o.mode === MODE_PERCEPTION;
    const influence = isAdaptive;
    const showStatic = o.filterMode !== "dynamic";
    const showDynamic = o.filterMode !== "static";

    const cells = frame.cells || [];
    const objects = frame.dynamic_objects || [];

    const floor = cells.reduce((z, c) => Math.min(z, c.z_min), 0) - 0.5;
    this.ground.position.y = floor;
    this.grid.position.y = floor + 0.01;

    // --- MODE 1: RAW LIDAR POINT CLOUD RENDERING PATH ---
    if (isPoints) {
      if (this.tiles) this.tiles.visible = false;
      this.rawPointCloud.visible = isPoints && showStatic;

      // Check if raw_points are present in incoming telemetry frame, or synthesize from cells
      let flatPoints = (frame.raw_points && frame.raw_points.length > 0)
        ? frame.raw_points
        : null;

      if (!flatPoints && cells.length > 0) {
        const synth = [];
        for (let cIdx = 0; cIdx < cells.length; cIdx++) {
          const c = cells[cIdx];
          const zMin = c.z_min ?? -1.68;
          const zMax = c.z_max ?? (zMin + 0.08);
          const halfS = (c.size || 1.0) * 0.35;
          synth.push(c.x, c.y, zMin);
          if (zMax > zMin + 0.05) {
            synth.push(c.x + halfS, c.y + halfS, zMax);
            synth.push(c.x - halfS, c.y - halfS, (zMin + zMax) * 0.5);
          }
        }
        flatPoints = synth;
      }

      if (flatPoints && flatPoints.length > 0) {
        const numPoints = Math.floor(flatPoints.length / 3);

        // Allocate or resize geometry attributes if point count changed
        let posAttr = this.rawPointCloudGeometry.getAttribute('position');
        let colAttr = this.rawPointCloudGeometry.getAttribute('color');

        if (!posAttr || posAttr.count !== numPoints) {
          posAttr = new THREE.BufferAttribute(new Float32Array(numPoints * 3), 3);
          colAttr = new THREE.BufferAttribute(new Float32Array(numPoints * 3), 3);
          this.rawPointCloudGeometry.setAttribute('position', posAttr);
          this.rawPointCloudGeometry.setAttribute('color', colAttr);
        }

        const positions = posAttr.array;
        const colors = colAttr.array;
        const tempColor = new THREE.Color();

        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        let minZ = Infinity, maxZ = -Infinity;

        for (let i = 0; i < numPoints; i++) {
          const x = flatPoints[i * 3];
          const y = flatPoints[i * 3 + 1];
          const z = flatPoints[i * 3 + 2];

          positions[i * 3]     = x;
          positions[i * 3 + 1] = z;  // Three.js Y-axis is vertical elevation
          positions[i * 3 + 2] = -y; // Three.js -Z is vehicle forward

          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (z < minY) minY = z;
          if (z > maxY) maxY = z;
          if (-y < minZ) minZ = -y;
          if (-y > maxZ) maxZ = -y;

          // Natural elevation spectral ramp for raw sensor returns (Z: -2.0m to +1.5m)
          // Low asphalt points render cool blue/cyan; elevated curbs, cars, trees render amber/red
          const normZ = Math.min(Math.max((z - (-2.0)) / 3.5, 0.0), 1.0);
          tempColor.setHSL(0.66 * (1.0 - normZ), 0.9, 0.5);

          colors[i * 3]     = tempColor.r;
          colors[i * 3 + 1] = tempColor.g;
          colors[i * 3 + 2] = tempColor.b;
        }

        posAttr.needsUpdate = true;
        colAttr.needsUpdate = true;

        this.rawPointCloudGeometry.computeBoundingBox();
        this.rawPointCloudGeometry.computeBoundingSphere();

        if (numPoints > 0 && minX < maxX) {
          if (!this.sweepBoundingBox) this.sweepBoundingBox = new THREE.Box3();
          this.sweepBoundingBox.min.set(minX, minY, minZ);
          this.sweepBoundingBox.max.set(maxX, maxY, maxZ);

          if (!this.sweepCenter) this.sweepCenter = new THREE.Vector3();
          this.sweepBoundingBox.getCenter(this.sweepCenter);

          if (!this.sweepSize) this.sweepSize = new THREE.Vector3();
          this.sweepBoundingBox.getSize(this.sweepSize);

          if (!this.hasFramedSweep && !this.userInteracted && this.options.mode !== MODE_PERCEPTION) {
            this.hasFramedSweep = true;
            this.resetCamera();
          }
        }
      }

      this.pointMaterial.size = 0.24 * (o.size || 1);
    } else {
      // --- MODE 2 (2.5D Adaptive Mapping) & TERRAIN MESH MODES ---
      this.rawPointCloud.visible = false;
      if (this.tiles) this.tiles.visible = true;

      const totalNeeded = cells.length + objects.length * 20;
      if (totalNeeded > this.capacity) {
        this.ensureCapacity(totalNeeded);
      }

      // Pre-calculate dynamic object influence geometry once per frame
      const activeObjects = [];
      if (influence && objects.length > 0) {
        for (let j = 0; j < objects.length; j++) {
          const obj = objects[j];
          const f = footprintOf(obj);
          const heading = obj.heading || 0;
          const c = Math.cos(heading);
          const sn = Math.sin(heading);
          const speed = obj.speed ?? Math.hypot(obj.vx || 0, obj.vy || 0);
          activeObjects.push({
            x: obj.x,
            y: obj.y,
            c,
            sn,
            f,
            kind: classOf(obj.class),
            speed,
            len: speed * 2,
            ux: speed > 0.001 ? (obj.vx || 0) / speed : 0,
            uy: speed > 0.001 ? (obj.vy || 0) / speed : 0,
          });
        }
      }

      // Ensure InstancedMesh renders at full opacity without alpha drops, respecting static layer filter
      this.tileMaterial.transparent = false;
      this.tileMaterial.opacity = 1.0;
      this.tiles.visible = !isPoints && showStatic;

      this.instancedTerrainMesh = this.tiles;
      this.maxTerrainCells = this.capacity;
      this.currentMode = o.mode;
      const isSemanticMode = (this.currentMode === 'semantic' || this.currentMode === 3 || isSemantic || o.mode === MODE_SEMANTIC);
      const isAdaptiveMode = (this.currentMode === 'adaptive' || this.currentMode === 2 || isAdaptive || o.mode === MODE_ADAPTIVE);

      let count = 0;
      let minX = Infinity, maxX = -Infinity;
      let minY = Infinity, maxY = -Infinity;
      let minZ = Infinity, maxZ = -Infinity;
      const minHeight = this.minHeight !== undefined ? this.minHeight : o.minHeight;
      const maxHeight = this.maxHeight !== undefined ? this.maxHeight : o.maxHeight;

      for (let i = 0; i < cells.length && count < this.maxTerrainCells; i++) {
        const cell = cells[i];

        const zMean = Number(cell.z ?? cell.z_mean ?? -1.68);
        const zMin = Number(cell.z_min ?? zMean);
        const zMax = Number(cell.z_max ?? zMean);
        const deltaZ = Number(cell.delta_z ?? (zMax - zMin) ?? 0.0);

        const isVisible = zMin >= minHeight && zMin <= maxHeight;
        if (!isVisible) continue;

        // Ground plane asphalt filter: flat pavement within drivable elevation limits
        const isGroundBand = (
          zMean >= -2.20 &&
          zMean <= -1.25 &&
          deltaZ <= 0.18 &&
          zMax <= -1.15
        );

        let effectiveCost;
        if (isGroundBand) {
          effectiveCost = 0; // Emerald Green: Safe Traversable Corridor
        } else {
          // Obstacle: prioritize semantic cost or raw cost
          effectiveCost = isSemanticMode
            ? Number(cell.semantic_cost ?? cell.cost ?? 0)
            : Number(cell.cost ?? cell.semantic_cost ?? 0);
        }

        // Assign color using defense colormap
        const colorHex = getCostColor(effectiveCost, zMean);
        this.instancedTerrainMesh.setColorAt(count, new THREE.Color(colorHex));

        // Determine low-relief step height and layout
        const isHazard = (effectiveCost >= 181);
        const isCaution = (effectiveCost > 50 && effectiveCost <= 180);

        const baseSize = Number(
          cell.size || (effectiveCost <= 50 ? 1.0 : (effectiveCost <= 180 ? 0.75 : 0.50))
        );

        let tileHeight;
        let xyScale;

        if (isHazard) {
          tileHeight = 0.20;           // 20 cm obstacle step
          xyScale = baseSize * 0.98;
        } else if (isCaution) {
          tileHeight = 0.10;           // 10 cm caution step
          xyScale = baseSize * 0.98;
        } else {
          tileHeight = 0.04;           // 4 cm flat pavement tile
          xyScale = baseSize * 1.02;   // Overlap micro-seams into unbroken road
        }

        const baseFloor = (zMin < -1.0) ? zMin : -1.68;
        const centerY = baseFloor + (tileHeight * 0.5);

        this.matrix.makeScale(xyScale, tileHeight, xyScale);
        this.matrix.setPosition(Number(cell.x), centerY, -Number(cell.y));
        this.instancedTerrainMesh.setMatrixAt(count, this.matrix);

        const halfX = xyScale * 0.5;
        const halfY = tileHeight * 0.5;
        const px = Number(cell.x);
        const py = centerY;
        const pz = -Number(cell.y);

        if (px - halfX < minX) minX = px - halfX;
        if (px + halfX > maxX) maxX = px + halfX;
        if (py - halfY < minY) minY = py - halfY;
        if (py + halfY > maxY) maxY = py + halfY;
        if (pz - halfX < minZ) minZ = pz - halfX;
        if (pz + halfX > maxZ) maxZ = pz + halfX;

        count++;
      }

      this.instancedTerrainMesh.count = count;
      if (this.instancedTerrainMesh.instanceMatrix) {
        this.instancedTerrainMesh.instanceMatrix.needsUpdate = true;
      }
      if (this.instancedTerrainMesh.instanceColor) {
        this.instancedTerrainMesh.instanceColor.needsUpdate = true;
      }

      // Update full 2.5D Quadtree grid mesh bounding box and camera target
      if (count > 0 && minX < maxX) {
        if (!this.tiles.boundingBox) this.tiles.boundingBox = new THREE.Box3();
        this.tiles.boundingBox.min.set(minX, minY, minZ);
        this.tiles.boundingBox.max.set(maxX, maxY, maxZ);

        if (!this.tiles.boundingSphere) this.tiles.boundingSphere = new THREE.Sphere();
        this.tiles.boundingBox.getBoundingSphere(this.tiles.boundingSphere);

        this.sweepBoundingBox = this.tiles.boundingBox.clone();
        if (!this.sweepCenter) this.sweepCenter = new THREE.Vector3();
        this.sweepBoundingBox.getCenter(this.sweepCenter);

        if (!this.sweepSize) this.sweepSize = new THREE.Vector3();
        this.sweepBoundingBox.getSize(this.sweepSize);

        if (!this.hasFramedSweep && !this.userInteracted && this.options.mode !== MODE_PERCEPTION) {
          this.hasFramedSweep = true;
          this.resetCamera();
        }
      }
    }

    const present = new Set();
    for (let i = 0; i < objects.length; i++) {
      const object = objects[i];
      present.add(object.id);
      let a = this.actors.get(object.id);
      if (a && a.object.class !== object.class) {
        this.scene.remove(a.root, a.hazard);
        this.actors.delete(object.id);
        a = null;
      }
      a ||= this.addActor(object);
      a.object = object;
      // Look up actual cell elevation at object's (x, y) location
      let groundZ = -1.68;
      let minDistanceSq = Infinity;
      for (let cIdx = 0; cIdx < cells.length; cIdx++) {
        const c = cells[cIdx];
        const half = (c.size || 1.0) * 0.5;
        const dx = Math.abs(object.x - c.x);
        const dy = Math.abs(object.y - c.y);
        if (dx <= half && dy <= half) {
          groundZ = c.z_min !== undefined ? c.z_min : -1.68;
          minDistanceSq = 0;
          break;
        }
        const dSq = dx * dx + dy * dy;
        if (dSq < minDistanceSq) {
          minDistanceSq = dSq;
          groundZ = c.z_min !== undefined ? c.z_min : -1.68;
        }
      }

      const elevZ = (typeof object.z === "number" && Number.isFinite(object.z) && object.z < -0.5)
        ? object.z
        : (minDistanceSq < 16 ? groundZ : (typeof object.z === "number" && Number.isFinite(object.z) ? object.z : groundZ));

      const actorY = a.vehicle
        ? (elevZ + 0.04 - 0.11)
        : a.human
          ? (elevZ + 0.04 - 0.12)
          : (elevZ + 0.04);

      a.root.position.set(object.x, actorY, -object.y);
      const vx = object.vx;
      const vy = object.vy;

      let heading = 0;
      if (vx !== undefined && vy !== undefined && (Math.hypot(vx, vy) > 0.1)) {
        heading = Math.atan2(-vy || 0, vx);
      } else if (object.heading !== undefined) {
        heading = Number(object.heading);
      }

      // Align avatar heading down +X corridor
      const targetRotY = heading + (Math.PI / 2);
      a.root.rotation.y = targetRotY;

      if (a.hazardCone) {
        a.hazardCone.rotation.y = targetRotY;
      }
      if (a.hazard) {
        a.hazard.rotation.y = targetRotY;
      }

      const bodiesVisible = (isSemantic || isPerception) && showDynamic;
      a.root.visible = bodiesVisible;
      const speed = Math.hypot(Number(vx || 0), Number(vy || 0));
      a.hazard.visible = bodiesVisible && speed >= 0.05 && showDynamic;
      a.hazard.position.set(object.x, elevZ + 0.042, -object.y);
      a.hazard.scale.set(
        (a.vehicle ? 1.4 : 0.6) + speed * 0.15,
        1.0,
        Math.max(1.0, speed * 2.0),
      );
    }
    for (const [id, a] of this.actors) {
      if (!present.has(id)) {
        this.scene.remove(a.root, a.hazard);
        this.actors.delete(id);
      }
    }
    const egoPose = frame.ego_pose || frame.pose || frame.ego || frame.vehicle;
    if (egoPose && typeof egoPose === "object") {
      this.updatePose(egoPose);
    } else if (this.vehicleMesh) {
      const heading = Number(frame.ego_yaw ?? frame.telemetry?.ego_yaw ?? 0);
      this.vehicleMesh.rotation.y = heading + (Math.PI / 2);
      if (this.sector) {
        this.sector.rotation.z = 0;
      }
    }
    this.updatePerception(frame, isPerception);
  }

  /** Shared-state helpers: every mode reads the SAME telemetry frame.
   * Nothing here invents a second simulation. */
  dynamicInfluence(objects, cell) {
    const half = Math.max(cell.size, 0.6) / 2;
    for (const object of objects) {
      const f = footprintOf(object);
      const c = Math.cos(object.heading),
        sn = Math.sin(object.heading);
      const dx = cell.x - object.x,
        dy = cell.y - object.y;
      const along = dx * c + dy * sn,
        across = -dx * sn + dy * c;
      if (
        Math.abs(along) <= f.l / 2 + half &&
        Math.abs(across) <= f.w / 2 + half
      )
        return { kind: classOf(object.class), height: f.h, projected: false };
    }
    for (const object of objects) {
      const speed = Math.hypot(object.vx, object.vy);
      if (speed < 0.2) continue;
      const f = footprintOf(object);
      const len = speed * 2,
        ux = object.vx / speed,
        uy = object.vy / speed;
      const dx = cell.x - object.x,
        dy = cell.y - object.y;
      const along = dx * ux + dy * uy;
      if (along < 0 || along > len) continue;
      if (Math.abs(-dx * uy + dy * ux) <= f.w / 2 + half)
        return { kind: classOf(object.class), height: f.h, projected: true };
    }
    return null;
  }
  /** Raw point mode: tracked objects appear only as moving point returns. */
  writeObjectPoints(frame, pos, col, start) {
    if (this.options.filterMode === "static") return start;
    let v = start;
    const limit = pos.count;
    for (const object of frame.dynamic_objects) {
      const f = footprintOf(object);
      const kind = classOf(object.class);
      const speed = Math.hypot(object.vx, object.vy);
      const base = semanticColor[kind];
      const c = Math.cos(object.heading),
        sn = Math.sin(object.heading);
      const salt = String(object.id)
        .split("")
        .reduce((a, ch) => a + ch.charCodeAt(0), 0);
      const density = Math.round(kind === "vehicle" ? 34 : 20);
      for (let i = 0; i < density && v < limit; i++, v++) {
        const lx = jitter(i, salt) * f.w,
          ly = jitter(i + 91, salt) * f.l,
          lz = (jitter(i + 173, salt) + 0.5) * f.h;
        const x = object.x + ly * c - lx * sn,
          y = object.y + ly * sn + lx * c;
        pos.setXYZ(v, x, lz, -y);
        this.color
          .copy(base)
          .multiplyScalar(0.75 + Math.min(0.45, speed * 0.12));
        col.setXYZ(v, this.color.r, this.color.g, this.color.b);
      }
    }
    return v;
  }

  /** Vehicle-centric perception layer. Built once, reused every frame, and
   * driven entirely by the same telemetry frame the other modes render. */
  buildPerception() {
    this.perception = new THREE.Group();
    this.perception.visible = false;
    this.scene.add(this.perception);
    this.detections = new Map();
    this.labels = new Map();
    const line = (color, opacity = 1) =>
      this.track(
        new THREE.LineBasicMaterial({ color, transparent: true, opacity }),
      );
    this.detectionMat = {
      vehicle: line("#F85149"),
      pedestrian: line("#D29922"),
      animal: line("#D29922"),
      obstacle: line("#F85149"),
    };
    this.trailMat = line("#D29922", 0.75);
    this.boxEdges = this.track(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
    );
    // Static environment, instanced by category for a light draw-call budget.
    const instanced = (geometry, material, capacity) => {
      const mesh = new THREE.InstancedMesh(
        this.track(geometry),
        this.track(material),
        capacity,
      );
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      mesh.frustumCulled = false;
      mesh.count = 0;
      this.perception.add(mesh);
      return mesh;
    };
    const standard = (color, opts = {}) =>
      new THREE.MeshStandardMaterial({ color, roughness: 0.8, ...opts });
    this.structures = instanced(
      new THREE.BoxGeometry(1, 1, 1),
      standard("#3d444d"),
      320,
    );
    const canopy = new THREE.ConeGeometry(0.55, 1, 6);
    canopy.translate(0, 0.5, 0);
    this.vegetation = instanced(canopy, standard("#2c7a4b"), 320);
    const dip = new THREE.CircleGeometry(0.5, 10);
    dip.rotateX(-Math.PI / 2);
    this.potholes = instanced(
      dip,
      new THREE.MeshBasicMaterial({ color: "#6E2B2B" }),
      160,
    );
    const post = new THREE.BoxGeometry(0.12, 0.55, 0.12);
    post.translate(0, 0.27, 0);
    this.boundaries = instanced(post, standard("#C9A227"), 320);
    this.staticObstacles = instanced(
      new THREE.BoxGeometry(0.8, 0.8, 0.8),
      standard("#8f4030"),
      240,
    );
    // Ego platform and its forward perception region.
    const ego = new THREE.Group();
    ego.add(
      this.mesh(
        this.box,
        this.track(standard("#E6F1FF")),
        [2, 0.7, 4.4],
        [0, 0.7, 0],
      ),
      this.mesh(
        this.box,
        this.track(standard("#21262d", { metalness: 0.1 })),
        [1.7, 0.62, 2],
        [0, 1.3, -0.2],
      ),
    );
    const sector = new THREE.CircleGeometry(26, 40, -0.7, 1.4);
    sector.rotateX(-Math.PI / 2);
    this.sector = new THREE.Mesh(
      this.track(sector),
      this.track(
        new THREE.MeshBasicMaterial({
          color: "#238636",
          transparent: true,
          opacity: 0.08,
          depthWrite: false,
          side: THREE.DoubleSide,
        }),
      ),
    );
    this.sector.position.y = 0.02;
    this.sector.rotation.y = 0;
    const ring = new THREE.RingGeometry(
      25.7,
      26,
      64,
      1,
      -0.7,
      1.4,
    );
    ring.rotateX(-Math.PI / 2);
    const rim = new THREE.Mesh(
      this.track(ring),
      this.track(
        new THREE.MeshBasicMaterial({
          color: "#238636",
          transparent: true,
          opacity: 0.5,
        }),
      ),
    );
    rim.position.y = 0.03;
    rim.rotation.y = 0;
    this.rim = rim;
    this.perception.add(ego, this.sector, rim);
    this.egoGroup = ego;
    this.vehicleGroup = ego;
    this.vehicleMesh = ego;
    const egoYaw = 0;
    this.vehicleMesh.rotation.y = egoYaw + Math.PI / 2;
  }
  label(text, color) {
    let sprite = this.labels.get(text);
    if (sprite) return sprite;
    const canvas = document.createElement("canvas");
    canvas.width = 256;
    canvas.height = 64;
    const ctx = canvas.getContext("2d");
    ctx.font = "700 34px Inter, Segoe UI, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "rgba(5, 8, 17, 0.72)";
    const width = ctx.measureText(text).width + 26;
    ctx.fillRect(128 - width / 2, 8, width, 48);
    ctx.fillStyle = color;
    ctx.fillText(text, 128, 33);
    const texture = this.track(new THREE.CanvasTexture(canvas));
    texture.colorSpace = THREE.SRGBColorSpace;
    const material = this.track(
      new THREE.SpriteMaterial({
        map: texture,
        transparent: true,
        depthTest: false,
      }),
    );
    sprite = new THREE.Sprite(material);
    sprite.scale.set(6, 1.5, 1);
    this.labels.set(text, sprite);
    return sprite;
  }
  detectionFor(object) {
    let d = this.detections.get(object.id);
    const kind = classOf(object.class);
    if (d && d.kind !== kind) {
      this.perception.remove(d.group);
      this.detections.delete(object.id);
      d = null;
    }
    if (d) return d;
    const group = new THREE.Group();
    const outline = new THREE.LineSegments(
      this.boxEdges,
      this.detectionMat[kind],
    );
    const geometry = this.track(new THREE.BufferGeometry());
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(6), 3).setUsage(
        THREE.DynamicDrawUsage,
      ),
    );
    const trail = new THREE.Line(geometry, this.trailMat);
    const tag = this.label(
      `${object.id} · ${kind}`,
      `#${semanticColor[kind].getHexString()}`,
    );
    group.add(outline, trail, tag);
    this.perception.add(group);
    d = { group, outline, trail, tag, kind };
    this.detections.set(object.id, d);
    return d;
  }
  /** Derives static environment categories from the received cell array.
   * Cells are the only source; nothing is randomly generated per mode. */
  updatePerception(frame, active) {
    this.perception.visible = active;
    if (!active) return;
    const showStatic = this.options.filterMode !== "dynamic";
    const showDynamic = this.options.filterMode !== "static";
    if (this.structures) this.structures.visible = showStatic;
    if (this.vegetation) this.vegetation.visible = showStatic;
    if (this.potholes) this.potholes.visible = showStatic;
    if (this.boundaries) this.boundaries.visible = showStatic;
    if (this.staticObstacles) this.staticObstacles.visible = showStatic;
    if (this.egoGroup) this.egoGroup.visible = showDynamic;
    if (this.sector) this.sector.visible = showDynamic;
    if (this.rim) this.rim.visible = showDynamic;

    const counts = {
      structures: 0,
      vegetation: 0,
      potholes: 0,
      boundaries: 0,
      staticObstacles: 0,
    };
    const place = (mesh, key, x, y, scale, base) => {
      if (counts[key] >= mesh.geometry.userData.cap) return;
      this.matrix.makeScale(scale[0], scale[1], scale[2]);
      this.matrix.setPosition(x, base, -y);
      mesh.setMatrixAt(counts[key]++, this.matrix);
    };
    for (const m of [
      "structures",
      "vegetation",
      "potholes",
      "boundaries",
      "staticObstacles",
    ])
      this[m].geometry.userData.cap = this[m].instanceMatrix.count;
    let index = 0;
    for (const cell of frame.cells) {
      const height = cell.z_max - cell.z_min;
      const band = costBand(cell.cost);
      index++;
      if (band === 2 && height >= 1.4)
        place(
          this.structures,
          "structures",
          cell.x,
          cell.y,
          [cell.size * 0.92, height, cell.size * 0.92],
          cell.z_min + height / 2,
        );
      else if (band === 2 && height >= 0.35)
        place(
          this.staticObstacles,
          "staticObstacles",
          cell.x,
          cell.y,
          [cell.size * 0.6, height * 1.2, cell.size * 0.6],
          cell.z_min + height / 2,
        );
      else if (band === 1 && height >= 0.8)
        place(
          this.vegetation,
          "vegetation",
          cell.x,
          cell.y,
          [cell.size, Math.min(3.2, height * 1.4), cell.size],
          cell.z_min,
        );
      else if (band === 1 && index % 3 === 0)
        place(
          this.boundaries,
          "boundaries",
          cell.x,
          cell.y,
          [1, 1, 1],
          cell.z_min,
        );
      if (band === 2 && height < 0.35)
        place(
          this.potholes,
          "potholes",
          cell.x,
          cell.y,
          [cell.size * 0.8, 1, cell.size * 0.8],
          cell.z_min + 0.02,
        );
    }
    for (const key of Object.keys(counts)) {
      this[key].count = counts[key];
      this[key].instanceMatrix.needsUpdate = true;
    }
    const present = new Set();
    const ranked = [...frame.dynamic_objects]
      .map((o) => ({ o, d2: o.x * o.x + o.y * o.y }))
      .sort((a, b) => a.d2 - b.d2);
    ranked.forEach(({ o: object }, i) => {
      present.add(object.id);
      const d = this.detectionFor(object);
      const f = footprintOf(object);
      const vx = Number(object.vx || 0);
      const vy = Number(object.vy || 0);
      const speed = Math.hypot(vx, vy);
      let heading = Number(object.heading ?? 0);
      if (speed >= 0.1) {
        heading = Math.atan2(-vy || 0, vx);
      }
      d.outline.scale.set(f.w + 0.3, f.h + 0.2, f.l + 0.3);
      d.outline.position.set(0, (f.h + 0.2) / 2, 0);
      d.outline.rotation.y = heading + Math.PI / 2;

      let groundZ = -1.68;
      if (frame.cells) {
        for (let cIdx = 0; cIdx < frame.cells.length; cIdx++) {
          const c = frame.cells[cIdx];
          const half = (c.size || 1.0) * 0.5;
          if (Math.abs(object.x - c.x) <= half && Math.abs(object.y - c.y) <= half) {
            groundZ = c.z_min !== undefined ? c.z_min : -1.68;
            break;
          }
        }
      }
      const elevZ = (typeof object.z === "number" && Number.isFinite(object.z) && object.z < -0.5)
        ? object.z
        : groundZ;
      d.group.position.set(object.x, elevZ + 0.04, -object.y);
      d.group.visible = showDynamic;
      const trail = d.trail.geometry.attributes.position;
      trail.setXYZ(0, 0, 0.12, 0);
      trail.setXYZ(1, object.vx * 2.2, 0.12, -object.vy * 2.2);
      trail.needsUpdate = true;
      d.trail.visible = speed >= 0.2 && showDynamic;
      // Keep the scene readable: identify only the nearest few tracks.
      d.tag.visible = i < 6 && showDynamic;
      d.tag.position.set(0, f.h + 1.5, 0);
    });
    for (const [id, d] of this.detections)
      if (!present.has(id)) {
        this.perception.remove(d.group);
        this.detections.delete(id);
      }
  }

  render(time) {
    if (this.disposed) return;
    if (this.dirty || (this.nextFrame && this.nextFrame !== this.frame)) {
      if (this.nextFrame) this.frame = this.nextFrame;
      this.updateFrame();
      this.ground.visible = this.options.ground && (this.options.filterMode !== "dynamic");
      this.grid.visible = this.options.grid && (this.options.filterMode !== "dynamic");
      this.axes.visible = this.options.axes;
      this.dirty = false;
    }
    const speedMultiplier = this.options.playbackSpeed || 1.0;
    for (const a of this.actors.values()) {
      if (!a.human || this.reduced.matches || this.options.motion === false)
        continue;
      const s = a.object.speed,
        idle = s < 0.2,
        run = s > 1.8,
        rate = idle ? 1.8 : Math.min(16, 3 + s * 3),
        phase = time * rate * speedMultiplier;
      a.body.position.y = idle
        ? Math.sin(phase) * 0.015
        : Math.abs(Math.sin(phase)) * (run ? 0.1 : 0.035);
      a.body.rotation.z = idle ? 0 : Math.sin(phase) * 0.025;
      a.limbs.forEach((limb, i) => {
        limb.rotation.x = idle
          ? 0
          : Math.sin(phase + (i % 2 ? Math.PI : 0)) * (run ? 0.85 : 0.4);
      });
    }
    if (this.tween) {
      const t = this.tween;
      t.elapsed += Math.min(0.05, Math.max(0, time - (t.last ?? time)));
      t.last = time;
      const k = Math.min(1, t.elapsed / t.seconds);
      const e = k < 0.5 ? 2 * k * k : 1 - (-2 * k + 2) ** 2 / 2;
      this.camera.position.lerpVectors(t.from, t.to, e);
      this.controls.target.lerpVectors(t.targetFrom, t.targetTo, e);
      if (k >= 1) this.tween = null;
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
  dispose() {
    this.disposed = true;
    this.renderer.setAnimationLoop(null);
    this.observer.disconnect();
    this.controls.dispose();
    this.renderer.domElement.removeEventListener(
      "webglcontextlost",
      this.onLost,
    );
    this.tiles?.dispose();
    this.points?.geometry.dispose();
    for (const h of [this.grid, this.axes]) {
      h.geometry.dispose();
      if (Array.isArray(h.material)) h.material.forEach((m) => m.dispose());
      else h.material.dispose();
    }
    this.resources.forEach((r) => {
      if (typeof r?.dispose === "function") r.dispose();
    });
    this.actors.clear();
    this.detections.clear();
    this.labels.clear();
    this.scene.clear();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
