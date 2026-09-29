import * as THREE from "three";
/** Abstract cartographic illustration, not sensor data. All geometry stays local. */
export function mountWelcomeScene(element) {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 200);
  camera.position.set(28, 24, 32);
  camera.lookAt(0, 0, 0);
  const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
  element.appendChild(renderer.domElement);
  const terrain = new THREE.Group();
  scene.add(terrain);
  const vertices = [],
    points = [];
  const h = (x, z) =>
    2.3 * Math.exp(-((x + 8) ** 2 + (z + 4) ** 2) / 32) +
    3.8 * Math.exp(-((x - 7) ** 2 + (z + 6) ** 2) / 20) +
    0.22 * Math.sin(x * 0.6) * Math.cos(z * 0.4);
  for (let z = -14; z < 14; z++)
    for (let x = -16; x < 16; x++) {
      const y = h(x, z);
      points.push(x, y, z);
      vertices.push(
        x,
        y,
        z,
        x + 1,
        h(x + 1, z),
        z,
        x,
        y,
        z,
        x,
        h(x, z + 1),
        z + 1,
      );
    }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
  const mat = new THREE.LineBasicMaterial({
    color: 0x238636, // DRDO Emerald Green: Safe Traversable Ground
    transparent: true,
    opacity: 0.35,
  });
  terrain.add(new THREE.LineSegments(geo, mat));
  const pgeo = new THREE.BufferGeometry();
  pgeo.setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
  const pmat = new THREE.PointsMaterial({ color: 0xd0d7de, size: 0.075 });
  terrain.add(new THREE.Points(pgeo, pmat));
  const rings = [];
  for (const radius of [4, 7, 10]) {
    const g = new THREE.RingGeometry(radius, radius + 0.035, 96);
    g.rotateX(-Math.PI / 2);
    const m = new THREE.MeshBasicMaterial({
      color: 0xd29922, // DRDO Solar Amber Yellow: Caution / Radar Ring
      transparent: true,
      opacity: 0.22,
      side: THREE.DoubleSide,
    });
    const ring = new THREE.Mesh(g, m);
    ring.position.y = 0.6;
    terrain.add(ring);
    rings.push(ring);
  }
  const pillar = new THREE.Mesh(
    new THREE.CylinderGeometry(0.08, 0.08, 5, 8),
    new THREE.MeshBasicMaterial({ color: 0x8b949e }),
  );
  pillar.position.y = 3;
  terrain.add(pillar);
  const beacon = new THREE.Mesh(
    new THREE.OctahedronGeometry(0.7),
    new THREE.MeshBasicMaterial({ color: 0xd0d7de, wireframe: true }),
  );
  beacon.position.y = 5.8;
  terrain.add(beacon);
  const resize = () => {
    const w = element.clientWidth,
      h = element.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  const observer = new ResizeObserver(resize);
  observer.observe(element);
  resize();
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  renderer.setAnimationLoop((t) => {
    const time = t / 1000;
    if (!reduced.matches) {
      terrain.rotation.y = Math.sin(time * 0.12) * 0.16;
      beacon.position.y = 5.8 + Math.sin(time * 0.9) * 0.25;
      beacon.rotation.y = time * 0.2;
      rings.forEach(
        (r, i) =>
          (r.material.opacity =
            0.14 + 0.12 * (1 + Math.sin(time * 0.6 - i)) * 0.5),
      );
    }
    renderer.render(scene, camera);
  });
  return () => {
    renderer.setAnimationLoop(null);
    observer.disconnect();
    scene.traverse((o) => {
      o.geometry?.dispose();
      o.material?.dispose();
    });
    renderer.dispose();
    renderer.domElement.remove();
  };
}
