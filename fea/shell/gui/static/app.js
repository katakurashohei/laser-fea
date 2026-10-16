// Interactive frontend for the fea.shell solver. All physics happens on
// the server (fea/shell/gui/server.py -> fea.shell.solver.solve); this file
// only handles rendering and the click-drag force-picking interaction.

const VIRIDIS_STOPS = [
  [0.267, 0.004, 0.329], [0.282, 0.140, 0.457], [0.253, 0.265, 0.529],
  [0.206, 0.371, 0.553], [0.164, 0.471, 0.558], [0.128, 0.567, 0.550],
  [0.135, 0.659, 0.518], [0.267, 0.748, 0.440], [0.478, 0.821, 0.318],
  [0.741, 0.873, 0.149], [0.993, 0.906, 0.144],
];
function viridis(t) {
  t = Math.min(1, Math.max(0, t));
  const n = VIRIDIS_STOPS.length - 1;
  const idx = Math.min(n - 1, Math.floor(t * n));
  const frac = t * n - idx;
  const a = VIRIDIS_STOPS[idx], b = VIRIDIS_STOPS[idx + 1];
  return [a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac, a[2] + (b[2] - a[2]) * frac];
}

const PLATE_COLOR = [0.357, 0.561, 0.839];
const JOINT_COLOR = [0.878, 0.639, 0.243];
const FREE_NODE_COLOR = 0xdddddd;
const FIXED_NODE_COLOR = 0x2266dd;
const FORCE_NODE_COLOR = 0xff5533;

let scene, camera, renderer, controls;
let mesh, geometry, nodeIdxPerVertex, positions, colors;
let nodes = [], nQuads = 0, nPlate = 0, elementType = [];
let nodeMarkers, nodePositionsVec3 = [];
let edgeLines, nodeSphereGeom;
let arrowHelper;
const objLoader = new THREE.OBJLoader();
let dragging = false, selectedNode = null, selectedMarker = null;
const dragPlane = new THREE.Plane();
const raycaster = new THREE.Raycaster();
const mouseNDC = new THREE.Vector2();

let fixMode = false;
let fixedNodes = new Set();
let autoBottomIds = [];
let fixClicking = false;

let colorBy = "displacement";
let deformScale = 20;
let forceScale = 4000;
let lastResult = null; // {displacements, ...} from the server, for re-coloring on control change

let plateE_GPa = 9;       // must match server.py's PLYWOOD_E default (9e9 Pa)
let jointFraction = 0.4;  // must match server.py's JOINT_STIFFNESS_FRACTION default
let materialTimer = null;

init();

async function init() {
  const holder = document.getElementById("canvas-holder");

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x12151a);

  camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 100);
  camera.up.set(0, 0, 1);
  camera.position.set(1.9, -2.1, 1.5);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  holder.appendChild(renderer.domElement);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.update();

  const data = await (await fetch("/api/mesh")).json();
  buildScene(data);

  arrowHelper = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), 0.1, 0xff5533, 0.03, 0.02);
  arrowHelper.visible = false;
  scene.add(arrowHelper);

  window.addEventListener("resize", onResize);
  renderer.domElement.addEventListener("pointerdown", onPointerDown, true);
  renderer.domElement.addEventListener("pointermove", onPointerMove, true);
  renderer.domElement.addEventListener("pointerup", onPointerUp, true);

  document.getElementById("colorBy").addEventListener("change", (e) => {
    colorBy = e.target.value;
    recolor();
  });
  document.getElementById("scale").addEventListener("input", (e) => {
    deformScale = parseFloat(e.target.value);
    document.getElementById("scaleVal").textContent = deformScale;
    if (lastResult) applyResult(lastResult);
  });
  document.getElementById("forceScale").addEventListener("input", (e) => {
    forceScale = parseFloat(e.target.value);
    document.getElementById("forceScaleVal").textContent = forceScale;
  });
  document.getElementById("plateStiffness").addEventListener("input", (e) => {
    plateE_GPa = parseFloat(e.target.value);
    document.getElementById("plateStiffnessVal").textContent = plateE_GPa.toFixed(1);
    scheduleMaterialUpdate();
  });
  document.getElementById("jointStiffness").addEventListener("input", (e) => {
    jointFraction = parseFloat(e.target.value) / 100;
    document.getElementById("jointStiffnessVal").textContent = e.target.value;
    scheduleMaterialUpdate();
  });
  document.getElementById("meshRes").addEventListener("input", (e) => {
    document.getElementById("meshResVal").textContent = e.target.value;
  });
  // Remesh only on release — each step rebuilds the mesh on the server.
  document.getElementById("meshRes").addEventListener("change", (e) => {
    changeResolution(parseInt(e.target.value, 10));
  });
  document.getElementById("resetBtn").addEventListener("click", resetLoad);
  document.getElementById("resetCubeBtn").addEventListener("click", resetToCube);

  document.getElementById("fixModeBtn").addEventListener("click", toggleFixMode);
  document.getElementById("clearFixedBtn").addEventListener("click", () => {
    fixedNodes.clear();
    colorNodeMarkers();
  });
  document.getElementById("autoFixedBtn").addEventListener("click", () => {
    fixedNodes = new Set(autoBottomIds);
    colorNodeMarkers();
  });

  setupDragAndDrop();
  animate();
}

function disposeScene() {
  if (mesh) { scene.remove(mesh); mesh.geometry.dispose(); mesh.material.dispose(); }
  if (edgeLines) { scene.remove(edgeLines); edgeLines.geometry.dispose(); edgeLines.material.dispose(); }
  if (nodeMarkers) {
    scene.remove(nodeMarkers);
    for (const s of nodeMarkers.children) s.material.dispose();
  }
  if (nodeSphereGeom) nodeSphereGeom.dispose();
  nodePositionsVec3 = [];
  selectedMarker = null;
  selectedNode = null;
}

function buildScene(data) {
  disposeScene();

  nodes = data.nodes;
  syncResolutionSlider(data);
  autoBottomIds = data.bottom_ids || [];
  fixedNodes = new Set(autoBottomIds);
  const quads = data.plate_quads.concat(data.joint_quads);
  nPlate = data.n_plate;
  nQuads = quads.length;
  elementType = quads.map((_, i) => (i < nPlate ? 0 : 1));

  nodeIdxPerVertex = new Int32Array(nQuads * 6);
  positions = new Float32Array(nQuads * 6 * 3);
  colors = new Float32Array(nQuads * 6 * 3);

  for (let q = 0; q < nQuads; q++) {
    const [a, b, c, d] = quads[q];
    const seq = [a, b, c, a, c, d];
    for (let k = 0; k < 6; k++) nodeIdxPerVertex[q * 6 + k] = seq[k];
  }

  geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const material = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
  mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);

  // Static undeformed wireframe for reference.
  const edgePositions = [];
  for (const [a, b, c, d] of quads) {
    for (const [p, q2] of [[a, b], [b, c], [c, d], [d, a]]) {
      edgePositions.push(...nodes[p], ...nodes[q2]);
    }
  }
  const edgeGeom = new THREE.BufferGeometry();
  edgeGeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(edgePositions), 3));
  const edgeMat = new THREE.LineBasicMaterial({ color: 0x444a55 });
  edgeLines = new THREE.LineSegments(edgeGeom, edgeMat);
  scene.add(edgeLines);

  // Pickable node markers, sized relative to the mesh's own extent so they
  // stay usable whether the box is 0.1 m or 10 m.
  const extent = boundingRadius(nodes) || 1;
  nodeMarkers = new THREE.Group();
  nodeSphereGeom = new THREE.SphereBufferGeometry(extent * 0.018, 8, 6);
  for (let i = 0; i < nodes.length; i++) {
    const mat = new THREE.MeshBasicMaterial({ color: FREE_NODE_COLOR });
    const s = new THREE.Mesh(nodeSphereGeom, mat);
    s.position.set(nodes[i][0], nodes[i][1], nodes[i][2]);
    s.userData.nodeIndex = i;
    nodeMarkers.add(s);
    nodePositionsVec3.push(s.position);
  }
  scene.add(nodeMarkers);
  colorNodeMarkers();

  lastResult = null;
  if (arrowHelper) arrowHelper.visible = false;
  colorByElementType();
  frameCamera();
}

function boundsOf(pts) {
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (const p of pts) {
    for (let k = 0; k < 3; k++) { min[k] = Math.min(min[k], p[k]); max[k] = Math.max(max[k], p[k]); }
  }
  const center = [0, 1, 2].map((k) => (min[k] + max[k]) / 2);
  const radius = Math.max(...[0, 1, 2].map((k) => max[k] - min[k])) / 2;
  return { center, radius };
}
function boundingRadius(pts) { return boundsOf(pts).radius; }

function frameCamera() {
  const { center, radius: r } = boundsOf(nodes);
  const radius = r || 1;

  controls.target.set(center[0], center[1], center[2]);
  const dist = radius * 3.2;
  camera.position.set(center[0] + dist * 0.66, center[1] - dist * 0.73, center[2] + dist * 0.52);
  camera.near = Math.max(radius / 100, 0.001);
  camera.far = Math.max(radius * 100, 100);
  camera.updateProjectionMatrix();
  controls.update();
}

function colorByElementType() {
  for (let q = 0; q < nQuads; q++) {
    const [r, g, b] = elementType[q] === 0 ? PLATE_COLOR : JOINT_COLOR;
    for (let k = 0; k < 6; k++) {
      const vi = q * 6 + k;
      const nodeId = nodeIdxPerVertex[vi];
      positions[vi * 3 + 0] = nodes[nodeId][0];
      positions[vi * 3 + 1] = nodes[nodeId][1];
      positions[vi * 3 + 2] = nodes[nodeId][2];
      colors[vi * 3 + 0] = r; colors[vi * 3 + 1] = g; colors[vi * 3 + 2] = b;
    }
  }
  geometry.attributes.position.needsUpdate = true;
  geometry.attributes.color.needsUpdate = true;
}

function applyResult(result) {
  const disp = result.displacements;
  const perElementValue = colorBy === "displacement"
    ? elementValuesFromNodal(result.displacement_magnitude)
    : result.curvature;
  const vmax = Math.max(...perElementValue, 1e-12);

  for (let q = 0; q < nQuads; q++) {
    const [r, g, b] = viridis(perElementValue[q] / vmax);
    for (let k = 0; k < 6; k++) {
      const vi = q * 6 + k;
      const nodeId = nodeIdxPerVertex[vi];
      const base = nodes[nodeId];
      const d = disp[nodeId];
      positions[vi * 3 + 0] = base[0] + d[0] * deformScale;
      positions[vi * 3 + 1] = base[1] + d[1] * deformScale;
      positions[vi * 3 + 2] = base[2] + d[2] * deformScale;
      colors[vi * 3 + 0] = r; colors[vi * 3 + 1] = g; colors[vi * 3 + 2] = b;
    }
  }
  geometry.attributes.position.needsUpdate = true;
  geometry.attributes.color.needsUpdate = true;
}

function elementValuesFromNodal(nodalValues) {
  // average the 4 corner node values for each quad element
  const quads = [];
  const out = new Array(nQuads);
  for (let q = 0; q < nQuads; q++) {
    const ids = [nodeIdxPerVertex[q * 6 + 0], nodeIdxPerVertex[q * 6 + 1],
                 nodeIdxPerVertex[q * 6 + 2], nodeIdxPerVertex[q * 6 + 5]];
    out[q] = ids.reduce((s, id) => s + nodalValues[id], 0) / 4;
  }
  return out;
}

function recolor() {
  if (lastResult) applyResult(lastResult);
}

function resetLoad() {
  lastResult = null;
  arrowHelper.visible = false;
  colorByElementType();
  document.getElementById("statsBody").innerHTML =
    '<span class="muted">Click a node and drag to apply a load.</span>';
  if (selectedMarker) { selectedMarker.material.color.set(baseNodeColor(selectedNode)); selectedMarker = null; }
}

function baseNodeColor(nodeIndex) {
  return fixedNodes.has(nodeIndex) ? FIXED_NODE_COLOR : FREE_NODE_COLOR;
}

function colorNodeMarkers() {
  if (!nodeMarkers) return;
  for (const s of nodeMarkers.children) {
    if (s === selectedMarker) continue; // keep the force-drag highlight if mid-selection
    s.material.color.set(baseNodeColor(s.userData.nodeIndex));
  }
}

function toggleFixMode() {
  fixMode = !fixMode;
  const btn = document.getElementById("fixModeBtn");
  btn.textContent = `Fix-support mode: ${fixMode ? "ON" : "OFF"}`;
  btn.classList.toggle("toggle-on", fixMode);
}

function setMouseNDC(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  mouseNDC.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouseNDC.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
}

function onPointerDown(event) {
  setMouseNDC(event);
  raycaster.setFromCamera(mouseNDC, camera);
  const hits = raycaster.intersectObjects(nodeMarkers.children);
  if (hits.length === 0) return; // let OrbitControls handle it normally

  event.stopImmediatePropagation();
  event.preventDefault();
  controls.enabled = false;

  const hit = hits[0].object;
  const nodeIndex = hit.userData.nodeIndex;

  if (fixMode) {
    fixClicking = true;
    if (fixedNodes.has(nodeIndex)) fixedNodes.delete(nodeIndex);
    else fixedNodes.add(nodeIndex);
    hit.material.color.set(baseNodeColor(nodeIndex));
    return;
  }

  selectedNode = nodeIndex;
  if (selectedMarker) selectedMarker.material.color.set(baseNodeColor(selectedMarker.userData.nodeIndex));
  selectedMarker = hit;
  selectedMarker.material.color.set(FORCE_NODE_COLOR);

  const camDir = new THREE.Vector3();
  camera.getWorldDirection(camDir);
  dragPlane.setFromNormalAndCoplanarPoint(camDir, hit.position);

  dragging = true;
  arrowHelper.position.copy(hit.position);
  arrowHelper.visible = true;
  arrowHelper.setLength(0.001, 0.03, 0.02);
}

function onPointerMove(event) {
  if (!dragging) return;
  event.stopImmediatePropagation();
  event.preventDefault();
  setMouseNDC(event);
  raycaster.setFromCamera(mouseNDC, camera);

  const hitPoint = new THREE.Vector3();
  if (!raycaster.ray.intersectPlane(dragPlane, hitPoint)) return;

  const nodePos = nodePositionsVec3[selectedNode];
  const delta = hitPoint.clone().sub(nodePos);
  const length = delta.length();
  if (length < 1e-5) {
    arrowHelper.setLength(0.001, 0.03, 0.02);
    return;
  }
  const dir = delta.clone().normalize();
  arrowHelper.setDirection(dir);
  arrowHelper.setLength(Math.min(length, 0.6), Math.min(0.05, length * 0.3), Math.min(0.03, length * 0.2));
  arrowHelper.userData.force = dir.clone().multiplyScalar(length * forceScale);
}

async function onPointerUp(event) {
  if (fixClicking) {
    fixClicking = false;
    controls.enabled = true;
    event.stopImmediatePropagation();
    event.preventDefault();
    return;
  }
  if (!dragging) return;
  event.stopImmediatePropagation();
  event.preventDefault();
  dragging = false;
  controls.enabled = true;

  const force = arrowHelper.userData.force;
  if (!force || force.length() < 1e-6) {
    arrowHelper.visible = false;
    return;
  }
  await solveCurrent(force);
}

async function solveCurrent(force) {
  if (fixedNodes.size === 0) {
    document.getElementById("statsBody").innerHTML =
      '<span class="muted">Pick at least one fixed node first (Fix-support mode).</span>';
    return;
  }

  document.getElementById("statsBody").innerHTML = '<span class="muted">Solving…</span>';
  try {
    const resp = await fetch("/api/solve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        node: selectedNode, force: [force.x, force.y, force.z],
        fixed_nodes: Array.from(fixedNodes),
      }),
    });
    const result = await resp.json();
    if (result.error) throw new Error(result.error);
    lastResult = result;
    applyResult(result);
    renderStats(force, result);
  } catch (err) {
    document.getElementById("statsBody").textContent = "Error: " + err.message;
  }
}

function renderStats(force, r) {
  const mag = force.length();
  const fmt = (x) => x.toFixed(3);
  const joinsMore = r.max_curvature_joint > r.max_curvature_plate;
  document.getElementById("statsBody").innerHTML = `
    Node <b>${selectedNode}</b> &nbsp; (${fixedNodes.size} nodes fixed)<br>
    Force: (${fmt(force.x)}, ${fmt(force.y)}, ${fmt(force.z)}) N &nbsp; |${mag.toFixed(1)} N|<br>
    Reaction sum: (${r.reaction_sum.map(fmt).join(", ")}) N<br>
    Max displacement: ${(r.max_displacement * 1000).toFixed(3)} mm<br>
    Peak curvature — plate: ${r.max_curvature_plate.toFixed(4)} 1/m<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;joint: ${r.max_curvature_joint.toFixed(4)} 1/m
    ${joinsMore ? " <b>&larr; bends more sharply</b>" : ""}
  `;
}

function scheduleMaterialUpdate() {
  clearTimeout(materialTimer);
  materialTimer = setTimeout(updateMaterials, 200); // debounce while dragging the slider
}

async function updateMaterials() {
  try {
    const resp = await fetch("/api/materials", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ plate_E: plateE_GPa * 1e9, joint_stiffness_fraction: jointFraction }),
    });
    const data = await resp.json();
    if (data.error) throw new Error(data.error);

    // If a force is already placed, re-solve immediately so the effect of
    // the new stiffness is visible right away instead of waiting for the
    // next drag.
    const force = arrowHelper && arrowHelper.userData.force;
    if (lastResult && force) await solveCurrent(force);
  } catch (err) {
    document.getElementById("statsBody").textContent = "Material update failed: " + err.message;
  }
}

function syncResolutionSlider(data) {
  if (data.resolution == null) return;
  const slider = document.getElementById("meshRes");
  slider.min = data.min_resolution;
  slider.max = data.max_resolution;
  slider.value = data.resolution;
  document.getElementById("meshResVal").textContent = data.resolution;
  document.getElementById("meshResKind").textContent = data.resolution_kind;
}

async function changeResolution(n) {
  const slider = document.getElementById("meshRes");
  slider.disabled = true;
  setImportMsg(`Remeshing at ${n} ${document.getElementById("meshResKind").textContent}…`, false);
  try {
    const resp = await fetch("/api/resolution", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ n }),
    });
    const data = await resp.json();
    if (data.error) {
      // Put the slider back where the mesh actually is.
      syncResolutionSlider(data);
      throw new Error(data.error);
    }
    buildScene(data);
    resetLoad();
    setImportMsg(`Remeshed: ${data.nodes.length} nodes, ${data.n_plate} plate + ` +
                 `${data.n_joint} joint elements.`, false);
  } catch (err) {
    setImportMsg("Remesh failed: " + err.message, true);
  } finally {
    slider.disabled = false;
  }
}

async function resetToCube() {
  setImportMsg("Resetting…", false);
  try {
    const resp = await fetch("/api/reset", { method: "POST" });
    const data = await resp.json();
    buildScene(data);
    resetLoad();
    setImportMsg("Reset to the default 1 m cube.", false);
  } catch (err) {
    setImportMsg("Reset failed: " + err.message, true);
  }
}

function setImportMsg(text, isError) {
  const el = document.getElementById("importMsg");
  el.textContent = text;
  el.style.color = isError ? "#ff8080" : "#e8e8e8";
  el.classList.remove("hidden");
  clearTimeout(setImportMsg._t);
  setImportMsg._t = setTimeout(() => el.classList.add("hidden"), 6000);
}

function setupDragAndDrop() {
  const overlay = document.getElementById("dropOverlay");
  let dragDepth = 0;

  window.addEventListener("dragenter", (e) => {
    e.preventDefault();
    dragDepth++;
    overlay.classList.add("active");
  });
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("dragleave", (e) => {
    e.preventDefault();
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) overlay.classList.remove("active");
  });
  window.addEventListener("drop", async (e) => {
    e.preventDefault();
    dragDepth = 0;
    overlay.classList.remove("active");

    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".obj")) {
      setImportMsg("Only .obj files are supported.", true);
      return;
    }
    await importObjFile(file);
  });
}

async function importObjFile(file) {
  setImportMsg(`Parsing ${file.name}…`, false);
  let text;
  try {
    text = await file.text();
  } catch (err) {
    setImportMsg("Could not read file: " + err.message, true);
    return;
  }

  let object3d;
  try {
    object3d = objLoader.parse(text);
  } catch (err) {
    setImportMsg("Could not parse OBJ: " + err.message, true);
    return;
  }

  const vertices = [];
  const triangles = [];
  object3d.traverse((child) => {
    if (!(child.isMesh && child.geometry && child.geometry.attributes.position)) return;
    const geom = child.geometry;
    const pos = geom.attributes.position;
    child.updateWorldMatrix(true, false);

    const worldPt = (i) => {
      const v = new THREE.Vector3().fromBufferAttribute(pos, i);
      v.applyMatrix4(child.matrixWorld);
      return [v.x, v.y, v.z];
    };

    for (let i = 0; i < pos.count; i++) vertices.push(worldPt(i));

    if (geom.index) {
      const idx = geom.index.array;
      for (let i = 0; i + 2 < idx.length; i += 3) {
        triangles.push([worldPt(idx[i]), worldPt(idx[i + 1]), worldPt(idx[i + 2])]);
      }
    } else {
      for (let i = 0; i + 2 < pos.count; i += 3) {
        triangles.push([worldPt(i), worldPt(i + 1), worldPt(i + 2)]);
      }
    }
  });

  if (vertices.length < 4) {
    setImportMsg("Could not find any geometry in that OBJ file.", true);
    return;
  }

  setImportMsg(`Analyzing ${triangles.length} triangles…`, false);
  try {
    const resp = await fetch("/api/import_obj", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ vertices, triangles }),
    });
    const data = await resp.json();
    if (data.error) throw new Error(data.error);
    buildScene(data);
    resetLoad();
    const [dx, dy, dz] = data.fitted_dimensions;
    const dims = `${dx.toFixed(3)} x ${dy.toFixed(3)} x ${dz.toFixed(3)} m`;
    if (data.method === "rectilinear reconstruction") {
      setImportMsg(
        `Reconstructed exactly as a rectilinear solid (bbox ${dims}). ` +
        `${data.n_plate} elements, clamped: ${data.bottom_face}.`, false);
    } else {
      setImportMsg(
        `Not rectilinear (${data.rectilinear_fallback_reason}) — fell back to a ` +
        `single fitted box: ${dims}. Remeshed at ${data.n_plate} plate + ` +
        `${data.n_joint} joint elements, clamped: ${data.bottom_face}.`, false);
    }
  } catch (err) {
    setImportMsg("Import failed: " + err.message, true);
  }
}

function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
