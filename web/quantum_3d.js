/**
 * Q-Glitch Quantum 3D Engine
 * Interactive Three.js WebGL Quantum Hilbert Space & Bloch Sphere Visualizer
 *
 * Author: Mallikarjun.R (Team Lead) & Team Q-Glitch
 * Event: VTU-Qiskit Fall Fest Hackathon 2026
 * Team ID: VTU-QFF-HC-0190
 */

(function () {
  'use strict';

  // Global state for 3D engine
  window.QGlitch3D = {
    backgroundScene: null,
    blochScene: null,
    updateQubitAngles: null,
    pulseEntanglement: null,
  };

  /* ==========================================================================
     1. BACKGROUND QUANTUM WAVE & PARTICLE LATTICE
     ========================================================================== */
  function initQuantumBackground() {
    const container = document.getElementById('quantum-canvas-container');
    if (!container || typeof THREE === 'undefined') return;

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x060913, 0.0018);

    const camera = new THREE.PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      1,
      2000
    );
    camera.position.set(0, 80, 280);

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);
    container.appendChild(renderer.domElement);

    // Particle field
    const particleCount = 750;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    const initialY = new Float32Array(particleCount);
    const speeds = new Float32Array(particleCount);

    const color1 = new THREE.Color(0x00f5d4); // Cyan
    const color2 = new THREE.Color(0x7c3aed); // Violet
    const color3 = new THREE.Color(0xec4899); // Magenta

    const spreadX = 700;
    const spreadY = 220;
    const spreadZ = 600;

    for (let i = 0; i < particleCount; i++) {
      const x = (Math.random() - 0.5) * spreadX;
      const y = (Math.random() - 0.5) * spreadY - 20;
      const z = (Math.random() - 0.5) * spreadZ;

      positions[i * 3] = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;
      initialY[i] = y;
      speeds[i] = 0.5 + Math.random() * 0.8;

      // Color interpolation
      const t = Math.random();
      const c = t < 0.5 ? color1.clone().lerp(color2, t * 2) : color2.clone().lerp(color3, (t - 0.5) * 2);
      colors[i * 3] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    // Custom circle texture for soft glowing particles
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    grad.addColorStop(0, 'rgba(255,255,255,1)');
    grad.addColorStop(0.3, 'rgba(0,245,212,0.8)');
    grad.addColorStop(0.7, 'rgba(124,58,237,0.3)');
    grad.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 64, 64);
    const particleTexture = new THREE.CanvasTexture(canvas);

    const material = new THREE.PointsMaterial({
      size: 4.5,
      map: particleTexture,
      transparent: true,
      vertexColors: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    const particles = new THREE.Points(geometry, material);
    scene.add(particles);

    // Dynamic wave mesh (representing quantum vacuum fluctuation)
    const gridCols = 36;
    const gridRows = 36;
    const waveGeo = new THREE.PlaneGeometry(650, 650, gridCols, gridRows);
    waveGeo.rotateX(-Math.PI / 2);
    waveGeo.translate(0, -90, 0);

    const waveMat = new THREE.MeshBasicMaterial({
      color: 0x312e81,
      wireframe: true,
      transparent: true,
      opacity: 0.18,
    });
    const waveMesh = new THREE.Mesh(waveGeo, waveMat);
    scene.add(waveMesh);

    // Mouse parallax tracking
    let mouseX = 0;
    let mouseY = 0;
    let targetCameraX = 0;
    let targetCameraY = 80;

    window.addEventListener('mousemove', (e) => {
      const halfW = window.innerWidth / 2;
      const halfH = window.innerHeight / 2;
      mouseX = (e.clientX - halfW) / halfW;
      mouseY = (e.clientY - halfH) / halfH;
      targetCameraX = mouseX * 50;
      targetCameraY = 80 + (-mouseY * 30);
    });

    // Resize handling
    window.addEventListener('resize', () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });

    // Animation Loop
    let clock = new THREE.Clock();

    function animate() {
      requestAnimationFrame(animate);
      const elapsed = clock.getElapsedTime();

      // Smooth camera interpolation
      camera.position.x += (targetCameraX - camera.position.x) * 0.03;
      camera.position.y += (targetCameraY - camera.position.y) * 0.03;
      camera.lookAt(0, 0, 0);

      // Particle wave motion
      const posAttr = geometry.attributes.position;
      const posArray = posAttr.array;
      for (let i = 0; i < particleCount; i++) {
        const x = posArray[i * 3];
        const z = posArray[i * 3 + 2];
        // Quantum probability wave equation
        const wave = Math.sin(x * 0.015 + elapsed * speeds[i]) * Math.cos(z * 0.015 + elapsed * 0.7) * 16;
        posArray[i * 3 + 1] = initialY[i] + wave;
      }
      posAttr.needsUpdate = true;

      // Animate ground wireframe wave
      const wavePos = waveGeo.attributes.position;
      const wArray = wavePos.array;
      for (let j = 0; j < wArray.length; j += 3) {
        const wx = wArray[j];
        const wz = wArray[j + 2];
        wArray[j + 1] = Math.sin(wx * 0.02 + elapsed * 0.9) * Math.cos(wz * 0.02 + elapsed * 0.9) * 12;
      }
      wavePos.needsUpdate = true;

      particles.rotation.y = elapsed * 0.03;
      renderer.render(scene, camera);
    }

    animate();
    window.QGlitch3D.backgroundScene = scene;
  }

  /* ==========================================================================
     2. INTERACTIVE 3D BLOCH SPHERE & 6-QUBIT LATTICE
     ========================================================================== */
  function initBlochLab() {
    const container = document.getElementById('three-bloch-container');
    if (!container || typeof THREE === 'undefined') return;

    const width = container.clientWidth || 600;
    const height = container.clientHeight || 420;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(0, 22, 68);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const pointCyan = new THREE.PointLight(0x00f5d4, 2.5, 120);
    pointCyan.position.set(30, 40, 30);
    scene.add(pointCyan);

    const pointViolet = new THREE.PointLight(0xa855f7, 2.5, 120);
    pointViolet.position.set(-30, -30, -30);
    scene.add(pointViolet);

    // Root Group for interactive rotation
    const systemGroup = new THREE.Group();
    scene.add(systemGroup);

    /* -----------------------------------------
       Central Bloch Sphere Core
       ----------------------------------------- */
    const sphereRadius = 14;

    // Translucent core sphere
    const sphereGeo = new THREE.SphereGeometry(sphereRadius, 36, 36);
    const sphereMat = new THREE.MeshPhysicalMaterial({
      color: 0x0f172a,
      roughness: 0.25,
      metalness: 0.1,
      transmission: 0.7,
      thickness: 1.2,
      transparent: true,
      opacity: 0.35,
      wireframe: false,
    });
    const blochSphere = new THREE.Mesh(sphereGeo, sphereMat);
    systemGroup.add(blochSphere);

    // Wireframe grid lines (Latitude & Meridian)
    const ringMatCyan = new THREE.LineBasicMaterial({ color: 0x00f5d4, transparent: true, opacity: 0.75 });
    const ringMatViolet = new THREE.LineBasicMaterial({ color: 0x8b5cf6, transparent: true, opacity: 0.75 });
    const ringMatWhite = new THREE.LineBasicMaterial({ color: 0x94a3b8, transparent: true, opacity: 0.4 });

    // Equator ring (X-Y plane)
    const equatorGeo = new THREE.BufferGeometry();
    const equatorPts = [];
    for (let i = 0; i <= 64; i++) {
      const theta = (i / 64) * Math.PI * 2;
      equatorPts.push(new THREE.Vector3(Math.cos(theta) * sphereRadius, 0, Math.sin(theta) * sphereRadius));
    }
    equatorGeo.setFromPoints(equatorPts);
    const equatorLine = new THREE.Line(equatorGeo, ringMatCyan);
    systemGroup.add(equatorLine);

    // Meridian ring 1 (X-Z plane)
    const meridian1Geo = new THREE.BufferGeometry();
    const meridian1Pts = [];
    for (let i = 0; i <= 64; i++) {
      const theta = (i / 64) * Math.PI * 2;
      meridian1Pts.push(new THREE.Vector3(Math.cos(theta) * sphereRadius, Math.sin(theta) * sphereRadius, 0));
    }
    meridian1Geo.setFromPoints(meridian1Pts);
    const meridian1Line = new THREE.Line(meridian1Geo, ringMatViolet);
    systemGroup.add(meridian1Line);

    // Meridian ring 2 (Y-Z plane)
    const meridian2Geo = new THREE.BufferGeometry();
    const meridian2Pts = [];
    for (let i = 0; i <= 64; i++) {
      const theta = (i / 64) * Math.PI * 2;
      meridian2Pts.push(new THREE.Vector3(0, Math.sin(theta) * sphereRadius, Math.cos(theta) * sphereRadius));
    }
    meridian2Geo.setFromPoints(meridian2Pts);
    const meridian2Line = new THREE.Line(meridian2Geo, ringMatWhite);
    systemGroup.add(meridian2Line);

    // Coordinate Axes (Z: |0> to |1>, X: |+>, Y: |i>)
    function createAxis(from, to, colorHex) {
      const mat = new THREE.LineBasicMaterial({ color: colorHex, linewidth: 2 });
      const geo = new THREE.BufferGeometry().setFromPoints([from, to]);
      return new THREE.Line(geo, mat);
    }
    const axisLen = sphereRadius * 1.35;
    systemGroup.add(createAxis(new THREE.Vector3(0, -axisLen, 0), new THREE.Vector3(0, axisLen, 0), 0x38bdf8)); // Z
    systemGroup.add(createAxis(new THREE.Vector3(-axisLen, 0, 0), new THREE.Vector3(axisLen, 0, 0), 0xa855f7)); // X
    systemGroup.add(createAxis(new THREE.Vector3(0, 0, -axisLen), new THREE.Vector3(0, 0, axisLen), 0x10b981)); // Y

    // State Vector |Ψ⟩ Arrow in Central Bloch Sphere
    const arrowDir = new THREE.Vector3(0.5, 0.8, 0.3).normalize();
    const arrowOrigin = new THREE.Vector3(0, 0, 0);
    const stateArrow = new THREE.ArrowHelper(arrowDir, arrowOrigin, sphereRadius * 0.95, 0xef4444, 2.5, 1.4);
    systemGroup.add(stateArrow);

    /* -----------------------------------------
       6 Superconducting Qubit Nodes
       ----------------------------------------- */
    const qubitNodes = [];
    const qubitNames = [
      'Q0: LMNA',
      'Q1: PROG',
      'Q2: TELO',
      'Q3: ZMP',
      'Q4: CYT',
      'Q5: CARD',
    ];
    const orbitRadius = 26;
    const qubitColors = [0x00f5d4, 0x38bdf8, 0x818cf8, 0xc084fc, 0xf472b6, 0xfb7185];

    // Helper: Canvas Text Sprite for Qubit Labels
    function makeTextSprite(message, colorHex) {
      const canvas = document.createElement('canvas');
      canvas.width = 256;
      canvas.height = 70;
      const c = canvas.getContext('2d');
      c.fillStyle = 'rgba(15, 23, 42, 0.85)';
      c.strokeStyle = colorHex;
      c.lineWidth = 3;
      c.roundRect(4, 4, 248, 62, 10);
      c.fill();
      c.stroke();

      c.font = 'bold 26px "JetBrains Mono", monospace';
      c.fillStyle = '#ffffff';
      c.textAlign = 'center';
      c.textBaseline = 'middle';
      c.fillText(message, 128, 35);

      const texture = new THREE.CanvasTexture(canvas);
      const spriteMaterial = new THREE.SpriteMaterial({ map: texture, transparent: true });
      const sprite = new THREE.Sprite(spriteMaterial);
      sprite.scale.set(7.5, 2.2, 1);
      return sprite;
    }

    for (let k = 0; k < 6; k++) {
      const angle = (k / 6) * Math.PI * 2;
      const x = Math.cos(angle) * orbitRadius;
      const z = Math.sin(angle) * orbitRadius;
      const y = Math.sin(k * 1.5) * 4.5; // slight 3D elevation

      const qGroup = new THREE.Group();
      qGroup.position.set(x, y, z);

      // Outer metallic ring
      const ringGeo = new THREE.TorusGeometry(2.2, 0.22, 16, 32);
      const ringMat = new THREE.MeshStandardMaterial({
        color: qubitColors[k],
        emissive: qubitColors[k],
        emissiveIntensity: 0.4,
        metalness: 0.9,
        roughness: 0.2,
      });
      const torus = new THREE.Mesh(ringGeo, ringMat);
      torus.rotation.x = Math.PI / 2;
      qGroup.add(torus);

      // Glowing core sphere
      const coreGeo = new THREE.SphereGeometry(1.2, 24, 24);
      const coreMat = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        emissive: qubitColors[k],
        emissiveIntensity: 0.9,
        roughness: 0.1,
      });
      const coreMesh = new THREE.Mesh(coreGeo, coreMat);
      qGroup.add(coreMesh);

      // Local state vector needle
      const needleDir = new THREE.Vector3(0, 1, 0).normalize();
      const needle = new THREE.ArrowHelper(needleDir, new THREE.Vector3(0, 0, 0), 3.2, qubitColors[k], 1.0, 0.6);
      qGroup.add(needle);

      // Label sprite
      const hexStr = '#' + qubitColors[k].toString(16).padStart(6, '0');
      const label = makeTextSprite(qubitNames[k], hexStr);
      label.position.set(0, 3.8, 0);
      qGroup.add(label);

      systemGroup.add(qGroup);

      qubitNodes.push({
        group: qGroup,
        core: coreMesh,
        torus: torus,
        needle: needle,
        baseAngle: angle,
        basePos: new THREE.Vector3(x, y, z),
        phaseAngle: 0.8 + k * 0.4, // Default radians
        color: qubitColors[k],
      });
    }

    /* -----------------------------------------
       Full Entanglement Network (ZZ Feature Map)
       ----------------------------------------- */
    const entanglementLines = [];
    const pulsePackets = [];

    const lineMat = new THREE.LineBasicMaterial({
      color: 0x7c3aed,
      transparent: true,
      opacity: 0.45,
    });

    // Create pairwise lines between all 6 qubits
    for (let i = 0; i < 6; i++) {
      for (let j = i + 1; j < 6; j++) {
        const p1 = qubitNodes[i].basePos;
        const p2 = qubitNodes[j].basePos;

        // Arching Bezier curve through 3D space
        const midPoint = new THREE.Vector3()
          .addVectors(p1, p2)
          .multiplyScalar(0.5);
        midPoint.y += 3.5; // elevate arch

        const curve = new THREE.QuadraticBezierCurve3(p1, midPoint, p2);
        const pts = curve.getPoints(24);
        const geo = new THREE.BufferGeometry().setFromPoints(pts);
        const line = new THREE.Line(geo, lineMat.clone());
        systemGroup.add(line);

        entanglementLines.push({ line, curve, p1Idx: i, p2Idx: j });

        // Add animated photon packet
        const packetGeo = new THREE.SphereGeometry(0.45, 12, 12);
        const packetMat = new THREE.MeshBasicMaterial({ color: 0x00f5d4 });
        const packetMesh = new THREE.Mesh(packetGeo, packetMat);
        packetMesh.visible = false;
        systemGroup.add(packetMesh);
        pulsePackets.push({ mesh: packetMesh, curve, progress: Math.random(), speed: 0.008 + Math.random() * 0.006 });
      }
    }

    /* -----------------------------------------
       Interaction: Mouse Orbit Controls & Dragging
       ----------------------------------------- */
    let isDragging = false;
    let prevMouseX = 0;
    let prevMouseY = 0;
    let rotationVelocityX = 0;
    let rotationVelocityY = 0.003; // idle spin
    let autoRotate = true;

    const domElement = renderer.domElement;

    domElement.addEventListener('mousedown', (e) => {
      isDragging = true;
      prevMouseX = e.clientX;
      prevMouseY = e.clientY;
    });

    window.addEventListener('mouseup', () => {
      isDragging = false;
    });

    window.addEventListener('mousemove', (e) => {
      if (!isDragging) return;
      const deltaX = e.clientX - prevMouseX;
      const deltaY = e.clientY - prevMouseY;
      prevMouseX = e.clientX;
      prevMouseY = e.clientY;

      rotationVelocityY = deltaX * 0.006;
      rotationVelocityX = deltaY * 0.006;

      systemGroup.rotation.y += rotationVelocityY;
      systemGroup.rotation.x += rotationVelocityX;
    });

    // Touch support for mobile / tablets
    domElement.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        isDragging = true;
        prevMouseX = e.touches[0].clientX;
        prevMouseY = e.touches[0].clientY;
      }
    }, { passive: true });

    window.addEventListener('touchend', () => {
      isDragging = false;
    });

    window.addEventListener('touchmove', (e) => {
      if (!isDragging || e.touches.length !== 1) return;
      const deltaX = e.touches[0].clientX - prevMouseX;
      const deltaY = e.touches[0].clientY - prevMouseY;
      prevMouseX = e.touches[0].clientX;
      prevMouseY = e.touches[0].clientY;

      systemGroup.rotation.y += deltaX * 0.006;
      systemGroup.rotation.x += deltaY * 0.006;
    }, { passive: true });

    // Zoom on wheel
    domElement.addEventListener('wheel', (e) => {
      e.preventDefault();
      camera.position.z += e.deltaY * 0.05;
      camera.position.z = Math.max(35, Math.min(110, camera.position.z));
    }, { passive: false });

    // UI Buttons: Auto-Rotate and Pulse
    const btnRotate = document.getElementById('btn-rotate-qubits');
    if (btnRotate) {
      btnRotate.addEventListener('click', () => {
        autoRotate = !autoRotate;
        btnRotate.classList.toggle('active', autoRotate);
      });
    }

    const btnPulse = document.getElementById('btn-pulse-entangle');
    function triggerEntanglementPulse() {
      // Light up all lines with electric violet surge
      entanglementLines.forEach(({ line }) => {
        line.material.color.setHex(0x00f5d4);
        line.material.opacity = 0.95;
      });
      // Show traveling packets
      pulsePackets.forEach((p) => {
        p.mesh.visible = true;
        p.speed = 0.035;
      });

      // Decay back
      setTimeout(() => {
        entanglementLines.forEach(({ line }) => {
          line.material.color.setHex(0x7c3aed);
          line.material.opacity = 0.45;
        });
        pulsePackets.forEach((p) => {
          p.speed = 0.008;
        });
      }, 1200);
    }

    if (btnPulse) {
      btnPulse.addEventListener('click', triggerEntanglementPulse);
    }

    // Dynamic angle update API
    window.QGlitch3D.updateQubitAngles = function (angles) {
      if (!Array.isArray(angles) || angles.length < 6) return;

      angles.forEach((rad, idx) => {
        if (qubitNodes[idx]) {
          qubitNodes[idx].phaseAngle = rad;
          // Rotate needle on local Y-Z plane according to rotation angle rad
          const dir = new THREE.Vector3(
            Math.sin(rad),
            Math.cos(rad),
            Math.sin(rad * 2) * 0.5
          ).normalize();
          qubitNodes[idx].needle.setDirection(dir);

          // Pulse intensity proportional to angle
          const intensity = 0.4 + (rad / Math.PI) * 1.2;
          qubitNodes[idx].core.material.emissiveIntensity = intensity;
        }
      });

      // Update central state vector direction to average superposition
      const avgX = Math.sin(angles[0]) + Math.cos(angles[1]);
      const avgY = Math.cos(angles[2]) + Math.sin(angles[3]);
      const avgZ = Math.cos(angles[4]) + Math.sin(angles[5]);
      const newDir = new THREE.Vector3(avgX, avgY, avgZ).normalize();
      stateArrow.setDirection(newDir);

      triggerEntanglementPulse();
    };

    window.QGlitch3D.pulseEntanglement = triggerEntanglementPulse;

    // Resize handling
    const resizeObserver = new ResizeObserver(() => {
      const w = container.clientWidth || 600;
      const h = container.clientHeight || 420;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });
    resizeObserver.observe(container);

    // Animation Loop
    let clock = new THREE.Clock();

    function renderBloch() {
      requestAnimationFrame(renderBloch);
      const elapsed = clock.getElapsedTime();

      // Inertia / damping
      if (!isDragging) {
        if (autoRotate) {
          systemGroup.rotation.y += 0.004;
        } else {
          systemGroup.rotation.y += rotationVelocityY;
          systemGroup.rotation.x += rotationVelocityX;
          rotationVelocityX *= 0.94;
          rotationVelocityY *= 0.94;
        }
      }

      // Gentle floating oscillation for qubit nodes
      qubitNodes.forEach((q, idx) => {
        const floatDelta = Math.sin(elapsed * 1.8 + idx) * 0.04;
        q.group.position.y = q.basePos.y + floatDelta;
        q.torus.rotation.z = elapsed * 0.8;
      });

      // Animate entanglement photons
      pulsePackets.forEach((p) => {
        p.progress = (p.progress + p.speed) % 1;
        const pt = p.curve.getPoint(p.progress);
        p.mesh.position.copy(pt);
      });

      // Pulse central sphere glow
      blochSphere.material.opacity = 0.3 + Math.sin(elapsed * 2) * 0.08;

      renderer.render(scene, camera);
    }

    renderBloch();
    window.QGlitch3D.blochScene = scene;
  }

  // Self-initialize once DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      initQuantumBackground();
      initBlochLab();
    });
  } else {
    initQuantumBackground();
    initBlochLab();
  }
})();
