'use client';

import { useEffect, useRef } from 'react';

const PARTICLE_COUNT = 100;

interface LearnFieldProps {
  color?: string;
}

/**
 * Three.js particle + wireframe icosahedra background for learn domain pages.
 * Follows NeuralField pattern: lazy import, prefers-reduced-motion guard,
 * full cleanup. No top-level three import.
 */
export function LearnField({ color = '#60A5FA' }: LearnFieldProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

    let cleanup: (() => void) | undefined;
    let cancelled = false;

    import('three').then((THREE) => {
      if (cancelled || !canvas) return;

      let renderer: InstanceType<typeof THREE.WebGLRenderer>;
      try {
        renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
      } catch {
        return; // no WebGL — fallback gradient stays visible
      }

      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
      camera.position.z = 12;

      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      renderer.setPixelRatio(dpr);

      const group = new THREE.Group();
      scene.add(group);

      // ── Layer 1: particles ────────────────────────────────────────────
      const positions = new Float32Array(PARTICLE_COUNT * 3);
      for (let i = 0; i < PARTICLE_COUNT; i++) {
        const r = 7 * Math.cbrt(Math.random());
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        positions[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
        positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta) * 0.6;
        positions[i * 3 + 2] = r * Math.cos(phi) * 0.6;
      }
      const particleGeo = new THREE.BufferGeometry();
      particleGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      const particleMat = new THREE.PointsMaterial({
        color: new THREE.Color(color),
        size: 0.07,
        transparent: true,
        opacity: 0.6,
        sizeAttenuation: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      const points = new THREE.Points(particleGeo, particleMat);
      group.add(points);

      // ── Layer 2: wireframe icosahedra ─────────────────────────────────
      type IcoMesh = InstanceType<typeof THREE.Mesh> & {
        _speed: number;
        _axis: 'x' | 'y' | 'z';
      };

      const icoGeos: InstanceType<typeof THREE.IcosahedronGeometry>[] = [];
      const icoMats: InstanceType<typeof THREE.MeshBasicMaterial>[] = [];
      const icos: IcoMesh[] = [];

      for (let i = 0; i < 8; i++) {
        const geo = new THREE.IcosahedronGeometry(0.35, 0);
        const mat = new THREE.MeshBasicMaterial({
          color: new THREE.Color(color),
          wireframe: true,
          transparent: true,
          opacity: 0.12,
        });
        const mesh = new THREE.Mesh(geo, mat) as unknown as IcoMesh;

        // random position within radius 5
        const r = 5 * Math.cbrt(Math.random());
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        mesh.position.set(
          r * Math.sin(phi) * Math.cos(theta),
          r * Math.sin(phi) * Math.sin(theta),
          r * Math.cos(phi),
        );

        // random rotation metadata
        mesh._speed = 0.3 + Math.random() * 0.5; // 0.3–0.8
        const axes = ['x', 'y', 'z'] as const;
        mesh._axis = axes[Math.floor(Math.random() * 3)];

        group.add(mesh);
        icoGeos.push(geo);
        icoMats.push(mat);
        icos.push(mesh);
      }

      const resize = () => {
        const parent = canvas.parentElement;
        if (!parent) return;
        const { clientWidth: w, clientHeight: h } = parent;
        renderer.setSize(w, h, false);
        camera.aspect = w / Math.max(h, 1);
        camera.updateProjectionMatrix();
      };
      resize();
      window.addEventListener('resize', resize);

      let raf = 0;
      const clock = new THREE.Clock();

      const animate = () => {
        const t = clock.getElapsedTime();
        const delta = clock.getDelta();

        group.rotation.y = t * 0.02;

        // rotate each icosahedron on its own axis
        for (const ico of icos) {
          ico.rotation[ico._axis] += ico._speed * delta;
        }

        raf = requestAnimationFrame(animate);
        renderer.render(scene, camera);
      };
      animate();

      cleanup = () => {
        cancelAnimationFrame(raf);
        window.removeEventListener('resize', resize);
        particleGeo.dispose();
        particleMat.dispose();
        icoGeos.forEach(g => g.dispose());
        icoMats.forEach(m => m.dispose());
        renderer.dispose();
      };
    });

    return () => {
      cancelled = true;
      cleanup?.();
    };
  }, [color]);

  return <canvas ref={canvasRef} className="h-full w-full" aria-hidden="true" />;
}

export default LearnField;
