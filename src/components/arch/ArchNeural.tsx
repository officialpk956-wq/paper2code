'use client';

import { useEffect, useRef } from 'react';

const PARTICLE_COUNT = 180;
const CONNECT_DISTANCE = 2.0;
const FIELD_RADIUS = 7;

interface ArchNeuralProps {
  color?: string;
}

/**
 * Particle background for architecture detail pages.
 * Follows the same pattern as NeuralField — lazy Three.js import inside
 * useEffect, full cleanup, prefers-reduced-motion guard.
 * No pointer tracking so content remains easily readable.
 */
export function ArchNeural({ color = '#7C5CFF' }: ArchNeuralProps) {
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
        return; // no WebGL — fallback stays visible
      }

      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
      camera.position.z = 16;

      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      renderer.setPixelRatio(dpr);

      const group = new THREE.Group();
      scene.add(group);

      // particles
      const positions = new Float32Array(PARTICLE_COUNT * 3);
      const phases = new Float32Array(PARTICLE_COUNT);
      for (let i = 0; i < PARTICLE_COUNT; i++) {
        const r = FIELD_RADIUS * Math.cbrt(Math.random());
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        positions[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
        positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta) * 0.6;
        positions[i * 3 + 2] = r * Math.cos(phi) * 0.6;
        phases[i] = Math.random() * Math.PI * 2;
      }

      const particleGeo = new THREE.BufferGeometry();
      particleGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));

      const particleColor = new THREE.Color(color);
      const particleMat = new THREE.PointsMaterial({
        color: particleColor,
        size: 0.08,
        transparent: true,
        opacity: 0.7,
        sizeAttenuation: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      const points = new THREE.Points(particleGeo, particleMat);
      group.add(points);

      // static connection lines — computed once
      const linePositions: number[] = [];
      for (let i = 0; i < PARTICLE_COUNT; i++) {
        for (let j = i + 1; j < PARTICLE_COUNT; j++) {
          const dx = positions[i * 3]     - positions[j * 3];
          const dy = positions[i * 3 + 1] - positions[j * 3 + 1];
          const dz = positions[i * 3 + 2] - positions[j * 3 + 2];
          const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
          if (dist < CONNECT_DISTANCE) {
            linePositions.push(
              positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2],
              positions[j * 3], positions[j * 3 + 1], positions[j * 3 + 2],
            );
          }
        }
      }
      const lineGeo = new THREE.BufferGeometry();
      lineGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(linePositions), 3));
      const lineMat = new THREE.LineBasicMaterial({
        color: 0x00e5ff,
        transparent: true,
        opacity: 0.07,
      });
      const lines = new THREE.LineSegments(lineGeo, lineMat);
      group.add(lines);

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
        // slow y-only rotation — no pointer tracking so text stays readable
        group.rotation.y = t * 0.015;
        const breathe = 1 + Math.sin(t * 0.3) * 0.03;
        group.scale.setScalar(breathe);
        raf = requestAnimationFrame(animate);
        renderer.render(scene, camera);
      };
      animate();

      cleanup = () => {
        cancelAnimationFrame(raf);
        window.removeEventListener('resize', resize);
        particleGeo.dispose();
        particleMat.dispose();
        lineGeo.dispose();
        lineMat.dispose();
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

export default ArchNeural;
