'use client';

import { useEffect, useRef } from 'react';
import { usePrefersReducedMotion } from './usePrefersReducedMotion';

interface Props {
  className?: string;
  /** Total size in px. */
  size?: number;
  /** "ico" wireframe or "sphere" point cloud. */
  variant?: 'ico' | 'sphere';
  /** Primary edge/glow color as "r,g,b". Default cyan. */
  rgbA?: string;
  /** Secondary edge color as "r,g,b". Default violet. */
  rgbB?: string;
  /** Solid node color. */
  node?: string;
}

interface V3 {
  x: number;
  y: number;
  z: number;
}

/**
 * Canvas-projected rotating wireframe (icosahedron) or a point-cloud sphere.
 * Slowly auto-rotates on all axes, gently bobs, and reacts subtly to the
 * cursor. Reduced motion -> static single frame. Colors are theme-driven.
 */
export function WireframeSolid({
  className,
  size = 260,
  variant = 'ico',
  rgbA = '0,229,255',
  rgbB = '124,92,255',
  node = '#66F5FF',
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const reduced = usePrefersReducedMotion();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const verts: V3[] = [];
    const edges: [number, number][] = [];

    if (variant === 'ico') {
      const t = (1 + Math.sqrt(5)) / 2;
      const raw = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
      ];
      const s = size * 0.32;
      const norm = Math.sqrt(1 + t * t);
      for (const [x, y, z] of raw) verts.push({ x: (x / norm) * s, y: (y / norm) * s, z: (z / norm) * s });
      const F: [number, number, number][] = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
      ];
      const seen = new Set<string>();
      for (const [a, b, c] of F) {
        for (const [i, j] of [[a, b], [b, c], [c, a]] as const) {
          const k = i < j ? `${i}-${j}` : `${j}-${i}`;
          if (!seen.has(k)) { seen.add(k); edges.push([i, j]); }
        }
      }
    } else {
      const N = 220;
      const r = size * 0.36;
      for (let i = 0; i < N; i++) {
        const phi = Math.acos(1 - (2 * (i + 0.5)) / N);
        const theta = Math.PI * (1 + Math.sqrt(5)) * i;
        verts.push({
          x: r * Math.sin(phi) * Math.cos(theta),
          y: r * Math.sin(phi) * Math.sin(theta),
          z: r * Math.cos(phi),
        });
      }
    }

    const cx = size / 2;
    const cy = size / 2;
    const focal = size * 1.6;
    let raf = 0;
    let running = true;
    let visible = true;
    const t0 = performance.now();
    const mouse = { x: 0, y: 0, tx: 0, ty: 0 };

    const rot = (v: V3, ax: number, ay: number, az: number): V3 => {
      let { x, y, z } = v;
      let c = Math.cos(ay), s = Math.sin(ay);
      let nx = c * x + s * z; let nz = -s * x + c * z;
      x = nx; z = nz;
      c = Math.cos(ax); s = Math.sin(ax);
      let ny = c * y - s * z; nz = s * y + c * z;
      y = ny; z = nz;
      c = Math.cos(az); s = Math.sin(az);
      nx = c * x - s * y; ny = s * x + c * y;
      x = nx; y = ny;
      return { x, y, z };
    };

    const project = (v: V3) => {
      const scale = focal / (focal + v.z);
      return { x: cx + v.x * scale, y: cy + v.y * scale, s: scale };
    };

    const draw = () => {
      const t = (performance.now() - t0) / 1000;
      mouse.x += (mouse.tx - mouse.x) * 0.06;
      mouse.y += (mouse.ty - mouse.y) * 0.06;

      const ax = t * 0.25 + mouse.y * 0.35;
      const ay = t * 0.4 + mouse.x * 0.5;
      const az = Math.sin(t * 0.3) * 0.15;
      const bob = Math.sin(t * 0.9) * 4;

      ctx.clearRect(0, 0, size, size);
      ctx.save();
      ctx.translate(0, bob);

      const projected = verts.map((v) => {
        const r = rot(v, ax, ay, az);
        return { p: project(r), z: r.z };
      });

      if (variant === 'ico') {
        for (const [a, b] of edges) {
          const pa = projected[a]; const pb = projected[b];
          const zavg = (pa.z + pb.z) / 2;
          const depth = (zavg + size * 0.4) / (size * 0.8);
          const alpha = Math.max(0.08, Math.min(0.9, 0.15 + depth * 0.75));
          const grad = ctx.createLinearGradient(pa.p.x, pa.p.y, pb.p.x, pb.p.y);
          grad.addColorStop(0, `rgba(${rgbA},${(alpha * 0.9).toFixed(3)})`);
          grad.addColorStop(1, `rgba(${rgbB},${(alpha * 0.7).toFixed(3)})`);
          ctx.strokeStyle = grad;
          ctx.lineWidth = 0.6 + depth * 1.4;
          ctx.beginPath();
          ctx.moveTo(pa.p.x, pa.p.y);
          ctx.lineTo(pb.p.x, pb.p.y);
          ctx.stroke();
        }
        for (const { p, z } of projected) {
          const depth = (z + size * 0.4) / (size * 0.8);
          const r = 1.5 + depth * 3.2;
          const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, r * 4);
          g.addColorStop(0, `rgba(${rgbA},${(0.4 * depth).toFixed(3)})`);
          g.addColorStop(1, 'rgba(0,0,0,0)');
          ctx.fillStyle = g;
          ctx.beginPath(); ctx.arc(p.x, p.y, r * 4, 0, Math.PI * 2); ctx.fill();
          ctx.fillStyle = node;
          ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2); ctx.fill();
        }
      } else {
        projected.sort((a, b) => a.z - b.z);
        for (const { p, z } of projected) {
          const depth = (z + size * 0.4) / (size * 0.8);
          const r = 0.6 + depth * 2.4;
          ctx.fillStyle = `rgba(${rgbB},${(0.15 + depth * 0.65).toFixed(3)})`;
          ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2); ctx.fill();
        }
      }

      ctx.restore();
    };

    const loop = () => {
      if (running && visible) draw();
      raf = requestAnimationFrame(loop);
    };

    draw();
    if (reduced) return;

    const onMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      mouse.tx = ((e.clientX - rect.left) / rect.width - 0.5) * 2;
      mouse.ty = ((e.clientY - rect.top) / rect.height - 0.5) * 2;
    };
    const onVis = () => { running = !document.hidden; };
    const io = new IntersectionObserver(
      (entries) => { for (const en of entries) visible = en.isIntersecting; },
      { threshold: 0.01 },
    );
    io.observe(canvas);
    window.addEventListener('mousemove', onMove);
    document.addEventListener('visibilitychange', onVis);
    raf = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('mousemove', onMove);
      document.removeEventListener('visibilitychange', onVis);
      io.disconnect();
    };
  }, [reduced, size, variant, rgbA, rgbB, node]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{ width: size, height: size, display: 'block' }}
      aria-hidden
    />
  );
}
