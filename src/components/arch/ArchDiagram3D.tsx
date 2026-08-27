'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';

import ArchDiagram from './ArchDiagram';
import { GENERIC_FLOWS, toDiagramSlug, type GenBlock } from './archFlows';

interface ArchDiagram3DProps {
  slug: string;
}

type LayoutKind = 'linear' | 'encoder-decoder' | 'gan' | 'unet';

const SLAB_WIDTH = 1.8;
const SLAB_HEIGHT = 0.92;
const SLAB_DEPTH = 0.36;
const CORE_COLOR = '#A78BFA';

export function resolve3DFlowBlocks(slug: string): GenBlock[] {
  return GENERIC_FLOWS[slug] ?? GENERIC_FLOWS[toDiagramSlug(slug) ?? ''] ?? [];
}

function chooseLayout(slug: string, blocks: GenBlock[]): LayoutKind {
  const resolvedSlug = toDiagramSlug(slug) ?? slug;
  const labels = blocks.map((block) => `${block.label} ${block.sub ?? ''}`.toLowerCase());
  const joinedLabels = labels.join(' ');

  if (resolvedSlug.includes('unet') || resolvedSlug.includes('u-net')) return 'unet';
  if (
    resolvedSlug === 'gan' ||
    (joinedLabels.includes('generator') && (joinedLabels.includes('discrim') || joinedLabels.includes('critic')))
  ) {
    return 'gan';
  }
  if (
    ['seq2seq', 'encdec', 'autoencoder', 'vae', 'transformer'].some((name) => resolvedSlug.includes(name)) ||
    (joinedLabels.includes('encoder') && joinedLabels.includes('decoder'))
  ) {
    return 'encoder-decoder';
  }
  return 'linear';
}

function linearPositions(count: number): THREE.Vector3[] {
  const gap = 2.18;
  return Array.from({ length: count }, (_, index) => (
    new THREE.Vector3((index - (count - 1) / 2) * gap, 0, index % 2 === 0 ? 0.04 : -0.04)
  ));
}

function towerPositions(count: number, splitIndex: number): THREE.Vector3[] {
  const safeSplit = Math.min(Math.max(splitIndex, 1), count - 1);
  const leftCount = safeSplit;
  const rightCount = count - safeSplit;
  const gap = 1.2;

  return Array.from({ length: count }, (_, index) => {
    if (index < safeSplit) {
      const y = ((leftCount - 1) / 2 - index) * gap;
      return new THREE.Vector3(-2.15, y, 0);
    }
    const rightIndex = index - safeSplit;
    const y = (rightIndex - (rightCount - 1) / 2) * gap;
    return new THREE.Vector3(2.15, y, 0);
  });
}

function uShapePositions(count: number): THREE.Vector3[] {
  if (count < 4) return linearPositions(count);

  const halfWidth = 2.45;
  const halfHeight = Math.max(1.35, Math.min(2.5, count * 0.28));
  const verticalLength = halfHeight * 2;
  const bottomLength = halfWidth * 2;
  const totalLength = verticalLength * 2 + bottomLength;

  return Array.from({ length: count }, (_, index) => {
    const distance = count === 1 ? 0 : (index / (count - 1)) * totalLength;
    if (distance <= verticalLength) {
      return new THREE.Vector3(-halfWidth, halfHeight - distance, 0);
    }
    if (distance <= verticalLength + bottomLength) {
      return new THREE.Vector3(-halfWidth + (distance - verticalLength), -halfHeight, 0);
    }
    return new THREE.Vector3(halfWidth, -halfHeight + (distance - verticalLength - bottomLength), 0);
  });
}

function layoutPositions(slug: string, blocks: GenBlock[]): THREE.Vector3[] {
  const layout = chooseLayout(slug, blocks);
  if (layout === 'unet') return uShapePositions(blocks.length);

  if (layout === 'gan') {
    const discriminatorIndex = blocks.findIndex((block) => /discrim|critic/i.test(block.label));
    return towerPositions(blocks.length, discriminatorIndex > 0 ? discriminatorIndex : Math.ceil(blocks.length / 2));
  }

  if (layout === 'encoder-decoder') {
    const decoderIndex = blocks.findIndex((block) => /decoder/i.test(`${block.label} ${block.sub ?? ''}`));
    return towerPositions(blocks.length, decoderIndex > 0 ? decoderIndex : Math.ceil(blocks.length / 2));
  }

  return linearPositions(blocks.length);
}

function createRoundedSlabGeometry(): THREE.ExtrudeGeometry {
  const width = SLAB_WIDTH;
  const height = SLAB_HEIGHT;
  const radius = 0.15;
  const shape = new THREE.Shape();

  shape.moveTo(-width / 2 + radius, -height / 2);
  shape.lineTo(width / 2 - radius, -height / 2);
  shape.quadraticCurveTo(width / 2, -height / 2, width / 2, -height / 2 + radius);
  shape.lineTo(width / 2, height / 2 - radius);
  shape.quadraticCurveTo(width / 2, height / 2, width / 2 - radius, height / 2);
  shape.lineTo(-width / 2 + radius, height / 2);
  shape.quadraticCurveTo(-width / 2, height / 2, -width / 2, height / 2 - radius);
  shape.lineTo(-width / 2, -height / 2 + radius);
  shape.quadraticCurveTo(-width / 2, -height / 2, -width / 2 + radius, -height / 2);

  const geometry = new THREE.ExtrudeGeometry(shape, {
    depth: SLAB_DEPTH,
    bevelEnabled: true,
    bevelSegments: 3,
    bevelSize: 0.045,
    bevelThickness: 0.045,
    curveSegments: 6,
    steps: 1,
  });
  geometry.center();
  return geometry;
}

function fitFont(context: CanvasRenderingContext2D, text: string, maxWidth: number, startSize: number): number {
  let size = startSize;
  while (size > 19) {
    context.font = `700 ${size}px Inter, Arial, sans-serif`;
    if (context.measureText(text).width <= maxWidth) break;
    size -= 1;
  }
  return size;
}

function createLabelTexture(block: GenBlock, accent: string): THREE.CanvasTexture | null {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 256;
  const context = canvas.getContext('2d');
  if (!context) return null;

  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = 'rgba(7, 7, 10, 0.82)';
  context.strokeStyle = accent;
  context.lineWidth = 4;
  context.beginPath();
  context.roundRect(12, 12, 488, 232, 34);
  context.fill();
  context.stroke();

  const titleSize = fitFont(context, block.label, 430, block.sub ? 42 : 48);
  context.font = `700 ${titleSize}px Inter, Arial, sans-serif`;
  context.fillStyle = '#FAFAFA';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillText(block.label, 256, block.sub ? 104 : 128, 430);

  if (block.sub) {
    context.font = '500 26px Inter, Arial, sans-serif';
    context.fillStyle = '#C4B5FD';
    context.fillText(block.sub, 256, 162, 430);
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.needsUpdate = true;
  return texture;
}

function fallbackColor(index: number, count: number): string {
  if (index === 0) return '#A5B4FC';
  if (index === count - 1) return '#34D399';
  return CORE_COLOR;
}

function connectBlocks(
  group: THREE.Group,
  start: THREE.Vector3,
  end: THREE.Vector3,
  material: THREE.MeshBasicMaterial,
  geometries: Set<THREE.BufferGeometry>,
): void {
  const direction = end.clone().sub(start);
  const fullLength = direction.length();
  if (fullLength <= 0.01) return;

  direction.normalize();
  const trim = Math.min(0.68, fullLength * 0.22);
  const beamStart = start.clone().addScaledVector(direction, trim);
  const beamEnd = end.clone().addScaledVector(direction, -trim);
  const beamLength = beamStart.distanceTo(beamEnd);
  if (beamLength <= 0.08) return;

  const beamGeometry = new THREE.CylinderGeometry(0.035, 0.035, beamLength, 10);
  geometries.add(beamGeometry);
  const beam = new THREE.Mesh(beamGeometry, material);
  beam.position.copy(beamStart).add(beamEnd).multiplyScalar(0.5);
  beam.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
  group.add(beam);

  const arrowGeometry = new THREE.ConeGeometry(0.11, 0.28, 12);
  geometries.add(arrowGeometry);
  const arrow = new THREE.Mesh(arrowGeometry, material);
  arrow.position.copy(beamEnd).addScaledVector(direction, -0.1);
  arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
  group.add(arrow);
}

export default function ArchDiagram3D({ slug }: ArchDiagram3DProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const blocks = useMemo(() => resolve3DFlowBlocks(slug), [slug]);
  const [webglFailed, setWebglFailed] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || blocks.length === 0) return;

    let cancelled = false;
    let renderer: THREE.WebGLRenderer | null = null;
    let resizeObserver: ResizeObserver | null = null;
    let animationFrame = 0;
    const geometries = new Set<THREE.BufferGeometry>();
    const materials = new Set<THREE.Material>();
    const textures = new Set<THREE.Texture>();
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const scene = new THREE.Scene();
    const flowGroup = new THREE.Group();
    const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 200);

    let dragging = false;
    let pointerX = 0;
    let pointerY = 0;
    let targetRotationX = -0.08;
    let targetRotationY = -0.22;
    let targetZoom = 10;
    let baseZoom = 10;
    let minZoom = 6;
    let maxZoom = 18;

    const render = () => renderer?.render(scene, camera);

    const onPointerDown = (event: PointerEvent) => {
      dragging = true;
      pointerX = event.clientX;
      pointerY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
      canvas.style.cursor = 'grabbing';
    };

    const onPointerMove = (event: PointerEvent) => {
      if (!dragging) return;
      const deltaX = event.clientX - pointerX;
      const deltaY = event.clientY - pointerY;
      pointerX = event.clientX;
      pointerY = event.clientY;
      targetRotationY += deltaX * 0.008;
      targetRotationX = THREE.MathUtils.clamp(targetRotationX + deltaY * 0.006, -0.72, 0.52);
    };

    const endPointerDrag = (event: PointerEvent) => {
      dragging = false;
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
      canvas.style.cursor = 'grab';
    };

    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      targetZoom = THREE.MathUtils.clamp(targetZoom + event.deltaY * 0.012, minZoom, maxZoom);
    };

    const teardown = () => {
      cancelAnimationFrame(animationFrame);
      resizeObserver?.disconnect();
      canvas.removeEventListener('pointerdown', onPointerDown);
      canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('pointerup', endPointerDrag);
      canvas.removeEventListener('pointercancel', endPointerDrag);
      canvas.removeEventListener('wheel', onWheel);
      geometries.forEach((geometry) => geometry.dispose());
      materials.forEach((material) => material.dispose());
      textures.forEach((texture) => texture.dispose());
      renderer?.dispose();
      renderer = null;
    };

    try {
      setWebglFailed(false);
      renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, powerPreference: 'high-performance' });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.08;
      renderer.setClearColor(0x0a0a0a, 1);

      scene.add(flowGroup);
      scene.add(new THREE.AmbientLight(0xffffff, 1.15));
      const keyLight = new THREE.DirectionalLight(0xffffff, 3.2);
      keyLight.position.set(4, 7, 9);
      scene.add(keyLight);

      const positions = layoutPositions(slug, blocks);
      const connectorMaterial = new THREE.MeshBasicMaterial({ color: '#76659A', transparent: true, opacity: 0.78 });
      materials.add(connectorMaterial);

      positions.slice(0, -1).forEach((position, index) => {
        connectBlocks(flowGroup, position, positions[index + 1], connectorMaterial, geometries);
      });

      blocks.forEach((block, index) => {
        const accent = block.accent ?? fallbackColor(index, blocks.length);
        const color = new THREE.Color(accent);
        const geometry = createRoundedSlabGeometry();
        geometries.add(geometry);

        const material = new THREE.MeshStandardMaterial({
          color,
          emissive: color.clone().multiplyScalar(0.16),
          emissiveIntensity: 0.9,
          metalness: 0.3,
          roughness: 0.34,
        });
        materials.add(material);

        const slab = new THREE.Mesh(geometry, material);
        slab.position.copy(positions[index]);
        flowGroup.add(slab);

        const edgeGeometry = new THREE.EdgesGeometry(geometry, 24);
        geometries.add(edgeGeometry);
        const edgeMaterial = new THREE.LineBasicMaterial({ color: color.clone().lerp(new THREE.Color('#FFFFFF'), 0.34), transparent: true, opacity: 0.72 });
        materials.add(edgeMaterial);
        slab.add(new THREE.LineSegments(edgeGeometry, edgeMaterial));

        const texture = createLabelTexture(block, accent);
        if (texture) {
          texture.anisotropy = Math.min(8, renderer?.capabilities.getMaxAnisotropy() ?? 1);
          textures.add(texture);
          const labelGeometry = new THREE.PlaneGeometry(1.62, 0.72);
          geometries.add(labelGeometry);
          const labelMaterial = new THREE.MeshBasicMaterial({
            map: texture,
            transparent: true,
            depthWrite: false,
            toneMapped: false,
          });
          materials.add(labelMaterial);
          const label = new THREE.Mesh(labelGeometry, labelMaterial);
          label.position.z = SLAB_DEPTH / 2 + 0.065;
          label.renderOrder = 2;
          slab.add(label);
        }
      });

      const bounds = new THREE.Box3().setFromPoints(positions);
      bounds.expandByVector(new THREE.Vector3(SLAB_WIDTH / 2 + 0.5, SLAB_HEIGHT / 2 + 0.5, 0.8));
      const center = bounds.getCenter(new THREE.Vector3());
      const size = bounds.getSize(new THREE.Vector3());
      flowGroup.position.sub(center);
      flowGroup.rotation.set(targetRotationX, targetRotationY, 0);

      const resize = () => {
        if (!renderer) return;
        const parent = canvas.parentElement;
        if (!parent) return;
        const width = Math.max(parent.clientWidth, 1);
        const height = Math.max(parent.clientHeight, 1);
        const priorZoomRatio = baseZoom > 0 ? targetZoom / baseZoom : 1;
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height, false);

        const halfHeight = Math.max(size.y / 2 + 0.9, (size.x / 2 + 0.9) / camera.aspect);
        baseZoom = halfHeight / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) + 1.8;
        minZoom = Math.max(4.5, baseZoom * 0.62);
        maxZoom = Math.max(minZoom + 2, baseZoom * 1.85);
        targetZoom = THREE.MathUtils.clamp(baseZoom * priorZoomRatio, minZoom, maxZoom);
        camera.position.set(0, 0, targetZoom);
        camera.lookAt(0, 0, 0);
        render();
      };

      resizeObserver = new ResizeObserver(resize);
      if (canvas.parentElement) resizeObserver.observe(canvas.parentElement);
      resize();

      if (!reducedMotion) {
        canvas.style.cursor = 'grab';
        canvas.addEventListener('pointerdown', onPointerDown);
        canvas.addEventListener('pointermove', onPointerMove);
        canvas.addEventListener('pointerup', endPointerDrag);
        canvas.addEventListener('pointercancel', endPointerDrag);
        canvas.addEventListener('wheel', onWheel, { passive: false });

        let previousTime = performance.now();
        const animate = (time: number) => {
          const deltaSeconds = Math.min((time - previousTime) / 1000, 0.05);
          previousTime = time;
          if (!dragging) targetRotationY += deltaSeconds * 0.075;
          flowGroup.rotation.x = THREE.MathUtils.lerp(flowGroup.rotation.x, targetRotationX, 0.12);
          flowGroup.rotation.y = THREE.MathUtils.lerp(flowGroup.rotation.y, targetRotationY, 0.1);
          camera.position.z = THREE.MathUtils.lerp(camera.position.z, targetZoom, 0.13);
          render();
          animationFrame = requestAnimationFrame(animate);
        };
        animationFrame = requestAnimationFrame(animate);
      } else {
        canvas.style.cursor = 'default';
        render();
      }
    } catch (error) {
      console.error('ArchDiagram3D initialization failed', error);
      teardown();
      if (!cancelled) setWebglFailed(true);
    }

    return () => {
      cancelled = true;
      teardown();
    };
  }, [blocks, slug]);

  if (blocks.length === 0) return null;
  if (webglFailed) return <ArchDiagram slug={slug} />;

  return (
    <div
      data-arch-diagram="3d"
      className="relative h-[360px] w-full overflow-hidden rounded-xl border border-[#262626] bg-[#0A0A0A]"
    >
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={`Interactive 3D flow diagram for ${slug}`}
        className="h-full w-full touch-none"
      />
      <div className="pointer-events-none absolute bottom-3 right-3 rounded-full border border-white/10 bg-black/55 px-2.5 py-1 text-[10px] text-[#737373] backdrop-blur">
        3D architecture flow
      </div>
    </div>
  );
}
