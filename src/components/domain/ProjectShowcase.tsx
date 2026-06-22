'use client';

import Link from 'next/link';
import { motion, useMotionValue, useSpring } from 'framer-motion';
import { Clock, ArrowUpRight, Code2 } from 'lucide-react';
import type { DomainProject } from '@/data/domains/types';

interface ProjectShowcaseProps {
  projects: DomainProject[];
  domainSlug: string;
}

const DIFFICULTY_LABEL = {
  beginner:     { className: 'badge-easy',   label: 'Beginner' },
  intermediate: { className: 'badge-medium', label: 'Intermediate' },
  advanced:     { className: 'badge-hard',   label: 'Advanced' },
};

function hexToRgb(hex: string): string {
  const h = hex.replace('#', '');
  return `${parseInt(h.slice(0,2),16)},${parseInt(h.slice(2,4),16)},${parseInt(h.slice(4,6),16)}`;
}

function ProjectCard({ project, index }: { project: DomainProject; index: number }) {
  const diff = DIFFICULTY_LABEL[project.difficulty];
  const rgb = hexToRgb(project.color);
  const rotateX = useMotionValue(0);
  const rotateY = useMotionValue(0);
  const smoothX = useSpring(rotateX, { stiffness: 120, damping: 18 });
  const smoothY = useSpring(rotateY, { stiffness: 120, damping: 18 });

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    rotateX.set(((e.clientY - cy) / rect.height) * -8);
    rotateY.set(((e.clientX - cx) / rect.width) * 8);
  };

  const handleMouseLeave = () => {
    rotateX.set(0);
    rotateY.set(0);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.4, delay: index * 0.08 }}
      style={{
        rotateX: smoothX,
        rotateY: smoothY,
        transformStyle: 'preserve-3d',
        perspective: 800,
      }}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
    >
      <Link
        href={project.href}
        className="block rounded-2xl p-5 transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 h-full"
        style={{
          background: 'var(--bg-surface)',
          border: `1px solid rgba(${rgb},0.18)`,
        }}
        aria-label={`${project.title}: ${project.difficulty}, ${project.estimatedHours} hours`}
        onMouseEnter={e => {
          const el = e.currentTarget as HTMLElement;
          el.style.borderColor = `rgba(${rgb},0.45)`;
          el.style.background = 'var(--bg-hover)';
        }}
        onMouseLeave={e => {
          const el = e.currentTarget as HTMLElement;
          el.style.borderColor = `rgba(${rgb},0.18)`;
          el.style.background = 'var(--bg-surface)';
        }}
        onFocus={e => {
          (e.currentTarget as HTMLElement).style.borderColor = `rgba(${rgb},0.45)`;
        }}
        onBlur={e => {
          (e.currentTarget as HTMLElement).style.borderColor = `rgba(${rgb},0.18)`;
        }}
      >
        {/* Icon + external link arrow */}
        <div className="flex items-start justify-between mb-3">
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center"
            style={{ background: `rgba(${rgb},0.14)` }}
            aria-hidden="true"
          >
            <Code2 size={18} style={{ color: project.color }} />
          </div>
          <ArrowUpRight size={14} style={{ color: 'var(--color-text-muted)' }} aria-hidden="true" />
        </div>

        {/* Title */}
        <div className="text-[14px] font-black leading-snug mb-1.5" style={{ color: 'var(--color-text-primary)' }}>
          {project.title}
        </div>

        {/* Description */}
        <p className="text-[11px] leading-relaxed line-clamp-2 mb-4" style={{ color: 'var(--color-text-secondary)' }}>
          {project.description}
        </p>

        {/* Skills */}
        <div className="flex flex-wrap gap-1 mb-4">
          {project.skills.map(skill => (
            <span
              key={skill}
              className="text-[10px] px-2 py-0.5 rounded-md"
              style={{ background: `rgba(${rgb},0.12)`, color: project.color }}
            >
              {skill}
            </span>
          ))}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between">
          <span className={diff.className}>{diff.label}</span>
          <span className="flex items-center gap-1 text-[10px]" style={{ color: 'var(--color-text-muted)' }}>
            <Clock size={9} aria-hidden="true" /> {project.estimatedHours}h
          </span>
        </div>
      </Link>
    </motion.div>
  );
}

export function ProjectShowcase({ projects, domainSlug }: ProjectShowcaseProps) {
  return (
    <section aria-labelledby="projects-heading" className="mb-10">
      <div className="flex items-center justify-between mb-5">
        <div>
          <h2 id="projects-heading" className="text-lg font-black" style={{ color: 'var(--color-text-primary)' }}>
            Projects
          </h2>
          <p className="text-xs mt-0.5" style={{ color: 'var(--color-text-secondary)' }}>
            Build real systems using this domain — paper-to-code implementations
          </p>
        </div>
        <Link
          href={`/learn/${domainSlug}/projects`}
          className="text-xs font-semibold transition-opacity hover:opacity-80"
          style={{ color: 'var(--accent-primary)' }}
        >
          View all →
        </Link>
      </div>

      {projects.length === 0 ? (
        <div
          className="rounded-2xl px-6 py-10 text-center"
          style={{ background: 'var(--bg-surface)', border: '1px solid rgba(255,255,255,0.06)' }}
        >
          <p className="text-sm" style={{ color: 'var(--color-text-tertiary)' }}>
            No projects available yet for this domain.
          </p>
        </div>
      ) : (
        <div
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4"
          role="list"
          aria-label="Domain projects"
        >
          {projects.map((project, i) => (
            <div key={project.id} role="listitem">
              <ProjectCard project={project} index={i} />
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
