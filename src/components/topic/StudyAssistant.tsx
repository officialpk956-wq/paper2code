'use client';

import Link from 'next/link';
import { useState, useEffect } from 'react';
import { CheckCircle2, Bookmark, StickyNote, ChevronRight, Target, BookOpen, Lightbulb } from 'lucide-react';
import type { TopicData } from '@/data/topics/types';
import { isCompleted, setCompleted, markInProgress } from '@/lib/progress';
import { load, save } from '@/lib/persistence';
import { resolveTopicByTitle } from '@/data/topics';

interface StudyAssistantProps {
  data: TopicData;
}

interface ActionState {
  completed: boolean;
  bookmarked: boolean;
  noted: boolean;
}

export function StudyAssistant({ data }: StudyAssistantProps) {
  const [loaded, setLoaded] = useState(false);
  const [actions, setActions] = useState<ActionState>({ completed: false, bookmarked: false, noted: false });
  const [notes, setNotes] = useState('');
  const [notesSaved, setNotesSaved] = useState(false);

  useEffect(() => {
    const slug = data.meta.slug;
    const isComp = isCompleted('topic', slug);
    const bookmarks = load<string[]>('p2c:bookmarks') ?? [];
    const savedNotes = load<string>(`p2c:topic-notes:${slug}`) ?? '';

    setActions({
      completed: isComp,
      bookmarked: bookmarks.includes(slug),
      noted: savedNotes.length > 0
    });
    setNotes(savedNotes);
    setLoaded(true);

    markInProgress('topic', slug);
  }, [data.meta.slug]);

  const toggleCompleted = () => {
    if (!loaded) return;
    const newVal = !actions.completed;
    setActions(p => ({ ...p, completed: newVal }));
    setCompleted('topic', data.meta.slug, newVal);
  };

  const toggleBookmark = () => {
    if (!loaded) return;
    const newVal = !actions.bookmarked;
    setActions(p => ({ ...p, bookmarked: newVal }));
    const slug = data.meta.slug;
    const bookmarks = load<string[]>('p2c:bookmarks') ?? [];
    if (newVal && !bookmarks.includes(slug)) {
      save('p2c:bookmarks', [...bookmarks, slug]);
    } else if (!newVal && bookmarks.includes(slug)) {
      save('p2c:bookmarks', bookmarks.filter(b => b !== slug));
    }
  };

  const toggleNoted = () => {
    setActions(p => ({ ...p, noted: !p.noted }));
  };

  const saveNotes = () => {
    save(`p2c:topic-notes:${data.meta.slug}`, notes);
    setNotesSaved(true);
    setTimeout(() => setNotesSaved(false), 2000);
    if (notes.trim().length > 0) {
      setActions(p => ({ ...p, noted: true }));
    } else {
      setActions(p => ({ ...p, noted: false }));
    }
  };

  return (
    <aside
      aria-label="Study assistant"
      className="flex flex-col gap-4"
    >
      {/* Prerequisites */}
      <section
        className="rounded-2xl p-4"
        style={{ background: 'var(--bg-surface)', border: '1px solid rgba(255,255,255,0.06)' }}
        aria-labelledby="prereq-heading"
      >
        <div className="flex items-center gap-2 mb-3">
          <Target size={13} style={{ color: 'var(--accent-primary)' }} aria-hidden="true" />
          <h2 id="prereq-heading" className="text-[12px] font-black" style={{ color: 'var(--color-text-primary)' }}>
            Prerequisites
          </h2>
        </div>
        <ul className="flex flex-col gap-1.5" aria-label="Required prerequisites">
          {data.prerequisites.map(prereq => {
            const resolvedPath = resolveTopicByTitle(prereq);
            const content = (
              <>
                <span className="w-1 h-1 rounded-full flex-none" style={{ background: 'var(--accent-primary)' }} aria-hidden="true" />
                {prereq}
              </>
            );

            return (
              <li key={prereq} className="flex items-center text-[11px]" style={{ color: 'var(--color-text-secondary)' }}>
                {resolvedPath ? (
                  <Link href={resolvedPath} className="flex items-center gap-2 hover:text-[--accent-primary] transition-colors">
                    {content}
                  </Link>
                ) : (
                  <span className="flex items-center gap-2">
                    {content}
                  </span>
                )}
              </li>
            );
          })}
        </ul>
      </section>

      {/* Related Topics */}
      <section
        className="rounded-2xl p-4"
        style={{ background: 'var(--bg-surface)', border: '1px solid rgba(255,255,255,0.06)' }}
        aria-labelledby="related-heading"
      >
        <div className="flex items-center gap-2 mb-3">
          <BookOpen size={13} style={{ color: 'var(--accent-cyan)' }} aria-hidden="true" />
          <h2 id="related-heading" className="text-[12px] font-black" style={{ color: 'var(--color-text-primary)' }}>
            Related Topics
          </h2>
        </div>
        <ul className="flex flex-col gap-1" aria-label="Related topics to explore">
          {data.relatedTopics.map(topic => (
            <li key={topic.slug}>
              <Link
                href={`/learn/${topic.domainSlug}/${topic.slug}`}
                className="flex items-center justify-between py-1.5 px-2 rounded-lg text-[11px] transition-colors hover:opacity-80 focus-visible:outline-none focus-visible:ring-1 group"
                style={{ color: 'var(--color-text-secondary)' }}
                onMouseEnter={e => (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.04)'}
                onMouseLeave={e => (e.currentTarget as HTMLElement).style.background = 'transparent'}
              >
                <span className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full flex-none" style={{ background: topic.color }} aria-hidden="true" />
                  {topic.title}
                </span>
                <ChevronRight size={10} className="opacity-0 group-hover:opacity-100 transition-opacity" aria-hidden="true" />
              </Link>
            </li>
          ))}
        </ul>
      </section>

      {/* Quick Revision */}
      <section
        className="rounded-2xl p-4"
        style={{ background: 'var(--bg-surface)', border: '1px solid rgba(255,255,255,0.06)' }}
        aria-labelledby="revision-heading"
      >
        <div className="flex items-center gap-2 mb-3">
          <Lightbulb size={13} style={{ color: 'var(--accent-amber)' }} aria-hidden="true" />
          <h2 id="revision-heading" className="text-[12px] font-black" style={{ color: 'var(--color-text-primary)' }}>
            Quick Revision
          </h2>
        </div>
        <ol className="flex flex-col gap-2" aria-label="Key revision points">
          {data.quickRevision.map((point, i) => (
            <li key={i} className="flex items-start gap-2 text-[11px] leading-relaxed" style={{ color: 'var(--color-text-secondary)' }}>
              <span
                className="flex-none w-4 h-4 rounded-full flex items-center justify-center text-[9px] font-black mt-0.5"
                style={{ background: 'rgba(245,158,11,0.15)', color: 'var(--accent-amber)' }}
                aria-hidden="true"
              >
                {i + 1}
              </span>
              {point}
            </li>
          ))}
        </ol>
      </section>

      {/* Learning Actions */}
      <section
        className="rounded-2xl p-4"
        style={{ background: 'var(--bg-surface)', border: '1px solid rgba(255,255,255,0.06)' }}
        aria-labelledby="actions-heading"
      >
        <h2 id="actions-heading" className="text-[12px] font-black mb-3" style={{ color: 'var(--color-text-primary)' }}>
          Learning Actions
        </h2>
        <div className="flex flex-col gap-2">
          <button
            onClick={toggleCompleted}
            className="flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-[12px] font-semibold transition-all focus-visible:outline-none focus-visible:ring-2"
            style={{
              background: actions.completed ? 'rgba(16,185,129,0.14)' : 'var(--bg-hover)',
              border: `1px solid ${actions.completed ? 'rgba(16,185,129,0.35)' : 'rgba(255,255,255,0.07)'}`,
              color: actions.completed ? '#10B981' : 'var(--color-text-secondary)',
              opacity: loaded ? 1 : 0.5,
            }}
            aria-pressed={actions.completed}
          >
            <CheckCircle2 size={13} aria-hidden="true" />
            {actions.completed ? 'Marked Complete ✓' : 'Mark Complete'}
          </button>

          <button
            onClick={toggleBookmark}
            className="flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-[12px] font-semibold transition-all focus-visible:outline-none focus-visible:ring-2"
            style={{
              background: actions.bookmarked ? 'rgba(124,58,237,0.14)' : 'var(--bg-hover)',
              border: `1px solid ${actions.bookmarked ? 'rgba(124,58,237,0.35)' : 'rgba(255,255,255,0.07)'}`,
              color: actions.bookmarked ? 'var(--accent-primary)' : 'var(--color-text-secondary)',
              opacity: loaded ? 1 : 0.5,
            }}
            aria-pressed={actions.bookmarked}
          >
            <Bookmark size={13} aria-hidden="true" />
            {actions.bookmarked ? 'Bookmarked ✓' : 'Add Bookmark'}
          </button>

          <button
            onClick={toggleNoted}
            className="flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-[12px] font-semibold transition-all focus-visible:outline-none focus-visible:ring-2"
            style={{
              background: actions.noted ? 'rgba(6,182,212,0.14)' : 'var(--bg-hover)',
              border: `1px solid ${actions.noted ? 'rgba(6,182,212,0.35)' : 'rgba(255,255,255,0.07)'}`,
              color: actions.noted ? 'var(--accent-cyan)' : 'var(--color-text-secondary)',
              opacity: loaded ? 1 : 0.5,
            }}
            aria-pressed={actions.noted}
          >
            <StickyNote size={13} aria-hidden="true" />
            {actions.noted ? 'Notes' : 'Add Notes'}
          </button>

          {actions.noted && (
            <div className="mt-2 flex flex-col gap-2">
              <textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Type your notes here..."
                className="w-full h-24 p-2 rounded-lg text-[12px] bg-[rgba(255,255,255,0.02)] border border-[rgba(255,255,255,0.1)] text-[--color-text-primary] placeholder:text-[--color-text-tertiary] focus:outline-none focus:border-[--accent-cyan]"
              />
              <button
                onClick={saveNotes}
                className="w-full py-1.5 rounded-lg text-[11px] font-bold text-white transition-all bg-[--accent-cyan] hover:bg-cyan-600"
              >
                {notesSaved ? 'Notes Saved ✓' : 'Save Notes'}
              </button>
            </div>
          )}
        </div>
      </section>
    </aside>
  );
}
