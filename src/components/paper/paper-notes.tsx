'use client';

import { MessageSquare, Send, Lightbulb } from 'lucide-react';
import { useState } from 'react';

interface Note {
  id: string;
  text: string;
  line: number;
  timestamp: string;
}

const sampleNotes: Note[] = [
  {
    id: '1',
    text: 'Multi-head attention allows parallel computation across sequence',
    line: 12,
    timestamp: '2 days ago',
  },
  {
    id: '2',
    text: 'Compare with recurrent approaches - why transformers are better',
    line: 28,
    timestamp: '1 day ago',
  },
  {
    id: '3',
    text: 'Implement sparse attention for sequence lengths > 2048',
    line: 45,
    timestamp: 'Today',
  },
];

export function PaperNotes() {
  const [notes, setNotes] = useState<Note[]>(sampleNotes);
  const [newNote, setNewNote] = useState('');
  const [activeTab, setActiveTab] = useState<'notes' | 'highlights'>('notes');

  const addNote = () => {
    if (newNote.trim()) {
      setNotes([
        ...notes,
        {
          id: Date.now().toString(),
          text: newNote,
          line: 0,
          timestamp: 'just now',
        },
      ]);
      setNewNote('');
    }
  };

  const deleteNote = (id: string) => {
    setNotes(notes.filter((n) => n.id !== id));
  };

  return (
    <div className="flex flex-col h-full bg-[--bg-panel] border-l border-[--color-border]">
      {/* Header */}
      <div className="p-4 border-b border-[--color-border]">
        <h3 className="text-sm font-semibold text-[--color-text-primary] mb-3">
          Study Notes
        </h3>
        <div className="flex gap-2 bg-[--bg-body] rounded p-1">
          <button
            onClick={() => setActiveTab('notes')}
            className={`flex-1 px-2 py-1.5 rounded text-xs font-medium transition-colors ${
              activeTab === 'notes'
                ? 'bg-[--accent-primary] text-white'
                : 'text-[--color-text-secondary] hover:text-[--color-text-primary]'
            }`}
          >
            <MessageSquare size={12} className="inline mr-1" />
            Notes
          </button>
          <button
            onClick={() => setActiveTab('highlights')}
            className={`flex-1 px-2 py-1.5 rounded text-xs font-medium transition-colors ${
              activeTab === 'highlights'
                ? 'bg-[--accent-primary] text-white'
                : 'text-[--color-text-secondary] hover:text-[--color-text-primary]'
            }`}
          >
            <Lightbulb size={12} className="inline mr-1" />
            Highlights
          </button>
        </div>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {notes.map((note) => (
          <div
            key={note.id}
            className="group p-3 bg-[--bg-body] rounded border border-[--color-border] hover:border-[--accent-primary]/50 transition-colors"
          >
            <div className="flex items-start justify-between gap-2 mb-2">
              <div className="text-xs text-[--color-text-tertiary] font-medium">
                Line {note.line}
              </div>
              <button
                onClick={() => deleteNote(note.id)}
                className="opacity-0 group-hover:opacity-100 text-xs text-[--color-text-tertiary] hover:text-red-400 transition-all"
              >
                ✕
              </button>
            </div>
            <p className="text-xs text-[--color-text-primary] leading-relaxed mb-2">
              {note.text}
            </p>
            <div className="text-xs text-[--color-text-tertiary]">
              {note.timestamp}
            </div>
          </div>
        ))}

        {notes.length === 0 && (
          <div className="flex items-center justify-center h-32 text-center">
            <div className="text-xs text-[--color-text-tertiary]">
              <MessageSquare size={20} className="mx-auto mb-2 opacity-50" />
              No notes yet. Add one below!
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="p-4 border-t border-[--color-border] bg-[--bg-body] space-y-3">
        <textarea
          value={newNote}
          onChange={(e) => setNewNote(e.target.value)}
          placeholder="Add a note or insight..."
          className="w-full px-3 py-2 bg-[--bg-surface] border border-[--color-border] rounded text-xs text-[--color-text-primary]
                   placeholder-[--color-text-tertiary] resize-none focus:border-[--accent-primary] focus:outline-none transition-colors"
          rows={3}
        />
        <button
          onClick={addNote}
          disabled={!newNote.trim()}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-[--accent-primary] hover:opacity-90
                   disabled:opacity-50 disabled:cursor-not-allowed text-xs font-medium text-white rounded transition-opacity"
        >
          <Send size={12} />
          Add Note
        </button>
      </div>
    </div>
  );
}
