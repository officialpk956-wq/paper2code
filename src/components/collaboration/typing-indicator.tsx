'use client';

import { useState, useEffect } from 'react';

interface TypingUser {
  id: string;
  name: string;
  avatar: string;
  location: string;
}

const typingUsers: TypingUser[] = [
  {
    id: '2',
    name: 'Sarah Chen',
    avatar: '👩',
    location: 'Canvas - Layer 3',
  },
  {
    id: '3',
    name: 'Alex Kumar',
    avatar: '👨',
    location: 'Notes - Attention Mechanism',
  },
];

interface TypingIndicatorProps {
  users?: TypingUser[];
}

export function TypingIndicator({ users = typingUsers }: TypingIndicatorProps) {
  const [dots, setDots] = useState<number>(1);

  useEffect(() => {
    if (users.length === 0) return;

    const interval = setInterval(() => {
      setDots((prev) => (prev === 3 ? 1 : prev + 1));
    }, 300);

    return () => clearInterval(interval);
  }, [users.length]);

  if (users.length === 0) return null;

  const renderDots = () => '•'.repeat(dots).padEnd(3, '·');

  return (
    <div className="space-y-2">
      {users.map((user) => (
        <div
          key={user.id}
          className="flex items-center gap-2 px-3 py-2 rounded-lg bg-[--accent-cyan]/10 border border-[--accent-cyan]/30 animate-pulse"
        >
          <div className="text-lg flex-shrink-0">{user.avatar}</div>

          <div className="flex-1 min-w-0">
            <div className="text-xs font-medium text-[--accent-cyan]">
              {user.name} is typing
            </div>
            <div className="text-xs text-[--color-text-tertiary] truncate">
              {user.location}
            </div>
          </div>

          {/* Animated dots */}
          <div className="text-sm font-bold text-[--accent-cyan] tracking-widest min-w-[24px]">
            {renderDots()}
          </div>
        </div>
      ))}
    </div>
  );
}
