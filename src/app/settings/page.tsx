'use client';

import { Settings, User, Bell, Palette, Shield, Code2 } from 'lucide-react';

const sections = [
  {
    id: 'profile',
    icon: User,
    label: 'Profile',
    description: 'Manage your display name, avatar, and account information.',
  },
  {
    id: 'appearance',
    icon: Palette,
    label: 'Appearance',
    description: 'Customize theme, font size, and editor preferences.',
  },
  {
    id: 'notifications',
    icon: Bell,
    label: 'Notifications',
    description: 'Configure alerts for progress milestones and collaborator activity.',
  },
  {
    id: 'editor',
    icon: Code2,
    label: 'Editor',
    description: 'Set default language, key bindings, and auto-save behavior.',
  },
  {
    id: 'privacy',
    icon: Shield,
    label: 'Privacy & Security',
    description: 'Control data sharing, session management, and export options.',
  },
];

export default function SettingsPage() {
  return (
    <div className="flex flex-col h-[calc(100vh-56px)] bg-[--bg-body] overflow-y-auto">
      {/* Header */}
      <div className="px-8 py-6 border-b border-[--color-border] bg-[--bg-panel]">
        <div className="flex items-center gap-3">
          <Settings size={20} className="text-[--accent-primary]" />
          <h1 className="text-2xl font-bold text-[--color-text-primary]">Settings</h1>
        </div>
        <p className="text-sm text-[--color-text-secondary] mt-1">
          Manage your account, appearance, and preferences.
        </p>
      </div>

      {/* Settings list */}
      <div className="max-w-2xl mx-auto w-full px-8 py-6 space-y-3">
        {sections.map((section) => {
          const Icon = section.icon;
          return (
            <button
              key={section.id}
              className="w-full flex items-start gap-4 p-4 rounded-lg border border-[--color-border] bg-[--bg-panel] hover:border-[--accent-primary]/50 hover:bg-[--bg-hover] transition-colors text-left"
            >
              <div className="w-9 h-9 rounded-lg bg-[--bg-body] border border-[--color-border] flex items-center justify-center flex-shrink-0">
                <Icon size={16} className="text-[--accent-primary]" />
              </div>
              <div>
                <div className="text-sm font-semibold text-[--color-text-primary]">{section.label}</div>
                <div className="text-xs text-[--color-text-secondary] mt-0.5">{section.description}</div>
              </div>
            </button>
          );
        })}

        <p className="text-xs text-[--color-text-tertiary] text-center pt-4">
          Full settings configuration coming soon.
        </p>
      </div>
    </div>
  );
}
