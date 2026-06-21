'use client';

import { X, CheckCircle, AlertCircle, Info, LogIn, LogOut } from 'lucide-react';
import { useState, useEffect } from 'react';

export type NotificationType = 'success' | 'error' | 'info' | 'join' | 'leave';

interface Toast {
  id: string;
  type: NotificationType;
  title: string;
  message?: string;
  duration?: number;
  avatar?: string;
}

const notificationConfig = {
  success: {
    icon: CheckCircle,
    color: 'text-green-400',
    bgColor: 'bg-green-400/10',
    borderColor: 'border-green-400/30',
  },
  error: {
    icon: AlertCircle,
    color: 'text-red-400',
    bgColor: 'bg-red-400/10',
    borderColor: 'border-red-400/30',
  },
  info: {
    icon: Info,
    color: 'text-blue-400',
    bgColor: 'bg-blue-400/10',
    borderColor: 'border-blue-400/30',
  },
  join: {
    icon: LogIn,
    color: 'text-green-400',
    bgColor: 'bg-green-400/10',
    borderColor: 'border-green-400/30',
  },
  leave: {
    icon: LogOut,
    color: 'text-gray-400',
    bgColor: 'bg-gray-400/10',
    borderColor: 'border-gray-400/30',
  },
};

const sampleToasts: Toast[] = [
  {
    id: '1',
    type: 'join',
    title: 'Sarah Chen joined the workspace',
    duration: 5000,
    avatar: '👩',
  },
  {
    id: '2',
    type: 'success',
    title: 'Changes saved successfully',
    message: 'Your modifications have been synced with all collaborators',
    duration: 4000,
  },
  {
    id: '3',
    type: 'info',
    title: 'New comment mention',
    message: 'Alex Kumar mentioned you in a comment',
    duration: 6000,
    avatar: '👨',
  },
];

interface NotificationToastProps {
  toasts?: Toast[];
  position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left';
  maxToasts?: number;
}

export function NotificationToast({
  toasts = sampleToasts,
  position = 'top-right',
  maxToasts = 3,
}: NotificationToastProps) {
  const [visibleToasts, setVisibleToasts] = useState<Toast[]>(toasts.slice(0, maxToasts));

  useEffect(() => {
    const timers = visibleToasts.map((toast) => {
      if (!toast.duration) return null;
      return setTimeout(() => {
        setVisibleToasts((prev) => prev.filter((t) => t.id !== toast.id));
      }, toast.duration);
    });

    return () => {
      timers.forEach((timer) => {
        if (timer) clearTimeout(timer);
      });
    };
  }, [visibleToasts]);

  const handleDismiss = (id: string) => {
    setVisibleToasts((prev) => prev.filter((toast) => toast.id !== id));
  };

  const positionClasses = {
    'top-right': 'top-4 right-4',
    'top-left': 'top-4 left-4',
    'bottom-right': 'bottom-4 right-4',
    'bottom-left': 'bottom-4 left-4',
  };

  return (
    <div className={`fixed ${positionClasses[position]} pointer-events-none`}>
      <div className="flex flex-col gap-3">
        {visibleToasts.map((toast) => {
          const config = notificationConfig[toast.type];
          const IconComponent = config.icon;

          return (
            <div
              key={toast.id}
              className={`pointer-events-auto flex items-start gap-3 px-4 py-3 rounded-lg border ${config.bgColor} ${config.borderColor} backdrop-blur-sm shadow-xl animate-slide-in-top`}
              style={{
                background: 'rgba(15, 15, 23, 0.9)',
                border: '1px solid ' + config.borderColor.replace('border-', ''),
              }}
            >
              {/* Avatar if present */}
              {toast.avatar && (
                <div className="text-lg flex-shrink-0">{toast.avatar}</div>
              )}

              {/* Icon if no avatar */}
              {!toast.avatar && (
                <IconComponent size={18} className={`flex-shrink-0 ${config.color}`} />
              )}

              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="text-sm font-semibold text-[--color-text-primary]">
                  {toast.title}
                </div>
                {toast.message && (
                  <div className="text-xs text-[--color-text-secondary] mt-0.5">
                    {toast.message}
                  </div>
                )}
              </div>

              {/* Close Button */}
              <button
                onClick={() => handleDismiss(toast.id)}
                className="flex-shrink-0 text-[--color-text-tertiary] hover:text-[--color-text-primary] transition-colors"
              >
                <X size={16} />
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
