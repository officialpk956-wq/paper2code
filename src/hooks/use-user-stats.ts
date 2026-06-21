import { useState, useEffect } from 'react';
import { getAllCompleted, getCompletedCount, getCompletionRate, getStreakDays } from '@/lib/progress';

export interface UserStats {
  loaded: boolean;
  xp: number;
  level: number;
  streak: number;
  completedLessons: number;
  overallProgress: number;
}

export function useUserStats(): UserStats {
  const [stats, setStats] = useState<UserStats>({
    loaded: false,
    xp: 0,
    level: 1,
    streak: 0,
    completedLessons: 0,
    overallProgress: 0,
  });

  useEffect(() => {
    // Only run on the client
    const allCompletedCount = getAllCompleted().length;
    const xp = allCompletedCount * 50;
    const level = Math.floor(xp / 200) + 1;
    const streak = getStreakDays();
    const completedLessons = getCompletedCount('topic');
    const overallProgress = getCompletionRate();

    setStats({
      loaded: true,
      xp,
      level,
      streak,
      completedLessons,
      overallProgress,
    });
  }, []);

  return stats;
}
