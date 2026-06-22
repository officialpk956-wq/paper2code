import { NextResponse } from 'next/server';
import { LEARNING_PATHS } from '@/data/learn/paths';
import type { LearningPath } from '@/data/learn/paths';

export interface PathsResponse {
  paths: LearningPath[];
  total: number;
}

export async function GET(): Promise<NextResponse<PathsResponse>> {
  return NextResponse.json({
    paths: LEARNING_PATHS,
    total: LEARNING_PATHS.length,
  });
}
