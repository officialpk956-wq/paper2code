import { NextResponse } from 'next/server';
import { RECOMMENDATIONS, RECENTLY_ADDED, CONTINUE_LEARNING } from '@/data/learn/recommendations';
import type { Recommendation, RecentlyAddedItem, ContinueLearningData } from '@/data/learn/recommendations';

export interface RecommendationsResponse {
  continueLearning: ContinueLearningData;
  recommendations: Recommendation[];
  recentlyAdded: RecentlyAddedItem[];
}

export async function GET(): Promise<NextResponse<RecommendationsResponse>> {
  return NextResponse.json({
    continueLearning: CONTINUE_LEARNING,
    recommendations: RECOMMENDATIONS,
    recentlyAdded: RECENTLY_ADDED,
  });
}
