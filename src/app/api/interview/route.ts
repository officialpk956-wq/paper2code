import { NextResponse } from 'next/server';
import { INTERVIEW_QUESTIONS } from '@/data/interview-questions';

export async function GET() {
  return NextResponse.json(INTERVIEW_QUESTIONS);
}
