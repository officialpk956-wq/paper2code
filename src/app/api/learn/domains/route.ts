import { NextResponse } from 'next/server';
import { DOMAINS } from '@/data/learn/domains';
import type { Domain } from '@/data/learn/domains';

export interface DomainsResponse {
  domains: Domain[];
  total: number;
}

export async function GET(): Promise<NextResponse<DomainsResponse>> {
  return NextResponse.json({
    domains: DOMAINS,
    total: DOMAINS.length,
  });
}
