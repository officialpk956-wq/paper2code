'use client';

import Link from 'next/link';
import { useState, useEffect, useRef, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import { Upload, ChevronDown, ChevronRight } from 'lucide-react';
import { apiGet, apiPostForm, isLoggedIn } from '@/lib/api';
import { PAPERS, PaperEntry } from '@/data/content/papers';
import { findArchLoose, dojoSlugFor, LIBRARY_TO_WORKSPACE_ID } from '@/lib/crosslinks';

type Paper = {
  id: string;
  title: string;
  authors: string;
  topics?: string[];
  status: 'Ready' | 'Processing';
  color?: string;
  created_at?: string;
  visibility?: 'public' | 'private';
};

const getDifficultyColor = (diff: string) => {
  switch(diff) {
    case 'Beginner': return '#4ADE80';
    case 'Intermediate': return '#FACC15';
    case 'Advanced': return '#F87171';
    case 'Expert': return '#A78BFA';
    default: return '#525252';
  }
};

function PaperRow({ paper }: { paper: PaperEntry }) {
  const [expanded, setExpanded] = useState(false);
  const workspaceId = LIBRARY_TO_WORKSPACE_ID[paper.slug];
  const hasWorkspace = !!workspaceId;

  return (
    <div className="border border-[#262626] rounded-xl bg-[#111111] mb-3 overflow-hidden">
      <div 
        className="p-4 flex gap-4 cursor-pointer hover:bg-[#111111] transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="w-8 text-[#A3A3A3] font-mono text-[12px] font-medium pt-0.5">#{paper.rank}</div>
        <div className="flex-1">
          <div className="flex items-start justify-between">
            {hasWorkspace ? (
              <Link href={`/papers/${workspaceId}`} className="text-[15px] font-bold text-white hover:text-[#A78BFA] transition-colors" onClick={(e) => e.stopPropagation()}>
                {paper.title}
              </Link>
            ) : (
              <div className="text-[15px] font-bold text-white">{paper.title}</div>
            )}
            <div className="flex items-center gap-2 flex-shrink-0 ml-4">
              {paper.year && <span className="text-[11px] text-[#525252]">{paper.year}</span>}
              {paper.difficulty && (
                <span className="text-[10px] px-2 py-0.5 rounded-md font-medium" style={{ backgroundColor: `${getDifficultyColor(paper.difficulty)}20`, color: getDifficultyColor(paper.difficulty) }}>
                  {paper.difficulty}
                </span>
              )}
              {expanded ? <ChevronDown size={14} className="text-[#A3A3A3]" /> : <ChevronRight size={14} className="text-[#A3A3A3]" />}
            </div>
          </div>
          {paper.authors && <div className="text-[12px] text-[#A3A3A3] mt-1">{paper.authors}</div>}
          {paper.whyImportant && (
            <div className="text-[12px] text-[#A3A3A3] mt-2 line-clamp-2 leading-relaxed">
              {paper.whyImportant}
            </div>
          )}
        </div>
      </div>
      
      {expanded && (
        <div className="border-t border-[#262626] bg-transparent p-4 text-[12px] space-y-3">
          {paper.conceptsIntroduced && (
            <div>
              <span className="text-[#A78BFA] font-medium">Concepts:</span> <span className="text-[#A3A3A3]">{paper.conceptsIntroduced}</span>
            </div>
          )}
          {paper.architecturesIntroduced && (
            <div>
              <span className="text-[#A78BFA] font-medium">Architectures:</span> <span className="text-[#A3A3A3]">{paper.architecturesIntroduced}</span>
            </div>
          )}
          {paper.industryImpact && (
            <div>
              <span className="text-[#A78BFA] font-medium">Industry Impact:</span> <span className="text-[#A3A3A3]">{paper.industryImpact}</span>
            </div>
          )}
          {paper.builtUponBy && (
            <div>
              <span className="text-[#A78BFA] font-medium">Built Upon By:</span> <span className="text-[#A3A3A3]">{paper.builtUponBy}</span>
            </div>
          )}
          {(() => {
            const chips = [];
            if (paper.architecturesIntroduced) {
               const tokens = paper.architecturesIntroduced.split(',');
               const seen = new Set<string>();
               let count = 0;
               for (const t of tokens) {
                 if (count >= 2) break;
                 const a = findArchLoose(t);
                 if (a && !seen.has(a.slug)) {
                   seen.add(a.slug);
                   chips.push(
                     <Link key={a.slug} href={`/architectures/${a.slug}`} className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-[#1A1A1A] border border-[#2E4436] rounded-md text-[11px] text-[#A78BFA] font-medium hover:bg-[#262626]">
                       Architecture: {a.name} →
                     </Link>
                   );
                   count++;
                 }
               }
            }
            const dojoSlug = dojoSlugFor(paper.conceptsIntroduced ?? '');
            if (dojoSlug) {
              chips.push(
                <Link key="dojo" href={`/dojo/${dojoSlug}`} className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-[#1A1A1A] border border-[#2E4436] rounded-md text-[11px] text-[#60A5FA] font-medium hover:bg-[#262626]">
                  Solve related problem →
                </Link>
              );
            }
            return chips.length > 0 ? (
              <div className="flex gap-2 pt-3 border-t border-[#1A1A1A] mt-2">
                {chips}
              </div>
            ) : null;
          })()}
        </div>
      )}
    </div>
  );
}

function PapersContent() {
  const searchParams = useSearchParams();
  const initTab = searchParams.get('tab') === 'library' ? 'Library' : 'Workspace';
  const [activeTab, setActiveTab] = useState<'Library' | 'Workspace'>(initTab);
  const [uploading, setUploading] = useState(false);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const isUserLoggedIn = isLoggedIn();

  useEffect(() => {
    fetchPapers();
  }, []);

  const fetchPapers = async () => {
    setLoading(true);
    setError('');
    try {
      const endpoint = isLoggedIn() ? '/api/papers' : '/api/papers/public';
      const data = await apiGet<Paper[]>(endpoint);
      setPapers(data);
    } catch (err: unknown) {
      setError((err as Error).message || 'Failed to load papers');
    } finally {
      setLoading(false);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!isUserLoggedIn) {
      alert('Please sign in to upload papers.');
      return;
    }

    setUploading(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('title', file.name.replace('.pdf', ''));
      formData.append('visibility', 'private');

      const newPaper = await apiPostForm<Paper>('/api/papers/upload', formData);
      setPapers(prev => [newPaper, ...prev]);
    } catch (err: unknown) {
      setError((err as Error).message || 'Upload failed');
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  // Group papers by section in file order
  const sectionMap = new Map<string, PaperEntry[]>();
  for (const p of PAPERS) {
    if (!sectionMap.has(p.section)) sectionMap.set(p.section, []);
    sectionMap.get(p.section)!.push(p);
  }
  const groupedPapers = Array.from(sectionMap.entries());

  return (
    <div className="flex flex-col overflow-hidden bg-transparent text-white" style={{ height: 'calc(100vh - 56px)' }}>
      {/* HEADER */}
      <div className="h-[68px] bg-[#0A0A0A] border-b border-[#1A1A1A] flex items-center px-8 gap-6 flex-shrink-0">
        <div className="flex items-center gap-4">
          <div className="text-[20px] font-bold text-white">Research Hub</div>
          <div className="flex bg-[#111111] border border-[#262626] rounded-lg p-1">
            <button
              onClick={() => setActiveTab('Library')}
              className={`px-4 py-1.5 text-[12px] font-medium rounded-md transition-colors ${activeTab === 'Library' ? 'bg-[#A78BFA] text-black' : 'text-[#A3A3A3] hover:text-white'}`}
            >
              Golden Library
            </button>
            <button
              onClick={() => setActiveTab('Workspace')}
              className={`px-4 py-1.5 text-[12px] font-medium rounded-md transition-colors ${activeTab === 'Workspace' ? 'bg-[#A78BFA] text-black' : 'text-[#A3A3A3] hover:text-white'}`}
            >
              My Workspace
            </button>
          </div>
        </div>
        <div className="flex-1" />
        <input type="text" placeholder="Search papers..."
          className="w-[240px] bg-[#111111] border border-[#262626] rounded-lg px-3 py-2 text-[13px] text-white placeholder:text-[#525252] focus:outline-none focus:border-[#A78BFA]/40" />
      </div>

      {error && activeTab === 'Workspace' && (
        <div className="bg-red-500/10 border-b border-red-500/20 px-8 py-2 text-xs text-red-500 flex justify-center">
          Could not load data — retrying… ({error})
        </div>
      )}

      {/* BODY */}
      <div className="flex flex-1 overflow-hidden">
        {activeTab === 'Library' ? (
          <div className="flex-1 overflow-y-auto p-8 max-w-4xl mx-auto w-full">
            {groupedPapers.map(([sectionName, sectionPapers]) => (
              <div key={sectionName} className="mb-10">
                <div className="flex items-baseline justify-between border-b border-[#262626] pb-2 mb-4">
                  <h2 className="text-[18px] font-bold text-[#A78BFA]">{sectionName}</h2>
                  <span className="text-[12px] font-medium text-[#525252]">{sectionPapers.length} papers</span>
                </div>
                <div>
                  {sectionPapers.map(p => (
                    <PaperRow key={p.slug} paper={p} />
                  ))}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="flex flex-1 gap-8 p-8 overflow-hidden w-full">
            {/* UPLOAD */}
            <div className="w-[400px] flex-shrink-0">
              <input type="file" accept=".pdf" ref={fileInputRef} className="hidden" onChange={handleFileChange} />
              
              <div onClick={handleUploadClick}
                className="border-2 border-dashed border-[#A78BFA]/35 rounded-2xl bg-[#0A0A0A] min-h-[220px] flex flex-col items-center justify-center gap-4 p-8 cursor-pointer hover:border-[#A78BFA]/60 hover:bg-[#A78BFA]/3 transition-all">
                <div className="w-14 h-14 bg-[#A78BFA]/12 rounded-full flex items-center justify-center">
                  <Upload className="text-[#A78BFA]" size={24} />
                </div>
                <div className="text-[15px] font-semibold text-white">Drop your PDF here</div>
                <div className="text-[12px] text-[#525252] -mt-2">or click to browse — arXiv PDFs supported</div>
                <button className="bg-[#A78BFA] text-black rounded-lg px-5 py-2.5 text-[13px] font-semibold mt-2 hover:brightness-110">
                  Browse Files
                </button>
                <div className="text-[11px] text-[#525252] mt-1">PDF files up to 50MB</div>
              </div>

              {uploading && (
                <div className="bg-[#111111] border border-[#262626] rounded-xl p-4 mt-4">
                  <div className="text-[13px] font-semibold text-white">Uploading...</div>
                  <div className="h-1.5 bg-[#1A1A1A] rounded-full mt-3 overflow-hidden">
                    <div className="h-full bg-[#A78BFA] rounded-full w-[65%] animate-pulse" />
                  </div>
                  <div className="text-[11px] text-[#A78BFA] mt-2">Processing document...</div>
                </div>
              )}
            </div>

            {/* PAPERS GRID */}
            <div className="flex-1 overflow-y-auto">
              <div className="text-[14px] font-semibold text-white mb-4">
                {isUserLoggedIn ? 'Your Papers' : 'Public Papers'}
              </div>
              
              {loading ? (
                <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
                  {[1, 2, 3].map(i => (
                    <div key={i} className="bg-[#111111] border border-[#262626] rounded-xl p-4 h-[140px] animate-pulse">
                      <div className="h-1 bg-[#262626] rounded-full mb-3 w-1/4" />
                      <div className="h-4 bg-[#262626] rounded mb-2" />
                      <div className="h-4 bg-[#262626] rounded mb-4 w-3/4" />
                      <div className="h-3 bg-[#262626] rounded w-1/2" />
                    </div>
                  ))}
                </div>
              ) : papers.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-[300px] border border-dashed border-[#262626] rounded-xl bg-[#111111]">
                  <div className="text-[#525252] text-sm mb-2">No papers yet</div>
                  {!isUserLoggedIn && (
                    <button className="text-[#A78BFA] hover:underline text-sm font-semibold" onClick={() => (window as Window & typeof globalThis & { showAuthModal?: () => void }).showAuthModal?.()}>
                      Sign up to upload your first paper
                    </button>
                  )}
                </div>
              ) : (
                <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
                  {papers.map(p => (
                    <Link key={p.id} href={`/papers/${p.id}`}
                      className="bg-[#111111] border border-[#262626] rounded-xl p-4 cursor-pointer hover:border-[#A78BFA]/35 transition-colors flex flex-col h-full">
                      <div className="h-1 rounded-full mb-3" style={{ backgroundColor: p.color || '#A78BFA' }} />
                      <div className="text-[12px] font-semibold text-white leading-snug line-clamp-2">{p.title}</div>
                      <div className="text-[10px] text-[#525252] mt-1">{p.authors || 'Unknown Author'}</div>
                      <div className="flex gap-2 mt-3 flex-wrap">
                        {(p.topics || []).map(t => (
                          <span key={t} className="bg-[#1A1A1A] text-[#A3A3A3] text-[10px] px-2 py-0.5 rounded-md">{t}</span>
                        ))}
                      </div>
                      <div className="mt-auto pt-4 flex justify-between items-center">
                        <span className={p.status === 'Ready' || p.status.toLowerCase() === 'ready'
                          ? 'text-[10px] font-semibold px-2 py-0.5 rounded-full bg-[#4ADE80]/12 text-[#4ADE80]'
                          : 'text-[10px] font-semibold px-2 py-0.5 rounded-full bg-[#FACC15]/12 text-[#FACC15]'}>
                          {p.status}
                        </span>
                        <span className="text-[#A78BFA] text-[11px] font-semibold hover:underline">Open →</span>
                      </div>
                    </Link>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default function PapersPage() {
  return (
    <Suspense fallback={<div className="bg-transparent min-h-screen text-white p-8">Loading...</div>}>
      <PapersContent />
    </Suspense>
  );
}
