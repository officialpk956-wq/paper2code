"use client";

import React, { useEffect, useState, useCallback, use } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import Editor from "@monaco-editor/react";
import { Button } from "@/components/ui/Button";

type ChallengePart = {
  id: number;
  title: string;
  description_md: string;
  paper_section_md: string;
  starter_code: string;
  setup_code: string;
  order_idx: number;
  xp_reward: number;
  user_passed: boolean;
  is_locked: boolean;
};

type Challenge = {
  id: number;
  title: string;
  parts: ChallengePart[];
};

type ProgressHistory = {
  id: number;
  passed: boolean;
  created_at: string;
};

type Progress = {
  attempts: number;
  passed: boolean;
  best_submission_id: number | null;
  history: ProgressHistory[];
};

type RunResult = {
  passed: boolean;
  stdout: string;
  stderr: string;
  time_ms: number;
  xp_earned: number;
};

export default function PaperImplementPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = use(params);
  const [challenges, setChallenges] = useState<Challenge[]>([]);
  const [loading, setLoading] = useState(true);
  const [activePartId, setActivePartId] = useState<number | null>(null);
  
  const [code, setCode] = useState("");
  const [progress, setProgress] = useState<Progress | null>(null);
  
  const [runState, setRunState] = useState<"idle" | "running" | "passed" | "failed">("idle");
  const [runResult, setRunResult] = useState<RunResult | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [token, setToken] = useState<string | null>(null);

  useEffect(() => {
    setToken(localStorage.getItem("access_token"));
  }, []);

  const fetchChallenges = useCallback(async () => {
    try {
      const currentToken = localStorage.getItem("access_token");
      const res = await fetch(
        (process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000") + `/api/papers/${slug}/challenges`,
        {
          headers: currentToken ? { Authorization: `Bearer ${currentToken}` } : {}
        }
      );
      if (res.ok) {
        const data = await res.json();
        setChallenges(data);
        return data;
      }
    } catch (err) {
      console.error(err);
    }
    return [];
  }, [slug]);

  useEffect(() => {
    fetchChallenges().then((data: Challenge[]) => {
      setLoading(false);
      if (data.length > 0 && data[0].parts.length > 0) {
        // Just set if activePartId is null, otherwise it keeps state
        setActivePartId(prev => {
          if (prev === null) {
            const firstUnlocked = data[0].parts.find(p => !p.is_locked) || data[0].parts[0];
            setCode(firstUnlocked.starter_code);
            fetchProgress(data[0].id, firstUnlocked.id);
            return firstUnlocked.id;
          }
          return prev;
        });
      }
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fetchChallenges]);

  const fetchProgress = useCallback((challengeId: number, partId: number) => {
    const currentToken = localStorage.getItem("access_token");
    if (!currentToken) return;
    fetch((process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000") + `/api/papers/${slug}/challenges/${challengeId}/parts/${partId}/progress`, {
      headers: { Authorization: `Bearer ${currentToken}` }
    })
      .then(res => res.json())
      .then(data => setProgress(data))
      .catch(console.error);
  }, [slug]);

  const handleSelectPart = (part: ChallengePart) => {
    if (part.is_locked) {
      setToast(`Complete Part ${part.order_idx - 1} first`);
      setTimeout(() => setToast(null), 3000);
      return;
    }
    setActivePartId(part.id);
    setCode(part.starter_code);
    setRunState("idle");
    setRunResult(null);
    setProgress(null);
    const challenge = challenges.find(c => c.parts.some(p => p.id === part.id));
    if (challenge) {
      fetchProgress(challenge.id, part.id);
    }
  };

  const activeChallenge = challenges.find(c => c.parts.some(p => p.id === activePartId));
  const activePart = activeChallenge?.parts.find(p => p.id === activePartId);

  const handleRun = async () => {
    if (!activeChallenge || !activePart) return;
    const currentToken = localStorage.getItem("access_token");
    if (!currentToken) {
      alert("Please log in to run code");
      return;
    }

    setRunState("running");
    try {
      const res = await fetch((process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000") + `/api/papers/${slug}/challenges/${activeChallenge.id}/parts/${activePart.id}/run`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${currentToken}`
        },
        body: JSON.stringify({ code })
      });
      const data = await res.json();
      setRunResult(data);
      setRunState(data.passed ? "passed" : "failed");
      fetchProgress(activeChallenge.id, activePart.id);
      
      if (data.passed) {
        await fetchChallenges();
      }
    } catch (err) {
      console.error(err);
      setRunState("failed");
    }
  };

  if (loading) return <div className="p-8 h-screen bg-[#0f1115] text-white">Loading...</div>;

  if (challenges.length === 0) {
    return <div className="p-8 h-screen bg-[#0f1115] text-white">No implementation challenges available for this paper yet</div>;
  }

  // Flatten parts for progress bar
  const allParts = challenges.flatMap(c => c.parts);
  const passedParts = allParts.filter(p => p.user_passed).length;
  const totalParts = allParts.length;

  return (
    <div className="flex h-screen overflow-hidden text-white bg-[#0f1115]">
      {toast && (
        <div className="fixed bottom-4 left-1/2 -translate-x-1/2 bg-gray-800 text-white px-4 py-2 rounded shadow-lg z-50 transition-opacity">
          {toast}
        </div>
      )}
      
      {/* LEFT COLUMN */}
      <div className="w-64 flex-shrink-0 flex flex-col border-r border-gray-800 overflow-y-auto">
        <div className="p-4 border-b border-gray-800">
          <div className="text-xs text-gray-500 uppercase font-semibold mb-1 truncate">{slug}</div>
          <div className="font-bold">{challenges[0]?.title}</div>
        </div>
        
        <div className="flex-1 p-4">
          <div className="space-y-2">
            {allParts.map((part) => {
              const isActive = activePartId === part.id;
              return (
                <button
                  key={part.id}
                  onClick={() => handleSelectPart(part)}
                  className={`w-full text-left px-3 py-2 rounded text-sm flex items-center justify-between ${
                    isActive ? "bg-[#7C3AED] bg-opacity-20 text-[#7C3AED]" : 
                    part.is_locked ? "text-gray-600 cursor-not-allowed" : "text-gray-300 hover:bg-gray-800"
                  }`}
                >
                  <div className="flex items-center gap-2 truncate">
                    {isActive ? (
                      <div className="w-2 h-2 rounded-full bg-[#7C3AED]" />
                    ) : part.user_passed ? (
                      <div className="text-green-500 font-bold">✓</div>
                    ) : part.is_locked ? (
                      <div className="w-2 h-2 rounded-full bg-gray-700" />
                    ) : (
                      <div className="w-2 h-2 rounded-full border border-gray-500" />
                    )}
                    <span className="truncate text-sm font-medium">Part {part.order_idx}: {part.title}</span>
                  </div>
                  {part.is_locked && <span className="text-xs">🔒</span>}
                </button>
              );
            })}
          </div>
        </div>
        
        <div className="p-4 border-t border-gray-800">
          <div className="text-xs text-gray-400 mb-2">{passedParts} / {totalParts} parts completed</div>
          <div className="h-1 bg-gray-800 rounded overflow-hidden">
            <div className="h-full bg-[#7C3AED] transition-all duration-300" style={{ width: `${(passedParts / Math.max(totalParts, 1)) * 100}%` }} />
          </div>
        </div>
      </div>

      {/* CENTER COLUMN */}
      <div className="flex-1 flex flex-col min-w-0">
        <div className="h-48 overflow-y-auto p-4 border-b border-gray-800 prose prose-invert max-w-none">
          {activePart && (
            <>
              <h2 className="font-bold text-xl mb-4 text-white">{activePart.title}</h2>
              <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkMath]}
                rehypePlugins={[rehypeKatex]}
              >
                {activePart.description_md}
              </ReactMarkdown>
            </>
          )}
        </div>
        
        <div className="flex-1 min-h-0 relative">
          {!token ? (
            <div className="absolute inset-0 flex items-center justify-center bg-[#1e1e1e]">
              <div className="text-gray-400">Please log in to run code</div>
            </div>
          ) : (
            <Editor
              language="python"
              theme="vs-dark"
              value={code}
              onChange={(val) => setCode(val || "")}
              options={{ minimap: { enabled: false }, fontSize: 14, lineNumbers: "on" }}
            />
          )}
        </div>
        
        <div className="h-12 flex items-center justify-between px-4 border-t border-gray-800 bg-[#0f1115]">
          <div className="text-xs text-gray-500">Python 3.11 · PyTorch CPU · E2B Sandbox</div>
          <Button
            variant="primary"
            onClick={handleRun}
            disabled={runState === "running" || !token}
            style={{ background: "#7C3AED", color: "white" }}
          >
            {runState === "running" ? (
              <span className="flex items-center gap-2">
                <span className="animate-spin rounded-full h-3 w-3 border-b-2 border-white"></span>
                Running...
              </span>
            ) : "Run Code"}
          </Button>
        </div>
      </div>

      {/* RIGHT COLUMN */}
      <div className="w-80 flex-shrink-0 flex flex-col border-l border-gray-800">
        <div className="p-4 border-b border-gray-800">
          {activePart?.paper_section_md ? (
            <blockquote className="border-l-2 border-[#7C3AED] pl-3 text-sm text-gray-300 italic">
              <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkMath]}
                rehypePlugins={[rehypeKatex]}
              >
                {activePart.paper_section_md}
              </ReactMarkdown>
            </blockquote>
          ) : (
            <div className="text-sm text-gray-500">No paper excerpt for this part</div>
          )}
        </div>

        <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-4">
          {runState === "idle" && (
            <div className="bg-gray-800 rounded p-4 text-center text-sm text-gray-400">
              Write your solution and click Run Code
            </div>
          )}
          
          {runState === "running" && (
            <div className="bg-gray-800 rounded p-4 text-center animate-pulse">
              <div className="text-sm font-bold mb-1">Running in E2B sandbox...</div>
              <div className="text-xs text-gray-500">PyTorch · typically 3–8 seconds</div>
            </div>
          )}
          
          {runState === "passed" && runResult && (
            <div>
              <div className="text-green-500 font-bold mb-2 flex items-center justify-between">
                <span>✓ All Tests Passed</span>
                <span className="text-xs font-normal text-yellow-500 bg-yellow-500 bg-opacity-10 px-2 py-0.5 rounded">
                  {runResult.xp_earned > 0 ? `+${runResult.xp_earned} XP` : "Already solved (+0 XP)"}
                </span>
              </div>
              {runResult.stdout && (
                <pre className="bg-black p-2 rounded text-xs text-gray-300 overflow-x-auto">
                  {runResult.stdout}
                </pre>
              )}
            </div>
          )}
          
          {runState === "failed" && runResult && (
            <div>
              <div className="text-red-500 font-bold mb-2">✗ Tests Failed</div>
              <pre className="bg-black p-2 rounded text-xs overflow-x-auto whitespace-pre-wrap">
                {runResult.stderr.split("\n").map((line, i) => (
                  <div key={i} className={line.includes("AssertionError") ? "text-red-400" : "text-gray-400"}>
                    {line}
                  </div>
                ))}
              </pre>
            </div>
          )}
        </div>

        <div className="p-4 border-t border-gray-800 max-h-40 overflow-y-auto">
          <div className="text-xs font-bold text-gray-500 mb-2 uppercase">
            Attempts: {progress?.attempts || 0}
          </div>
          <div className="space-y-1">
            {progress?.history?.map((h, i) => (
              <div key={h.id} className="text-xs flex items-center justify-between">
                <span className={h.passed ? "text-green-500" : "text-red-500"}>
                  #{progress.history.length - i} {h.passed ? "✓" : "✗"}
                </span>
                <span className="text-gray-600">
                  {new Date(h.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
