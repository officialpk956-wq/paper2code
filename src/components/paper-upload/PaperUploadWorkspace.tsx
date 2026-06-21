'use client';

import { useMemo, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Spinner } from '@/components/ui/Spinner';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { ThreeColumnLayout } from '@/components/layout/three-column-layout';
import { Upload, FileText, ShieldCheck, Sparkles } from 'lucide-react';

type UploadResponse = {
  paper_id: number;
  title: string;
  classification?: string;
  status?: string;
  report?: {
    nodes?: number;
    edges?: number;
    modules?: number;
    parameters?: number;
    flops?: number;
    figure_count?: number;
    equation_count?: number;
    detected_components?: string[];
  };
  ingestion?: {
    page_count?: number;
    text_extraction_method?: string;
    figure_count?: number;
    equation_count?: number;
    source_filename?: string;
  };
};

const STATUS_STEPS = [
  'Select a PDF and optional title',
  'Upload to the ingestion pipeline',
  'Extract text, figures, and equations',
  'Build graph, modules, and persistence record',
  'Open the generated workspace',
];

export function PaperUploadWorkspace() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [paperName, setPaperName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<UploadResponse | null>(null);

  const fileSummary = useMemo(() => {
    if (!selectedFile) return 'No PDF selected';
    const sizeKb = Math.max(1, Math.round(selectedFile.size / 1024));
    return `${selectedFile.name} · ${sizeKb} KB`;
  }, [selectedFile]);

  async function handleSubmit() {
    if (!selectedFile) {
      setError('Choose a PDF before uploading.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      if (paperName.trim()) {
        formData.append('paperName', paperName.trim());
      }

      const response = await fetch('/api/papers/upload', {
        method: 'POST',
        body: formData,
      });

      const payload = (await response.json()) as UploadResponse & { error?: string };

      if (!response.ok || payload.error) {
        throw new Error(payload.error ?? `Upload failed with status ${response.status}`);
      }

      setResult(payload);
      router.push(`/papers/upload/${payload.paper_id}`);
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : String(submitError));
    } finally {
      setLoading(false);
    }
  }

  const left = (
    <div className="h-full overflow-y-auto p-5 space-y-5">
      <div className="rounded-2xl border border-[--color-border] bg-[--bg-panel] p-4 shadow-[0_24px_80px_rgba(0,0,0,0.18)]">
        <SectionLabel>Upload Paper</SectionLabel>
        <h1 className="mt-2 text-2xl font-semibold text-[--color-text-primary]">
          Phase 13 ingestion pipeline
        </h1>
        <p className="mt-2 text-sm leading-6 text-[--color-text-secondary]">
          Upload a research PDF and generate a persisted paper record with parsed metadata,
          extracted figures, equations, modules, and graph data.
        </p>
      </div>

      <div className="rounded-2xl border border-[--color-border] bg-[--bg-surface] p-4 space-y-4">
        <SectionLabel>Source PDF</SectionLabel>
        <Input
          id="paper-name"
          label="Optional paper title"
          placeholder="Attention Is All You Need"
          value={paperName}
          onChange={(event) => setPaperName(event.target.value)}
          disabled={loading}
        />

        <div className="rounded-xl border border-dashed border-[--color-border] bg-[--bg-body] p-4">
          <div className="flex items-start gap-3">
            <FileText className="mt-0.5 h-5 w-5 text-[--accent-cyan]" />
            <div className="min-w-0">
              <div className="text-sm font-medium text-[--color-text-primary]">Select a PDF</div>
              <div className="mt-1 text-xs text-[--color-text-secondary]">
                Maximum size: 20 MB. The upload is validated before it reaches the parser.
              </div>
            </div>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="application/pdf"
            className="sr-only"
            onChange={(event) => setSelectedFile(event.target.files?.[0] ?? null)}
          />

          <div className="mt-4 flex flex-wrap gap-3">
            <Button variant="secondary" type="button" onClick={() => fileInputRef.current?.click()}>
              <Upload className="h-4 w-4" />
              Choose file
            </Button>
            <Button variant="primary" type="button" disabled={loading} onClick={handleSubmit}>
              {loading ? <Spinner size={14} color="white" /> : <Sparkles className="h-4 w-4" />}
              {loading ? 'Processing' : 'Generate workspace'}
            </Button>
          </div>

          <p className="mt-3 text-xs text-[--color-text-muted]">{fileSummary}</p>
        </div>

        {error && (
          <div className="rounded-xl border border-[rgba(var(--color-error-rgb),0.35)] bg-[rgba(var(--color-error-rgb),0.08)] p-3 text-sm text-[--color-error]">
            {error}
          </div>
        )}
      </div>
    </div>
  );

  const center = (
    <div className="h-full overflow-y-auto p-5">
      <div className="rounded-3xl border border-[--color-border] bg-[linear-gradient(180deg,rgba(12,18,31,0.96),rgba(8,11,18,0.96))] p-6 shadow-[0_32px_100px_rgba(0,0,0,0.24)]">
        <div className="flex items-center justify-between gap-4">
          <div>
            <SectionLabel>Ingestion Flow</SectionLabel>
            <h2 className="mt-2 text-2xl font-semibold text-[--color-text-primary]">
              PDF → metadata → graph → workspace
            </h2>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-[--color-text-secondary]">
              The backend parses the PDF, extracts figures and equations, runs the existing
              Paper2Code pipeline, stores the result, and returns a paper record ready for study.
            </p>
          </div>
          <div className="rounded-full border border-[--color-border] bg-[--bg-surface] px-3 py-1 text-xs text-[--color-text-tertiary]">
            Deterministic pipeline
          </div>
        </div>

        <div className="mt-6 grid gap-3">
          {STATUS_STEPS.map((step, index) => (
            <div
              key={step}
              className={`flex items-center gap-3 rounded-2xl border px-4 py-3 ${
                loading && index === 1
                  ? 'border-[--accent-cyan]/40 bg-[rgba(6,182,212,0.08)]'
                  : 'border-[--color-border] bg-[--bg-body]'
              }`}
            >
              <div className="flex h-8 w-8 items-center justify-center rounded-full border border-[--color-border] text-xs font-semibold text-[--color-text-secondary]">
                {index + 1}
              </div>
              <div className="min-w-0 flex-1">
                <div className="text-sm font-medium text-[--color-text-primary]">{step}</div>
                {index === 1 && loading && (
                  <div className="mt-1 flex items-center gap-2 text-xs text-[--accent-cyan]">
                    <Spinner size={12} />
                    Running parser
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        {result && (
          <div className="mt-6 rounded-2xl border border-emerald-500/30 bg-emerald-500/8 p-4">
            <div className="flex items-center gap-2 text-sm font-semibold text-emerald-300">
              <ShieldCheck className="h-4 w-4" />
              Upload complete
            </div>
            <p className="mt-2 text-sm text-[--color-text-secondary]">
              Redirecting to the generated workspace for {result.title}.
            </p>
          </div>
        )}
      </div>
    </div>
  );

  const right = (
    <div className="h-full overflow-y-auto p-5 space-y-5">
      <div className="rounded-2xl border border-[--color-border] bg-[--bg-panel] p-4">
        <SectionLabel>Persisted Output</SectionLabel>
        <div className="mt-3 space-y-3 text-sm text-[--color-text-secondary]">
          <div>Paper record with architecture graph and module order</div>
          <div>Figure metadata stored on the paper ingestion payload</div>
          <div>Equation candidates persisted for the generated workspace</div>
        </div>
      </div>

      <div className="rounded-2xl border border-[--color-border] bg-[--bg-surface] p-4">
        <SectionLabel>What you get</SectionLabel>
        <div className="mt-4 space-y-3 text-sm text-[--color-text-secondary]">
          <div className="flex items-center justify-between gap-3">
            <span>Parsed metadata</span>
            <span className="text-[--color-text-primary]">Title, source, pages</span>
          </div>
          <div className="flex items-center justify-between gap-3">
            <span>Figures</span>
            <span className="text-[--color-text-primary]">Image xrefs + captions</span>
          </div>
          <div className="flex items-center justify-between gap-3">
            <span>Equations</span>
            <span className="text-[--color-text-primary]">Extracted from text</span>
          </div>
          <div className="flex items-center justify-between gap-3">
            <span>Workspace</span>
            <span className="text-[--color-text-primary]">Generated paper detail page</span>
          </div>
        </div>
      </div>
    </div>
  );

  return <ThreeColumnLayout left={left} center={center} right={right} leftWidth="w-[26rem]" rightWidth="w-[22rem]" />;
}
