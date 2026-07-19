'use client';

import { useState, useRef, useCallback } from 'react';
import { Upload, AlertCircle, Loader2 } from 'lucide-react';
import { parseModel, parsePytorchModel } from '@/lib/model-viz-api';
import type { ParsedGraph } from '@/lib/model-viz-api';
import InputShapeForm from './InputShapeForm';

const MAX_BYTES = 50 * 1024 * 1024; // 50 MB — matches backend

type Props = {
  onParsed: (graph: ParsedGraph, filename: string, format: 'onnx' | 'pytorch') => void;
};

type Status = 'idle' | 'awaiting_shape' | 'parsing' | 'error';

export default function UploadZone({ onParsed }: Props) {
  const [status, setStatus] = useState<Status>('idle');
  const [error, setError] = useState('');
  const [dragging, setDragging] = useState(false);
  // Held while we wait for the user to enter input shape (PyTorch flow)
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // ── file selection ────────────────────────────────────────────────────────

  const handleFile = useCallback(
    (file: File) => {
      const lower = file.name.toLowerCase();

      if (!lower.endsWith('.onnx') && !lower.endsWith('.pt') && !lower.endsWith('.pth')) {
        setError('Only .onnx, .pt, and .pth files are supported.');
        setStatus('error');
        return;
      }
      if (file.size > MAX_BYTES) {
        setError('File too large. Maximum size is 50 MB.');
        setStatus('error');
        return;
      }

      if (lower.endsWith('.onnx')) {
        // ONNX path: parse immediately
        startOnnxParse(file);
      } else {
        // PyTorch path: need input shape first
        setPendingFile(file);
        setStatus('awaiting_shape');
        setError('');
      }
    },
    [],
  );

  // ── parsing ────────────────────────────────────────────────────────────────

  async function startOnnxParse(file: File) {
    setStatus('parsing');
    setError('');
    try {
      const graph = await parseModel(file);
      onParsed(graph, file.name, 'onnx');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Parse failed');
      setStatus('error');
    }
  }

  async function startPytorchParse(shape: number[]) {
    if (!pendingFile) return;
    const file = pendingFile;
    setStatus('parsing');
    setError('');
    try {
      const graph = await parsePytorchModel(file, shape);
      onParsed(graph, file.name, 'pytorch');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Parse failed');
      setStatus('error');
      setPendingFile(null);
    }
  }

  // ── drag-and-drop ─────────────────────────────────────────────────────────

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  // ── render ────────────────────────────────────────────────────────────────

  return (
    <div style={{ maxWidth: 500, width: '100%', padding: '0 24px' }}>
      {/* Heading */}
      <div style={{ textAlign: 'center', marginBottom: 32 }}>
        <div
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: 44,
            height: 44,
            borderRadius: '50%',
            background: '#A78BFA1A',
            marginBottom: 12,
          }}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#A78BFA" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="3" />
            <circle cx="5" cy="5" r="2" />
            <circle cx="19" cy="5" r="2" />
            <circle cx="5" cy="19" r="2" />
            <circle cx="19" cy="19" r="2" />
            <line x1="7" y1="5" x2="10" y2="10" />
            <line x1="17" y1="5" x2="14" y2="10" />
            <line x1="7" y1="19" x2="10" y2="14" />
            <line x1="17" y1="19" x2="14" y2="14" />
          </svg>
        </div>
        <h1 style={{ fontSize: 24, fontWeight: 700, color: '#f5f5f5', marginBottom: 6 }}>
          Model Visualizer
        </h1>
        <p style={{ fontSize: 13, color: '#737373' }}>
          Upload an ONNX or PyTorch model to render its architecture as an interactive graph
        </p>
      </div>

      {/* Drop zone — hidden while awaiting shape */}
      {status !== 'awaiting_shape' && (
        <div
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => status !== 'parsing' && inputRef.current?.click()}
          style={{
            border: `2px dashed ${dragging ? '#A78BFA' : '#2a2a2a'}`,
            borderRadius: 12,
            padding: '48px 32px',
            textAlign: 'center',
            cursor: status === 'parsing' ? 'default' : 'pointer',
            transition: 'border-color 0.15s, background 0.15s',
            background: dragging ? 'rgba(167,139,250,0.04)' : 'transparent',
          }}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".onnx,.pt,.pth"
            style={{ display: 'none' }}
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleFile(file);
              e.target.value = '';
            }}
          />

          {status === 'parsing' ? (
            <>
              <Loader2
                size={28}
                style={{ color: '#A78BFA', margin: '0 auto 12px', display: 'block' }}
                className="animate-spin"
              />
              <p style={{ fontSize: 13, color: '#a3a3a3' }}>
                {pendingFile?.name.endsWith('.onnx') === false
                  ? 'Sending to sandbox…'
                  : 'Reading graph structure…'}
              </p>
              <p style={{ fontSize: 11, color: '#525252', marginTop: 4 }}>
                {pendingFile && !pendingFile.name.endsWith('.onnx')
                  ? 'Running in E2B sandbox — may take 2–4 min on first use'
                  : 'Running shape inference on all tensors'}
              </p>
            </>
          ) : (
            <>
              <Upload
                size={28}
                style={{ color: '#525252', margin: '0 auto 12px', display: 'block' }}
              />
              <p style={{ fontSize: 13, color: '#a3a3a3', marginBottom: 4 }}>
                Drop your{' '}
                <span style={{ color: '#f5f5f5', fontFamily: 'monospace' }}>.onnx</span>
                {' / '}
                <span style={{ color: '#f97316', fontFamily: 'monospace' }}>.pt</span>
                {' '}file here
              </p>
              <p style={{ fontSize: 11, color: '#525252' }}>or click to browse · max 50 MB</p>
            </>
          )}
        </div>
      )}

      {/* PyTorch input-shape form (shown between file pick and parse) */}
      {status === 'awaiting_shape' && pendingFile && (
        <InputShapeForm
          filename={pendingFile.name}
          onSubmit={startPytorchParse}
          onCancel={() => { setPendingFile(null); setStatus('idle'); }}
          disabled={false}
        />
      )}

      {/* Error */}
      {status === 'error' && (
        <div
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: 8,
            marginTop: 12,
            padding: '10px 14px',
            background: 'rgba(239,68,68,0.07)',
            border: '1px solid rgba(239,68,68,0.2)',
            borderRadius: 8,
          }}
        >
          <AlertCircle size={14} style={{ color: '#ef4444', flexShrink: 0, marginTop: 1 }} />
          <span style={{ fontSize: 12, color: '#f87171', lineHeight: 1.5 }}>{error}</span>
        </div>
      )}

      {/* Example models — only shown in idle / error state */}
      {(status === 'idle' || status === 'error') && (
        <>
          <div style={{ marginTop: 28 }}>
            <p
              style={{
                fontSize: 10,
                color: '#404040',
                textAlign: 'center',
                marginBottom: 10,
                textTransform: 'uppercase',
                letterSpacing: '0.1em',
              }}
            >
              Try an example — download then upload
            </p>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'center', flexWrap: 'wrap' }}>
              {EXAMPLES.map((m) => (
                <a
                  key={m.name}
                  href={m.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={(e) => e.stopPropagation()}
                  style={{
                    fontSize: 11,
                    color: '#A78BFA',
                    padding: '4px 12px',
                    border: '1px solid rgba(167,139,250,0.2)',
                    borderRadius: 6,
                    textDecoration: 'none',
                  }}
                >
                  {m.name}
                </a>
              ))}
            </div>
          </div>

          {/* Export hint */}
          <div
            style={{
              marginTop: 24,
              padding: '10px 14px',
              background: '#111',
              borderRadius: 8,
              border: '1px solid #1a1a1a',
            }}
          >
            <p style={{ fontSize: 10, color: '#525252', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              Export your PyTorch model
            </p>
            <code style={{ fontSize: 11, color: '#a3a3a3', display: 'block', lineHeight: 1.6, fontFamily: 'monospace' }}>
              <span style={{ color: '#737373' }}># Option A — ONNX (recommended)</span><br />
              torch.onnx.export(model, dummy, &quot;model.onnx&quot;, opset_version=17)<br />
              <br />
              <span style={{ color: '#737373' }}># Option B — PyTorch native (.pt)</span><br />
              torch.save(model, &quot;model.pt&quot;){'  '}<span style={{ color: '#737373' }}># save full model, not state_dict</span>
            </code>
          </div>
        </>
      )}
    </div>
  );
}

const EXAMPLES = [
  {
    name: 'ResNet-50',
    url: 'https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx',
  },
  {
    name: 'MobileNetV2',
    url: 'https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx',
  },
  {
    name: 'SqueezeNet',
    url: 'https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.0-7.onnx',
  },
];
