import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ExecutableGraphViewer } from '@/components/paper-upload/ExecutableGraphViewer';
import type { ExecutableGraph } from '@/components/paper-upload/ExecutableGraphViewer';

// ---------------------------------------------------------------------------
// Test data
// ---------------------------------------------------------------------------

const RESNET_EG: ExecutableGraph = {
  id: 'eg-1-resnet',
  nodes: [
    {
      id: 'stem',
      name: 'Stem (7×7 Conv)',
      type: 'conv',
      shape: [1, 64, 112, 112],
      params: 47_040,
      flops: 235.9,
      metadata: { input_shape: [1, 3, 224, 224] },
    },
    {
      id: 'pool1',
      name: 'Max Pool',
      type: 'pooling',
      shape: [1, 64, 56, 56],
      params: 0,
      flops: 0,
      metadata: { input_shape: [1, 64, 112, 112] },
    },
    {
      id: 'head',
      name: 'Classifier (→1000)',
      type: 'head',
      shape: [1, 1000],
      params: 2_049_000,
      flops: 4.1,
      metadata: { input_shape: [1, 2048] },
    },
  ],
  edges: [
    { source: 'stem',  target: 'pool1', edge_type: 'sequential' },
    { source: 'pool1', target: 'head',  edge_type: 'sequential' },
  ],
  input_spec:  [1, 3, 224, 224],
  output_spec: [1, 1000],
  parameter_count: 2_096_040,
  flops: 240.0,
  validation_status: 'valid',
  validation_report: { valid: true, errors: [], warnings: [] },
};

const UNET_EG: ExecutableGraph = {
  ...RESNET_EG,
  id: 'eg-2-unet',
  nodes: [
    {
      id: 'enc1',
      name: 'Encoder 1',
      type: 'conv',
      shape: [1, 64, 256, 256],
      params: 700_000,
      flops: 512.0,
      metadata: { input_shape: [1, 1, 256, 256] },
    },
    {
      id: 'dec1',
      name: 'Decoder 1',
      type: 'conv',
      shape: [1, 64, 256, 256],
      params: 700_000,
      flops: 512.0,
      metadata: { input_shape: [1, 64, 256, 256] },
    },
    {
      id: 'out',
      name: 'Output',
      type: 'head',
      shape: [1, 2, 256, 256],
      params: 10_000,
      flops: 0.5,
      metadata: { input_shape: [1, 64, 256, 256] },
    },
  ],
  edges: [
    { source: 'enc1', target: 'dec1', edge_type: 'sequential' },
    { source: 'dec1', target: 'out',  edge_type: 'sequential' },
    { source: 'enc1', target: 'dec1', edge_type: 'concat' },
  ],
  validation_status: 'valid',
  validation_report: { valid: true, errors: [], warnings: [] },
};

const INVALID_EG: ExecutableGraph = {
  ...RESNET_EG,
  id: 'eg-3-invalid',
  validation_status: 'invalid',
  validation_report: {
    valid: false,
    errors: ['Graph contains a directed cycle'],
    warnings: ['Shape mismatch on stem→pool1'],
  },
};

const WARNINGS_EG: ExecutableGraph = {
  ...RESNET_EG,
  id: 'eg-4-warnings',
  validation_status: 'warnings',
  validation_report: {
    valid: true,
    errors: [],
    warnings: ["Node 'orphan' is not connected"],
  },
};

// ---------------------------------------------------------------------------
// Empty state
// ---------------------------------------------------------------------------

describe('ExecutableGraphViewer — empty state', () => {
  it('renders empty-state message when graph is null', () => {
    render(<ExecutableGraphViewer graph={null} />);
    expect(
      screen.getByText(/no executable graph available/i),
    ).toBeInTheDocument();
  });

  it('empty state has role=status', () => {
    render(<ExecutableGraphViewer graph={null} />);
    expect(screen.getByRole('status')).toBeInTheDocument();
  });

  it('does not render toolbar when graph is null', () => {
    render(<ExecutableGraphViewer graph={null} />);
    expect(
      screen.queryByRole('button', { name: /zoom in/i }),
    ).not.toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Basic rendering
// ---------------------------------------------------------------------------

describe('ExecutableGraphViewer — rendering', () => {
  it('renders the SVG canvas with role=img', () => {
    const { container } = render(<ExecutableGraphViewer graph={RESNET_EG} />);
    expect(container.querySelector('svg[role="img"]')).toBeTruthy();
  });

  it('renders graph id in toolbar', () => {
    render(<ExecutableGraphViewer graph={RESNET_EG} />);
    expect(screen.getByText('eg-1-resnet')).toBeInTheDocument();
  });

  it('renders node count and edge count', () => {
    render(<ExecutableGraphViewer graph={RESNET_EG} />);
    expect(
      screen.getByText(`${RESNET_EG.nodes.length} nodes · ${RESNET_EG.edges.length} edges`),
    ).toBeInTheDocument();
  });

  it('renders validation status badge', () => {
    render(<ExecutableGraphViewer graph={RESNET_EG} />);
    expect(screen.getByText(/valid/i)).toBeInTheDocument();
  });

  it('renders each node as an SVG button', () => {
    render(<ExecutableGraphViewer graph={RESNET_EG} />);
    const btns = screen
      .getAllByRole('button')
      .filter((b) => b.getAttribute('aria-label')?.includes('—'));
    expect(btns.length).toBe(RESNET_EG.nodes.length);
  });

  it('renders zoom controls', () => {
    render(<ExecutableGraphViewer graph={RESNET_EG} />);
    expect(screen.getByRole('button', { name: /zoom in/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /zoom out/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /reset view/i })).toBeInTheDocument();
  });

  it('renders export buttons', () => {
    render(<ExecutableGraphViewer graph={RESNET_EG} />);
    expect(screen.getByRole('button', { name: /export as json/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /export as mermaid/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /export as dot/i })).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Node selection
// ---------------------------------------------------------------------------

describe('ExecutableGraphViewer — node selection', () => {
  it('shows node inspector on click', async () => {
    render(<ExecutableGraphViewer graph={RESNET_EG} />);
    const btn = screen.getByRole('button', { name: /Stem.*conv/i });
    await userEvent.click(btn);
    expect(
      screen.getByRole('region', { name: /selected node details/i }),
    ).toBeInTheDocument();
  });

  it('marks selected node as aria-pressed=true', async () => {
    render(<ExecutableGraphViewer graph={RESNET_EG} />);
    const btn = screen.getByRole('button', { name: /Stem.*conv/i });
    expect(btn.getAttribute('aria-pressed')).toBe('false');
    await userEvent.click(btn);
    expect(btn.getAttribute('aria-pressed')).toBe('true');
  });

  it('deselects on second click', async () => {
    render(<ExecutableGraphViewer graph={RESNET_EG} />);
    const btn = screen.getByRole('button', { name: /Stem.*conv/i });
    await userEvent.click(btn);
    expect(
      screen.getByRole('region', { name: /selected node details/i }),
    ).toBeInTheDocument();
    await userEvent.click(btn);
    expect(
      screen.queryByRole('region', { name: /selected node details/i }),
    ).not.toBeInTheDocument();
  });

  it('inspector shows close button', async () => {
    render(<ExecutableGraphViewer graph={RESNET_EG} />);
    await userEvent.click(screen.getByRole('button', { name: /Stem.*conv/i }));
    expect(
      screen.getByRole('button', { name: /close details panel/i }),
    ).toBeInTheDocument();
  });

  it('supports Enter key to select node', () => {
    render(<ExecutableGraphViewer graph={RESNET_EG} />);
    const btn = screen.getByRole('button', { name: /Stem.*conv/i });
    btn.focus();
    fireEvent.keyDown(btn, { key: 'Enter' });
    expect(
      screen.getByRole('region', { name: /selected node details/i }),
    ).toBeInTheDocument();
  });

  it('supports Space key to select node', () => {
    render(<ExecutableGraphViewer graph={RESNET_EG} />);
    const btn = screen.getByRole('button', { name: /Stem.*conv/i });
    btn.focus();
    fireEvent.keyDown(btn, { key: ' ' });
    expect(
      screen.getByRole('region', { name: /selected node details/i }),
    ).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Zoom controls
// ---------------------------------------------------------------------------

describe('ExecutableGraphViewer — zoom', () => {
  it('starts at 100%', () => {
    render(<ExecutableGraphViewer graph={RESNET_EG} />);
    expect(screen.getByText('100%')).toBeInTheDocument();
  });

  it('increments to 120% on zoom in', async () => {
    render(<ExecutableGraphViewer graph={RESNET_EG} />);
    await userEvent.click(screen.getByRole('button', { name: /zoom in/i }));
    expect(screen.getByText('120%')).toBeInTheDocument();
  });

  it('decrements to 80% on zoom out', async () => {
    render(<ExecutableGraphViewer graph={RESNET_EG} />);
    await userEvent.click(screen.getByRole('button', { name: /zoom out/i }));
    expect(screen.getByText('80%')).toBeInTheDocument();
  });

  it('resets to 100% on reset view', async () => {
    render(<ExecutableGraphViewer graph={RESNET_EG} />);
    await userEvent.click(screen.getByRole('button', { name: /zoom in/i }));
    await userEvent.click(screen.getByRole('button', { name: /reset view/i }));
    expect(screen.getByText('100%')).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Validation display
// ---------------------------------------------------------------------------

describe('ExecutableGraphViewer — validation display', () => {
  it('shows validation report region for invalid graph', () => {
    render(<ExecutableGraphViewer graph={INVALID_EG} />);
    expect(
      screen.getByRole('region', { name: /graph validation report/i }),
    ).toBeInTheDocument();
  });

  it('displays error messages', () => {
    render(<ExecutableGraphViewer graph={INVALID_EG} />);
    expect(screen.getByText(/directed cycle/i)).toBeInTheDocument();
  });

  it('displays warning messages', () => {
    render(<ExecutableGraphViewer graph={INVALID_EG} />);
    expect(screen.getByText(/shape mismatch/i)).toBeInTheDocument();
  });

  it('displays warning-only report', () => {
    render(<ExecutableGraphViewer graph={WARNINGS_EG} />);
    expect(
      screen.getByRole('region', { name: /graph validation report/i }),
    ).toBeInTheDocument();
    expect(screen.getByText(/orphan/i)).toBeInTheDocument();
  });

  it('does not render validation region for a valid graph', () => {
    render(<ExecutableGraphViewer graph={RESNET_EG} />);
    expect(
      screen.queryByRole('region', { name: /graph validation report/i }),
    ).not.toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Export buttons
// ---------------------------------------------------------------------------

describe('ExecutableGraphViewer — export', () => {
  it('JSON button triggers download', async () => {
    const createObjURL = vi.fn(() => 'blob:mock');
    const revokeObjURL = vi.fn();
    global.URL.createObjectURL = createObjURL;
    global.URL.revokeObjectURL = revokeObjURL;

    render(<ExecutableGraphViewer graph={RESNET_EG} />);
    await userEvent.click(screen.getByRole('button', { name: /export as json/i }));
    expect(createObjURL).toHaveBeenCalled();
  });

  it('Mermaid button triggers download', async () => {
    const createObjURL = vi.fn(() => 'blob:mock');
    global.URL.createObjectURL = createObjURL;

    render(<ExecutableGraphViewer graph={RESNET_EG} />);
    await userEvent.click(screen.getByRole('button', { name: /export as mermaid/i }));
    expect(createObjURL).toHaveBeenCalled();
  });

  it('DOT button triggers download', async () => {
    const createObjURL = vi.fn(() => 'blob:mock');
    global.URL.createObjectURL = createObjURL;

    render(<ExecutableGraphViewer graph={RESNET_EG} />);
    await userEvent.click(screen.getByRole('button', { name: /export as dot/i }));
    expect(createObjURL).toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// PaperWorkspaceTabs — 6th tab integration
// ---------------------------------------------------------------------------

describe('PaperWorkspaceTabs — Executable Graph tab', () => {
  it('renders the Executable Graph tab button', async () => {
    const { PaperWorkspaceTabs } = await import(
      '@/components/paper-upload/PaperWorkspaceTabs'
    );
    render(
      <PaperWorkspaceTabs
        fullTitle="Test Paper"
        abstract={null}
        architectureType="ResNet"
        depth={50}
        parameterCount={25_000_000}
        flops={4_000_000}
        figures={[]}
        equations={[]}
        textExtractionMethod={undefined}
        pageCount={undefined}
        detectedComponents={undefined}
        moduleCount={0}
        knowledgeGraph={{ nodes: [], edges: [] }}
        architectureBlueprint={null}
        executableGraph={null}
      />,
    );
    expect(
      screen.getByRole('tab', { name: 'Executable Graph' }),
    ).toBeInTheDocument();
  });

  it('switches to Executable Graph tab and shows empty state', async () => {
    const { PaperWorkspaceTabs } = await import(
      '@/components/paper-upload/PaperWorkspaceTabs'
    );
    render(
      <PaperWorkspaceTabs
        fullTitle="Test Paper"
        abstract={null}
        architectureType="ResNet"
        depth={50}
        parameterCount={25_000_000}
        flops={4_000_000}
        figures={[]}
        equations={[]}
        textExtractionMethod={undefined}
        pageCount={undefined}
        detectedComponents={undefined}
        moduleCount={0}
        knowledgeGraph={{ nodes: [], edges: [] }}
        architectureBlueprint={null}
        executableGraph={null}
      />,
    );
    await userEvent.click(screen.getByRole('tab', { name: 'Executable Graph' }));
    expect(
      screen.getByRole('tab', { name: 'Executable Graph', selected: true }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/no executable graph available/i),
    ).toBeInTheDocument();
  });

  it('switches to Executable Graph tab and shows viewer when graph provided', async () => {
    const { PaperWorkspaceTabs } = await import(
      '@/components/paper-upload/PaperWorkspaceTabs'
    );
    render(
      <PaperWorkspaceTabs
        fullTitle="ResNet Paper"
        abstract={null}
        architectureType="ResNet"
        depth={50}
        parameterCount={25_000_000}
        flops={4_000_000}
        figures={[]}
        equations={[]}
        textExtractionMethod={undefined}
        pageCount={undefined}
        detectedComponents={undefined}
        moduleCount={0}
        knowledgeGraph={{ nodes: [], edges: [] }}
        architectureBlueprint={null}
        executableGraph={RESNET_EG}
      />,
    );
    await userEvent.click(screen.getByRole('tab', { name: 'Executable Graph' }));
    expect(screen.getByText('eg-1-resnet')).toBeInTheDocument();
  });
});
