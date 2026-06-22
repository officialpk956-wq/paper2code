import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { PaperKnowledgeGraph } from '@/components/paper-upload/PaperKnowledgeGraph';
import type { KGNode, KGEdge } from '@/components/paper-upload/PaperKnowledgeGraph';

// ---------------------------------------------------------------------------
// Test data helpers
// ---------------------------------------------------------------------------

const PAPER_NODE: KGNode = {
  id: 'paper-resnet-paper',
  name: 'ResNet Paper',
  type: 'paper',
  href: null,
};

const ARCH_NODE: KGNode = {
  id: 'arch-resnet',
  name: 'ResNet',
  type: 'architecture',
  href: '/architectures/resnet',
};

const CONCEPT_NODE: KGNode = {
  id: 'concept-residual-connection',
  name: 'Residual Connection',
  type: 'concept',
  href: null,
};

const DATASET_NODE: KGNode = {
  id: 'dataset-imagenet',
  name: 'ImageNet',
  type: 'dataset',
  href: null,
};

const METRIC_NODE: KGNode = {
  id: 'metric-accuracy',
  name: 'Accuracy',
  type: 'metric',
  href: null,
};

const EDGES: KGEdge[] = [
  { source: 'paper-resnet-paper', target: 'arch-resnet', relation: 'introduces' },
  { source: 'paper-resnet-paper', target: 'dataset-imagenet', relation: 'evaluates_on' },
  { source: 'paper-resnet-paper', target: 'metric-accuracy', relation: 'reports' },
  { source: 'arch-resnet', target: 'concept-residual-connection', relation: 'uses' },
];

const ALL_NODES: KGNode[] = [PAPER_NODE, ARCH_NODE, CONCEPT_NODE, DATASET_NODE, METRIC_NODE];

// ---------------------------------------------------------------------------
// Empty state
// ---------------------------------------------------------------------------

describe('PaperKnowledgeGraph — empty state', () => {
  it('shows empty-state message when no nodes provided', () => {
    render(<PaperKnowledgeGraph nodes={[]} edges={[]} />);
    expect(
      screen.getByText(/no knowledge graph data extracted/i),
    ).toBeInTheDocument();
  });

  it('does not render the toolbar when empty', () => {
    render(<PaperKnowledgeGraph nodes={[]} edges={[]} />);
    expect(screen.queryByRole('button', { name: /zoom in/i })).not.toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Basic rendering with nodes
// ---------------------------------------------------------------------------

describe('PaperKnowledgeGraph — rendering', () => {
  it('renders the SVG canvas', () => {
    const { container } = render(
      <PaperKnowledgeGraph nodes={ALL_NODES} edges={EDGES} />,
    );
    expect(container.querySelector('svg')).toBeTruthy();
  });

  it('renders node count in toolbar badge', () => {
    render(<PaperKnowledgeGraph nodes={ALL_NODES} edges={EDGES} />);
    expect(screen.getByText(`${ALL_NODES.length} nodes · ${EDGES.length} edges`)).toBeInTheDocument();
  });

  it('renders zoom-in, zoom-out and reset buttons', () => {
    render(<PaperKnowledgeGraph nodes={ALL_NODES} edges={EDGES} />);
    expect(screen.getByRole('button', { name: /zoom in/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /zoom out/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /reset view/i })).toBeInTheDocument();
  });

  it('renders accessible SVG with role=img and aria-label', () => {
    const { container } = render(
      <PaperKnowledgeGraph nodes={ALL_NODES} edges={EDGES} />,
    );
    const svg = container.querySelector('svg[role="img"]');
    expect(svg).toBeTruthy();
    expect(svg?.getAttribute('aria-label')).toBeTruthy();
  });

  it('renders each node as a group with role=button', () => {
    render(<PaperKnowledgeGraph nodes={ALL_NODES} edges={EDGES} />);
    const nodeButtons = screen.getAllByRole('button').filter(
      (btn) => btn.getAttribute('aria-label')?.includes('—'),
    );
    expect(nodeButtons.length).toBe(ALL_NODES.length);
  });

  it('renders legend entries for each unique node type present', () => {
    render(<PaperKnowledgeGraph nodes={ALL_NODES} edges={EDGES} />);
    // Legend contains one entry per unique type (use getAllByText — types also appear in node labels)
    const types = ['paper', 'architecture', 'concept', 'dataset', 'metric'];
    for (const type of types) {
      expect(screen.getAllByText(type, { exact: false }).length).toBeGreaterThan(0);
    }
  });
});

// ---------------------------------------------------------------------------
// Node selection
// ---------------------------------------------------------------------------

describe('PaperKnowledgeGraph — node selection', () => {
  it('shows info panel with node name when node is clicked', async () => {
    render(<PaperKnowledgeGraph nodes={ALL_NODES} edges={EDGES} />);
    const resnetBtn = screen.getByRole('button', { name: /ResNet — architecture/i });
    await userEvent.click(resnetBtn);
    // Info panel should appear with the node name
    expect(screen.getByRole('region', { name: /selected node details/i })).toBeInTheDocument();
    expect(screen.getAllByText('ResNet').length).toBeGreaterThan(0);
  });

  it('shows "Open →" link when the selected node has an href', async () => {
    render(<PaperKnowledgeGraph nodes={ALL_NODES} edges={EDGES} />);
    await userEvent.click(screen.getByRole('button', { name: /ResNet — architecture/i }));
    expect(screen.getByRole('link', { name: /open/i })).toBeInTheDocument();
  });

  it('hides "Open →" link when node has no href', async () => {
    render(<PaperKnowledgeGraph nodes={ALL_NODES} edges={EDGES} />);
    await userEvent.click(screen.getByRole('button', { name: /Residual Connection — concept/i }));
    expect(screen.queryByRole('link', { name: /open/i })).not.toBeInTheDocument();
  });

  it('deselects node on second click (toggles off)', async () => {
    render(<PaperKnowledgeGraph nodes={ALL_NODES} edges={EDGES} />);
    const btn = screen.getByRole('button', { name: /ResNet — architecture/i });
    await userEvent.click(btn);
    expect(screen.getByRole('region', { name: /selected node details/i })).toBeInTheDocument();
    await userEvent.click(btn);
    expect(screen.queryByRole('region', { name: /selected node details/i })).not.toBeInTheDocument();
  });

  it('marks selected node button as aria-pressed=true', async () => {
    render(<PaperKnowledgeGraph nodes={ALL_NODES} edges={EDGES} />);
    const btn = screen.getByRole('button', { name: /ResNet — architecture/i });
    expect(btn.getAttribute('aria-pressed')).toBe('false');
    await userEvent.click(btn);
    expect(btn.getAttribute('aria-pressed')).toBe('true');
  });

  it('supports keyboard selection via Enter key', async () => {
    render(<PaperKnowledgeGraph nodes={ALL_NODES} edges={EDGES} />);
    const btn = screen.getByRole('button', { name: /ResNet — architecture/i });
    btn.focus();
    fireEvent.keyDown(btn, { key: 'Enter' });
    expect(screen.getByRole('region', { name: /selected node details/i })).toBeInTheDocument();
  });

  it('supports keyboard selection via Space key', async () => {
    render(<PaperKnowledgeGraph nodes={ALL_NODES} edges={EDGES} />);
    const btn = screen.getByRole('button', { name: /ResNet — architecture/i });
    btn.focus();
    fireEvent.keyDown(btn, { key: ' ' });
    expect(screen.getByRole('region', { name: /selected node details/i })).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Zoom controls
// ---------------------------------------------------------------------------

describe('PaperKnowledgeGraph — zoom controls', () => {
  it('shows 100% zoom label initially', () => {
    render(<PaperKnowledgeGraph nodes={ALL_NODES} edges={EDGES} />);
    expect(screen.getByText('100%')).toBeInTheDocument();
  });

  it('increases zoom percentage when zoom in is clicked', async () => {
    render(<PaperKnowledgeGraph nodes={ALL_NODES} edges={EDGES} />);
    await userEvent.click(screen.getByRole('button', { name: /zoom in/i }));
    expect(screen.getByText('120%')).toBeInTheDocument();
  });

  it('decreases zoom percentage when zoom out is clicked', async () => {
    render(<PaperKnowledgeGraph nodes={ALL_NODES} edges={EDGES} />);
    await userEvent.click(screen.getByRole('button', { name: /zoom out/i }));
    expect(screen.getByText('80%')).toBeInTheDocument();
  });

  it('resets zoom to 100% after reset view click', async () => {
    render(<PaperKnowledgeGraph nodes={ALL_NODES} edges={EDGES} />);
    await userEvent.click(screen.getByRole('button', { name: /zoom in/i }));
    await userEvent.click(screen.getByRole('button', { name: /zoom in/i }));
    await userEvent.click(screen.getByRole('button', { name: /reset view/i }));
    expect(screen.getByText('100%')).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// PaperWorkspaceTabs integration
// ---------------------------------------------------------------------------

describe('PaperWorkspaceTabs — tab switching', () => {
  it('renders the Overview tab by default', async () => {
    const { PaperWorkspaceTabs } = await import('@/components/paper-upload/PaperWorkspaceTabs');
    render(
      <PaperWorkspaceTabs
        fullTitle="Test Paper"
        abstract="A test abstract."
        architectureType="Transformer"
        depth={12}
        parameterCount={110_000_000}
        flops={3600000}
        figures={[]}
        equations={[]}
        textExtractionMethod="pdfplumber"
        pageCount={8}
        detectedComponents={['architecture']}
        moduleCount={5}
        knowledgeGraph={{ nodes: [], edges: [] }}
      />,
    );
    expect(screen.getByRole('tab', { name: 'Overview', selected: true })).toBeInTheDocument();
    expect(screen.getByText('Architecture')).toBeInTheDocument();
  });

  it('switches to Knowledge Graph tab on click', async () => {
    const { PaperWorkspaceTabs } = await import('@/components/paper-upload/PaperWorkspaceTabs');
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
        knowledgeGraph={{ nodes: ALL_NODES, edges: EDGES }}
      />,
    );

    await userEvent.click(screen.getByRole('tab', { name: 'Knowledge Graph' }));
    expect(screen.getByRole('tab', { name: 'Knowledge Graph', selected: true })).toBeInTheDocument();
    // The graph toolbar / empty-state should be visible
    expect(
      screen.getByText(`${ALL_NODES.length} nodes · ${EDGES.length} edges`),
    ).toBeInTheDocument();
  });
});
