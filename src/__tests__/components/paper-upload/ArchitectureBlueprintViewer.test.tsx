import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ArchitectureBlueprintViewer } from '@/components/paper-upload/ArchitectureBlueprintViewer';
import type { ArchitectureBlueprint } from '@/components/paper-upload/ArchitectureBlueprintViewer';

// ---------------------------------------------------------------------------
// Test data
// ---------------------------------------------------------------------------

const RESNET_BLUEPRINT: ArchitectureBlueprint = {
  id: 'bp-1-resnet',
  name: 'ResNet',
  paper_id: 1,
  components: [
    { id: 'stem',   name: 'Stem (7×7 Conv)',    type: 'conv',    shape: [1,64,112,112], metadata: {} },
    { id: 'pool1',  name: 'Max Pool',            type: 'pooling', shape: [1,64,56,56],  metadata: {} },
    { id: 'stage1', name: 'Stage 1 ×3 (bottleneck)', type: 'conv', shape: [1,256,56,56], metadata: { repeat_count: 3, has_residual: true } },
    { id: 'head',   name: 'Classifier (→1000)', type: 'head',    shape: [1,1000],       metadata: {} },
  ],
  connections: [
    { source: 'stem',   target: 'pool1',  connection_type: 'sequential' },
    { source: 'pool1',  target: 'stage1', connection_type: 'sequential' },
    { source: 'stage1', target: 'head',   connection_type: 'sequential' },
  ],
  input_shape:  [1, 3, 224, 224],
  output_shape: [1, 1000],
  estimated_parameters: 25_000_000,
  estimated_flops:       4000.0,
  confidence_score:      0.78,
  tensor_flow: [
    { component_id: 'stem',   component_name: 'Stem',    type: 'conv',    input_shape: [1,3,224,224],  output_shape: [1,64,112,112], flops_mflops: 235.9, params_M: 0.047 },
    { component_id: 'pool1',  component_name: 'MaxPool', type: 'pooling', input_shape: [1,64,112,112], output_shape: [1,64,56,56],  flops_mflops: 0,     params_M: 0     },
    { component_id: 'stage1', component_name: 'Stage 1', type: 'conv',    input_shape: [1,64,56,56],   output_shape: [1,256,56,56], flops_mflops: 1200,  params_M: 4.2   },
    { component_id: 'head',   component_name: 'Head',    type: 'head',    input_shape: [1,2048],        output_shape: [1,1000],      flops_mflops: 4.1,   params_M: 2.049 },
  ],
  architecture_href: '/architectures/resnet',
};

const UNET_BLUEPRINT: ArchitectureBlueprint = {
  ...RESNET_BLUEPRINT,
  id: 'bp-2-unet',
  name: 'U-Net',
  components: [
    { id: 'enc1', name: 'Encoder 1', type: 'conv', shape: [1,64,256,256], metadata: {} },
    { id: 'dec1', name: 'Decoder 1', type: 'conv', shape: [1,64,256,256], metadata: {} },
    { id: 'head', name: 'Output',    type: 'head', shape: [1,2,256,256],  metadata: {} },
  ],
  connections: [
    { source: 'enc1', target: 'dec1', connection_type: 'sequential' },
    { source: 'dec1', target: 'head', connection_type: 'sequential' },
    { source: 'enc1', target: 'dec1', connection_type: 'concat' },
  ],
  tensor_flow: [
    { component_id: 'enc1', component_name: 'Enc1', type: 'conv', input_shape: [1,1,256,256], output_shape: [1,64,256,256], flops_mflops: 512, params_M: 0.7 },
    { component_id: 'dec1', component_name: 'Dec1', type: 'conv', input_shape: [1,64,256,256], output_shape: [1,64,256,256], flops_mflops: 512, params_M: 0.7 },
    { component_id: 'head', component_name: 'Head', type: 'head', input_shape: [1,64], output_shape: [1,2], flops_mflops: 0.1, params_M: 0.1 },
  ],
  architecture_href: '/architectures/unet',
  confidence_score: 0.55,
};

// ---------------------------------------------------------------------------
// Empty / null state
// ---------------------------------------------------------------------------

describe('ArchitectureBlueprintViewer — empty state', () => {
  it('renders empty-state message when blueprint is null', () => {
    render(<ArchitectureBlueprintViewer blueprint={null} />);
    expect(screen.getByText(/no architecture blueprint available/i)).toBeInTheDocument();
  });

  it('does not render toolbar when blueprint is null', () => {
    render(<ArchitectureBlueprintViewer blueprint={null} />);
    expect(screen.queryByRole('button', { name: /zoom in/i })).not.toBeInTheDocument();
  });

  it('empty state has role=status', () => {
    render(<ArchitectureBlueprintViewer blueprint={null} />);
    expect(screen.getByRole('status')).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Basic rendering
// ---------------------------------------------------------------------------

describe('ArchitectureBlueprintViewer — rendering', () => {
  it('renders the SVG canvas with role=img', () => {
    const { container } = render(<ArchitectureBlueprintViewer blueprint={RESNET_BLUEPRINT} />);
    const svg = container.querySelector('svg[role="img"]');
    expect(svg).toBeTruthy();
  });

  it('renders confidence score', () => {
    render(<ArchitectureBlueprintViewer blueprint={RESNET_BLUEPRINT} />);
    expect(screen.getByText('78%')).toBeInTheDocument();
  });

  it('renders architecture name in toolbar', () => {
    render(<ArchitectureBlueprintViewer blueprint={RESNET_BLUEPRINT} />);
    expect(screen.getByText(/ResNet/)).toBeInTheDocument();
  });

  it('renders node count badge', () => {
    render(<ArchitectureBlueprintViewer blueprint={RESNET_BLUEPRINT} />);
    expect(
      screen.getByText(`${RESNET_BLUEPRINT.components.length} nodes · ${RESNET_BLUEPRINT.connections.length} edges`)
    ).toBeInTheDocument();
  });

  it('renders zoom controls', () => {
    render(<ArchitectureBlueprintViewer blueprint={RESNET_BLUEPRINT} />);
    expect(screen.getByRole('button', { name: /zoom in/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /zoom out/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /reset view/i })).toBeInTheDocument();
  });

  it('renders each component as an SVG button', () => {
    render(<ArchitectureBlueprintViewer blueprint={RESNET_BLUEPRINT} />);
    const nodeBtns = screen.getAllByRole('button').filter(
      btn => btn.getAttribute('aria-label')?.includes('—')
    );
    expect(nodeBtns.length).toBe(RESNET_BLUEPRINT.components.length);
  });

  it('renders "Open Architecture" link when href is set', () => {
    render(<ArchitectureBlueprintViewer blueprint={RESNET_BLUEPRINT} />);
    expect(screen.getByRole('link', { name: /open architecture/i })).toBeInTheDocument();
  });

  it('does not render architecture link when href is null', () => {
    const noHref = { ...RESNET_BLUEPRINT, architecture_href: null };
    render(<ArchitectureBlueprintViewer blueprint={noHref} />);
    expect(screen.queryByRole('link', { name: /open architecture/i })).not.toBeInTheDocument();
  });

  it('renders connection legend', () => {
    render(<ArchitectureBlueprintViewer blueprint={UNET_BLUEPRINT} />);
    expect(screen.getByText(/Sequential/i)).toBeInTheDocument();
    expect(screen.getByText(/Skip\/Concat/i)).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Node selection
// ---------------------------------------------------------------------------

describe('ArchitectureBlueprintViewer — node selection', () => {
  it('shows info panel on component click', async () => {
    render(<ArchitectureBlueprintViewer blueprint={RESNET_BLUEPRINT} />);
    const btn = screen.getByRole('button', { name: /Stem.*conv/i });
    await userEvent.click(btn);
    expect(screen.getByRole('region', { name: /selected node details/i })).toBeInTheDocument();
  });

  it('shows component name in info panel', async () => {
    render(<ArchitectureBlueprintViewer blueprint={RESNET_BLUEPRINT} />);
    await userEvent.click(screen.getByRole('button', { name: /Stem.*conv/i }));
    expect(screen.getAllByText(/Stem.*Conv|Stem \(7/i).length).toBeGreaterThan(0);
  });

  it('deselects on second click', async () => {
    render(<ArchitectureBlueprintViewer blueprint={RESNET_BLUEPRINT} />);
    const btn = screen.getByRole('button', { name: /Stem.*conv/i });
    await userEvent.click(btn);
    expect(screen.getByRole('region', { name: /selected node details/i })).toBeInTheDocument();
    await userEvent.click(btn);
    expect(screen.queryByRole('region', { name: /selected node details/i })).not.toBeInTheDocument();
  });

  it('marks selected node as aria-pressed=true', async () => {
    render(<ArchitectureBlueprintViewer blueprint={RESNET_BLUEPRINT} />);
    const btn = screen.getByRole('button', { name: /Stem.*conv/i });
    expect(btn.getAttribute('aria-pressed')).toBe('false');
    await userEvent.click(btn);
    expect(btn.getAttribute('aria-pressed')).toBe('true');
  });

  it('supports Enter key selection', () => {
    render(<ArchitectureBlueprintViewer blueprint={RESNET_BLUEPRINT} />);
    const btn = screen.getByRole('button', { name: /Stem.*conv/i });
    btn.focus();
    fireEvent.keyDown(btn, { key: 'Enter' });
    expect(screen.getByRole('region', { name: /selected node details/i })).toBeInTheDocument();
  });

  it('supports Space key selection', () => {
    render(<ArchitectureBlueprintViewer blueprint={RESNET_BLUEPRINT} />);
    const btn = screen.getByRole('button', { name: /Stem.*conv/i });
    btn.focus();
    fireEvent.keyDown(btn, { key: ' ' });
    expect(screen.getByRole('region', { name: /selected node details/i })).toBeInTheDocument();
  });

  it('info panel has close button', async () => {
    render(<ArchitectureBlueprintViewer blueprint={RESNET_BLUEPRINT} />);
    await userEvent.click(screen.getByRole('button', { name: /Stem.*conv/i }));
    expect(screen.getByRole('button', { name: /close details panel/i })).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Zoom controls
// ---------------------------------------------------------------------------

describe('ArchitectureBlueprintViewer — zoom', () => {
  it('starts at 100%', () => {
    render(<ArchitectureBlueprintViewer blueprint={RESNET_BLUEPRINT} />);
    expect(screen.getByText('100%')).toBeInTheDocument();
  });

  it('increments to 120% on zoom in', async () => {
    render(<ArchitectureBlueprintViewer blueprint={RESNET_BLUEPRINT} />);
    await userEvent.click(screen.getByRole('button', { name: /zoom in/i }));
    expect(screen.getByText('120%')).toBeInTheDocument();
  });

  it('decrements to 80% on zoom out', async () => {
    render(<ArchitectureBlueprintViewer blueprint={RESNET_BLUEPRINT} />);
    await userEvent.click(screen.getByRole('button', { name: /zoom out/i }));
    expect(screen.getByText('80%')).toBeInTheDocument();
  });

  it('resets to 100% on reset view', async () => {
    render(<ArchitectureBlueprintViewer blueprint={RESNET_BLUEPRINT} />);
    await userEvent.click(screen.getByRole('button', { name: /zoom in/i }));
    await userEvent.click(screen.getByRole('button', { name: /reset view/i }));
    expect(screen.getByText('100%')).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// PaperWorkspaceTabs — 5th tab integration
// ---------------------------------------------------------------------------

describe('PaperWorkspaceTabs — Architecture Blueprint tab', () => {
  it('renders the Architecture Blueprint tab button', async () => {
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
        knowledgeGraph={{ nodes: [], edges: [] }}
        architectureBlueprint={null}
      />
    );
    expect(screen.getByRole('tab', { name: 'Architecture Blueprint' })).toBeInTheDocument();
  });

  it('switches to Architecture Blueprint tab and shows empty state', async () => {
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
        knowledgeGraph={{ nodes: [], edges: [] }}
        architectureBlueprint={null}
      />
    );
    await userEvent.click(screen.getByRole('tab', { name: 'Architecture Blueprint' }));
    expect(screen.getByRole('tab', { name: 'Architecture Blueprint', selected: true })).toBeInTheDocument();
    expect(screen.getByText(/no architecture blueprint available/i)).toBeInTheDocument();
  });

  it('switches to Architecture Blueprint tab and shows viewer when blueprint is provided', async () => {
    const { PaperWorkspaceTabs } = await import('@/components/paper-upload/PaperWorkspaceTabs');
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
        architectureBlueprint={RESNET_BLUEPRINT}
      />
    );
    await userEvent.click(screen.getByRole('tab', { name: 'Architecture Blueprint' }));
    // Confidence score should appear
    expect(screen.getByText('78%')).toBeInTheDocument();
  });
});
