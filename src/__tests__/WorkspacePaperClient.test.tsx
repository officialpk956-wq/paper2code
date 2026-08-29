import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import WorkspacePaperClient from '../app/(protected)/papers/[id]/WorkspacePaperClient';
import { apiGet, apiPost } from '@/lib/api';
import { vi, describe, beforeEach, test, expect } from 'vitest';

vi.mock('@/lib/api');
vi.mock('next/dynamic', () => ({
  default: () => {
    const DynamicComponent = (props: any) => {
      return <textarea 
        data-testid="monaco-editor-stub"
        value={props.value}
        onChange={e => props.onChange(e.target.value)}
      />;
    };
    return DynamicComponent;
  }
}));

vi.mock('@monaco-editor/react', () => ({
  __esModule: true,
  loader: { config: vi.fn() }
}));

const mockApiGet = apiGet as any;
const mockApiPost = apiPost as any;

describe('WorkspacePaperClient - Implement Tab', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockApiGet.mockImplementation((url) => {
      if (url.includes('/api/papers/5/executable-graph')) {
        return Promise.resolve({
          status: 'success',
          language: 'python',
          code: "print('generated code ready')",
          verification_report: {
            passed: true,
            entrypoint_class: 'TestModel',
            input_shape: [1, 3, 224, 224],
            output_shape: [1, 1000],
          },
        });
      }
      if (url.includes('/api/papers/5/implement')) {
        return Promise.resolve({
          status: 'ok',
          starter_code: 'import torch\nclass TestModel(nn.Module):\n    pass',
          shapes: { 'layer1': { input: [1, 3, 224, 224], output: [1, 64, 112, 112] } },
          layer_docs: { 'layer1': { summary: 'A simple layer' } }
        });
      }
      if (url.includes('/api/papers/5')) {
        return Promise.resolve({
          title: 'Test Paper',
          authors: 'John Doe',
          year: 2024,
          abstract: 'Test abstract',
          color: '#ffffff',
          contributions: []
        });
      }
      return Promise.resolve({});
    });
  });

  test('Implement tab renders Monaco editor and fetches data', async () => {
    render(<WorkspacePaperClient id="5" />);
    
    // Wait for initial load
    await waitFor(() => {
      expect(screen.getByText('Test Paper')).toBeInTheDocument();
    });

    // Click Implement tab
    const implementTab = screen.getByRole('button', { name: /Implement/i });
    fireEvent.click(implementTab);

    // Wait for the starter code to load into the text area
    await waitFor(() => {
      const editorStub = screen.getByTestId('monaco-editor-stub');
      expect(editorStub).toHaveValue('import torch\nclass TestModel(nn.Module):\n    pass');
    });

    // Verify shapes and docs appear
    expect(screen.getByText('Tensor Shapes')).toBeInTheDocument();
    expect(screen.getAllByText('layer1').length).toBeGreaterThan(0);
    expect(screen.getByText('(1, 3, 224, 224) → (1, 64, 112, 112)')).toBeInTheDocument();
    expect(screen.getByText('A simple layer')).toBeInTheDocument();
    
    // Verify run button is disabled
    const runBtn = screen.getByRole('button', { name: /Run in Sandbox/i });
    expect(runBtn).toBeDisabled();
  });

  test('Executable tab renders persisted generated code and runs it in the sandbox', async () => {
    mockApiPost.mockResolvedValue({ stdout: 'generated code ready\n', stderr: '', exit_code: 0 });
    render(<WorkspacePaperClient id="5" />);

    await waitFor(() => expect(screen.getByText('Test Paper')).toBeInTheDocument());
    fireEvent.click(screen.getByRole('button', { name: 'Executable' }));

    expect(await screen.findByText(/print\('generated code ready'\)/)).toBeInTheDocument();
    expect(screen.getByText('Phase 1 verified')).toBeInTheDocument();
    expect(screen.getByText(/Entrypoint:/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Run' }));
    await waitFor(() => {
      expect(mockApiPost).toHaveBeenCalledWith('/api/dojo/execute', {
        code: "print('generated code ready')",
        stdin: '',
      });
    });
    expect(await screen.findByText('generated code ready')).toBeInTheDocument();
  });
});
