import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { Problem } from '@/data/problems';

// Mock child components to isolate DojoProblemPage behaviour
vi.mock('@/components/dojo/DojoEditor', () => ({
  DojoEditor: ({
    onRun,
    onSubmit,
    runState,
    submitState,
  }: {
    onRun: (c: string) => void;
    onSubmit: (c: string) => void;
    runState: string;
    submitState: string;
  }) => (
    <div data-testid="dojo-editor">
      <button data-testid="mock-run-btn" onClick={() => onRun('def f(): pass')}>
        Run
      </button>
      <button data-testid="mock-submit-btn" onClick={() => onSubmit('def f(): pass')}>
        Submit
      </button>
      <span data-testid="run-state">{runState}</span>
      <span data-testid="submit-state">{submitState}</span>
    </div>
  ),
}));

vi.mock('@/components/dojo/TheoryPanel', () => ({
  TheoryPanel: ({
    activeTab,
    onTabChange,
  }: {
    activeTab: string;
    onTabChange: (t: string) => void;
  }) => (
    <div data-testid="theory-panel" data-tab={activeTab}>
      <button data-testid="theory-tab-btn" onClick={() => onTabChange('theory')}>
        Switch to theory
      </button>
    </div>
  ),
}));

vi.mock('@/components/dojo/TestResultPanel', () => ({
  TestResultPanel: ({ mode }: { mode: string }) => (
    <div data-testid="test-result-panel" data-mode={mode} />
  ),
}));

vi.mock('@/components/dojo/SubmissionHistory', () => ({
  SubmissionHistory: ({ slug }: { slug: string }) => (
    <div data-testid="submission-history" data-slug={slug} />
  ),
}));

vi.mock('@/lib/dojo', () => ({
  saveSubmission: vi.fn(),
  getProblemProgress: vi.fn().mockReturnValue({
    slug: 'matrix-multiply',
    solved: false,
    attempted: false,
    lastSubmittedAt: null,
    bestRuntimeMs: null,
    submissionCount: 0,
  }),
  generateSubmissionId: vi.fn().mockReturnValue('sub_test_123'),
}));

const MOCK_PROBLEM: Problem = {
  id: 'la-1',
  slug: 'matrix-multiply',
  title: 'Implement Matrix Multiplication',
  category: 'linear-algebra',
  difficulty: 'beginner',
  estimatedTime: 20,
  description: 'Implement matrix multiplication from scratch.',
  tags: ['numpy', 'linear-algebra'],
  relatedArchitectures: [],
  relatedPapers: [],
  relatedMath: [],
  learningPoints: ['Matrix multiplication', 'NumPy broadcasting'],
  pythonTemplate: 'def matrix_multiply(A, B):\n    pass',
  testCases: [
    { input: { A: [[1, 0], [0, 1]], B: [[1, 2], [3, 4]] }, output: [[1, 2], [3, 4]], visible: true },
  ],
  hints: { hint1: 'Think rows × columns', hint2: '', hint3: '', solutionOutline: '', fullSolution: '' },
  explanation: {
    stepByStep: '',
    complexityAnalysis: 'O(n³)',
    commonMistakes: '',
    interviewDiscussion: '',
  },
};

describe('DojoProblemPage', () => {
  beforeEach(() => {
    global.fetch = vi.fn();
  });

  it('renders the problem title in the header', async () => {
    const { DojoProblemPage } = await import('@/components/dojo/DojoProblemPage');
    render(<DojoProblemPage problem={MOCK_PROBLEM} />);
    expect(screen.getByText('Implement Matrix Multiplication')).toBeInTheDocument();
  });

  it('renders the back link with correct aria-label', async () => {
    const { DojoProblemPage } = await import('@/components/dojo/DojoProblemPage');
    render(<DojoProblemPage problem={MOCK_PROBLEM} />);
    expect(screen.getByLabelText('Back to Dojo')).toBeInTheDocument();
    expect(screen.getByLabelText('Back to Dojo')).toHaveAttribute('href', '/dojo');
  });

  it('renders the difficulty badge', async () => {
    const { DojoProblemPage } = await import('@/components/dojo/DojoProblemPage');
    render(<DojoProblemPage problem={MOCK_PROBLEM} />);
    expect(screen.getByText('Easy')).toBeInTheDocument();
  });

  it('renders estimated time', async () => {
    const { DojoProblemPage } = await import('@/components/dojo/DojoProblemPage');
    render(<DojoProblemPage problem={MOCK_PROBLEM} />);
    expect(screen.getByText('~20 min')).toBeInTheDocument();
  });

  it('renders the editor component', async () => {
    const { DojoProblemPage } = await import('@/components/dojo/DojoProblemPage');
    render(<DojoProblemPage problem={MOCK_PROBLEM} />);
    expect(screen.getByTestId('dojo-editor')).toBeInTheDocument();
  });

  it('renders the theory panel', async () => {
    const { DojoProblemPage } = await import('@/components/dojo/DojoProblemPage');
    render(<DojoProblemPage problem={MOCK_PROBLEM} />);
    expect(screen.getByTestId('theory-panel')).toBeInTheDocument();
  });

  it('switches to history tab to show SubmissionHistory', async () => {
    const user = userEvent.setup();
    const { DojoProblemPage } = await import('@/components/dojo/DojoProblemPage');
    render(<DojoProblemPage problem={MOCK_PROBLEM} />);

    // Trigger a submit that returns accepted — mock fetch response
    vi.mocked(global.fetch).mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          results: [{ index: 0, passed: true, actual: [[1, 2], [3, 4]], expected: [[1, 2], [3, 4]], runtime_ms: 0.5 }],
          status: 'accepted',
          passedTests: 1,
          totalTests: 1,
          runtimeMs: 0.5,
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
    );

    await user.click(screen.getByTestId('mock-submit-btn'));

    await waitFor(() => {
      expect(screen.getByTestId('submission-history')).toBeInTheDocument();
    });
  });

  it('shows result panel in run mode by default', async () => {
    const { DojoProblemPage } = await import('@/components/dojo/DojoProblemPage');
    render(<DojoProblemPage problem={MOCK_PROBLEM} />);
    expect(screen.getByTestId('test-result-panel')).toHaveAttribute('data-mode', 'run');
  });
});
