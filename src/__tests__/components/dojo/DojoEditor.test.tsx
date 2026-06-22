import { describe, it, expect, vi } from 'vitest';
import { render, screen, act, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Mock next/dynamic before importing anything that uses it
vi.mock('next/dynamic', () => ({
  default: (_importFn: unknown, _opts?: unknown) => {
    const MockEditor = ({
      value,
      onChange,
    }: {
      value?: string;
      onChange?: (v: string | undefined) => void;
    }) => (
      <textarea
        data-testid="monaco-editor"
        value={value ?? ''}
        onChange={(e) => onChange?.(e.target.value)}
      />
    );
    return MockEditor;
  },
}));

import { DojoEditor } from '@/components/dojo/DojoEditor';

describe('DojoEditor', () => {
  it('renders the toolbar with Python label', () => {
    render(
      <DojoEditor initialCode="def f(): pass" onRun={vi.fn()} onSubmit={vi.fn()} runState="idle" submitState="idle" />,
    );
    expect(screen.getByText('Python 3')).toBeInTheDocument();
  });

  it('renders Run and Submit buttons', () => {
    render(
      <DojoEditor initialCode="" onRun={vi.fn()} onSubmit={vi.fn()} runState="idle" submitState="idle" />,
    );
    expect(screen.getByText('▶ Run')).toBeInTheDocument();
    expect(screen.getByText('↑ Submit')).toBeInTheDocument();
  });

  it('calls onRun with current code when Run clicked', async () => {
    const user = userEvent.setup();
    const onRun = vi.fn();
    render(
      <DojoEditor initialCode="def solution(): return 42" onRun={onRun} onSubmit={vi.fn()} runState="idle" submitState="idle" />,
    );
    await user.click(screen.getByText('▶ Run'));
    expect(onRun).toHaveBeenCalledWith('def solution(): return 42');
  });

  it('calls onSubmit with current code when Submit clicked', async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <DojoEditor initialCode="def solution(): pass" onRun={vi.fn()} onSubmit={onSubmit} runState="idle" submitState="idle" />,
    );
    await user.click(screen.getByText('↑ Submit'));
    expect(onSubmit).toHaveBeenCalledWith('def solution(): pass');
  });

  it('shows Running text when runState=running', () => {
    render(
      <DojoEditor initialCode="" onRun={vi.fn()} onSubmit={vi.fn()} runState="running" submitState="idle" />,
    );
    expect(screen.getByText('Running…')).toBeInTheDocument();
    expect(screen.queryByText('▶ Run')).not.toBeInTheDocument();
  });

  it('shows Submitting text when submitState=running', () => {
    render(
      <DojoEditor initialCode="" onRun={vi.fn()} onSubmit={vi.fn()} runState="idle" submitState="running" />,
    );
    expect(screen.getByText('Submitting…')).toBeInTheDocument();
    expect(screen.queryByText('↑ Submit')).not.toBeInTheDocument();
  });

  it('disables buttons when runState=running', () => {
    render(
      <DojoEditor initialCode="" onRun={vi.fn()} onSubmit={vi.fn()} runState="running" submitState="idle" />,
    );
    screen.getAllByRole('button').forEach((btn) => expect(btn).toBeDisabled());
  });

  it('disables buttons when submitState=running', () => {
    render(
      <DojoEditor initialCode="" onRun={vi.fn()} onSubmit={vi.fn()} runState="idle" submitState="running" />,
    );
    screen.getAllByRole('button').forEach((btn) => expect(btn).toBeDisabled());
  });

  it('code state updates when editor content changes', async () => {
    const user = userEvent.setup();
    const onRun = vi.fn();
    render(
      <DojoEditor initialCode="# old code" onRun={onRun} onSubmit={vi.fn()} runState="idle" submitState="idle" />,
    );
    const editor = screen.getByTestId('monaco-editor');
    await user.clear(editor);
    await user.type(editor, '# new code');
    await user.click(screen.getByText('▶ Run'));
    expect(onRun).toHaveBeenCalledWith('# new code');
  });

  it('renders the keyboard shortcut hint', () => {
    render(
      <DojoEditor initialCode="" onRun={vi.fn()} onSubmit={vi.fn()} runState="idle" submitState="idle" />,
    );
    expect(screen.getByText('Ctrl+Enter to run')).toBeInTheDocument();
  });

  describe('CDN Fallback', () => {
    it('falls back to a textarea if Monaco takes too long to load', async () => {
      // For this test, we want to prevent the mock Monaco from mounting immediately,
      // but our current mock returns a functional component that mounts right away.
      // However, DojoEditor's handleEditorMount is what sets editorReady.
      // Since our mock textarea doesn't call onMount, editorReady stays false!
      
      vi.useFakeTimers();
      
      const onRun = vi.fn();
      render(
        <DojoEditor initialCode="fallback test" onRun={onRun} onSubmit={vi.fn()} runState="idle" submitState="idle" />
      );

      // Initially, mock monaco is rendered (because we mocked next/dynamic to just return it)
      // but onMount was never called.
      expect(screen.queryByText(/Code editor unavailable/i)).not.toBeInTheDocument();

      // Fast forward 10s
      act(() => {
        vi.advanceTimersByTime(10000);
      });

      // Now the fallback should appear
      expect(screen.getByText(/Code editor unavailable/i)).toBeInTheDocument();
      
      // The fallback is a textarea with the code
      const fallbackEditor = screen.getByRole('textbox');
      expect(fallbackEditor).toHaveValue('fallback test');

      // Typing in it and running works
      fireEvent.change(fallbackEditor, { target: { value: 'new fallback code' } });
      fireEvent.click(screen.getByText('▶ Run'));
      expect(onRun).toHaveBeenCalledWith('new fallback code');

      vi.useRealTimers();
    });
  });
});
