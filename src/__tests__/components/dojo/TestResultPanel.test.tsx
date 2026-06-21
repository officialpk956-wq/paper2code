import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { TestResultPanel } from '@/components/dojo/TestResultPanel';
import type { TestResult } from '@/lib/dojo';

function makeResult(passed: boolean, index = 0): TestResult {
  return {
    index,
    passed,
    input: { x: index + 1 },
    expected: index + 1,
    actual: passed ? index + 1 : index + 2,
    runtime_ms: 0.5,
  };
}

describe('TestResultPanel', () => {
  it('shows running message in run mode', () => {
    render(<TestResultPanel results={[]} isRunning={true} mode="run" />);
    expect(screen.getByText('Running against sample cases…')).toBeInTheDocument();
  });

  it('shows running message in submit mode', () => {
    render(<TestResultPanel results={[]} isRunning={true} mode="submit" />);
    expect(screen.getByText('Running all test cases…')).toBeInTheDocument();
  });

  it('shows error panel when error prop provided', () => {
    render(<TestResultPanel results={[]} error="SyntaxError: invalid syntax" mode="run" />);
    expect(screen.getByText('SyntaxError: invalid syntax')).toBeInTheDocument();
    expect(screen.getByText('Execution Error')).toBeInTheDocument();
  });

  it('shows empty state when no results', () => {
    render(<TestResultPanel results={[]} mode="run" />);
    expect(screen.getByText('Run your code to see test results')).toBeInTheDocument();
  });

  it('renders passed test case with check mark', () => {
    render(<TestResultPanel results={[makeResult(true)]} mode="run" />);
    expect(screen.getByText('Case 1')).toBeInTheDocument();
    expect(screen.getByText('✓')).toBeInTheDocument();
  });

  it('renders failed test case with cross mark', () => {
    render(<TestResultPanel results={[makeResult(false)]} mode="run" />);
    expect(screen.getByText('Case 1')).toBeInTheDocument();
    expect(screen.getByText('✗')).toBeInTheDocument();
  });

  it('renders multiple test cases', () => {
    render(<TestResultPanel results={[makeResult(true, 0), makeResult(false, 1)]} mode="run" />);
    expect(screen.getByText('Case 1')).toBeInTheDocument();
    expect(screen.getByText('Case 2')).toBeInTheDocument();
  });

  it('shows verdict banner in submit mode when status provided', () => {
    render(
      <TestResultPanel
        results={[makeResult(true)]}
        mode="submit"
        status="accepted"
        passedTests={1}
        totalTests={1}
      />,
    );
    expect(screen.getByText('Accepted')).toBeInTheDocument();
    expect(screen.getByText('1/1 tests passed')).toBeInTheDocument();
  });

  it('shows wrong answer verdict', () => {
    render(
      <TestResultPanel
        results={[makeResult(false)]}
        mode="submit"
        status="wrong_answer"
        passedTests={0}
        totalTests={1}
      />,
    );
    expect(screen.getByText('Wrong Answer')).toBeInTheDocument();
    expect(screen.getByText('0/1 tests passed')).toBeInTheDocument();
  });

  it('shows timeout verdict', () => {
    render(
      <TestResultPanel
        results={[makeResult(false)]}
        mode="submit"
        status="timeout"
        passedTests={0}
        totalTests={5}
      />,
    );
    expect(screen.getByText('Time Limit Exceeded')).toBeInTheDocument();
  });

  it('shows runtime in verdict banner when provided', () => {
    render(
      <TestResultPanel
        results={[makeResult(true)]}
        mode="submit"
        status="accepted"
        passedTests={1}
        totalTests={1}
        runtimeMs={42}
      />,
    );
    expect(screen.getByText('42 ms')).toBeInTheDocument();
  });

  it('does not show verdict banner when no status provided', () => {
    render(
      <TestResultPanel
        results={[makeResult(true)]}
        mode="run"
        // no status prop
      />,
    );
    expect(screen.queryByText('Accepted')).not.toBeInTheDocument();
    expect(screen.queryByText('Wrong Answer')).not.toBeInTheDocument();
  });
});
