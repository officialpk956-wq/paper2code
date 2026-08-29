import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, test, vi } from 'vitest';

import PapersPage from '@/app/(protected)/papers/page';
import { apiGet, apiPostForm, isLoggedIn } from '@/lib/api';

const push = vi.fn();

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push }),
  useSearchParams: () => new URLSearchParams('tab=workspace'),
}));

vi.mock('@/lib/api', () => ({
  apiGet: vi.fn(),
  apiPostForm: vi.fn(),
  isLoggedIn: vi.fn(),
}));

const mockApiGet = vi.mocked(apiGet);
const mockApiPostForm = vi.mocked(apiPostForm);
const mockIsLoggedIn = vi.mocked(isLoggedIn);

describe('paper upload contract', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockIsLoggedIn.mockReturnValue(true);
    mockApiGet.mockImplementation(async (path) => {
      if (path === '/api/papers') return { summary: {}, papers: [] } as never;
      if (path === '/api/tasks/task-1') {
        return {
          id: 'task-1',
          status: 'completed',
          result: { stage: 'complete', paper_id: 42, generation_status: 'success' },
        } as never;
      }
      throw new Error(`Unexpected GET ${path}`);
    });
    mockApiPostForm.mockResolvedValue({
      task_id: 'task-1',
      status: 'pending',
      poll_url: '/api/tasks/task-1',
      paper_id: null,
      message: 'queued',
    } as never);
  });

  test('requires Terms acceptance before browsing', async () => {
    render(<PapersPage />);
    const browse = await screen.findByRole('button', { name: 'Browse Files' });
    expect(browse).toBeDisabled();

    fireEvent.click(screen.getByRole('button', { name: /drop your pdf here/i }));
    expect(await screen.findByText(/must accept the Terms of Service/i)).toBeInTheDocument();
  });

  test('sends accepted upload, polls task, and opens the paper workspace', async () => {
    const { container } = render(<PapersPage />);
    fireEvent.click(await screen.findByRole('checkbox'));

    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    const file = new File(['%PDF-1.4 fixture'], 'phase1.pdf', { type: 'application/pdf' });
    fireEvent.change(fileInput, { target: { files: [file] } });

    await waitFor(() => expect(mockApiPostForm).toHaveBeenCalledOnce());
    const form = mockApiPostForm.mock.calls[0][1] as FormData;
    expect(form.get('terms_accepted')).toBe('true');
    expect(form.get('visibility')).toBe('private');

    await waitFor(() => expect(mockApiGet).toHaveBeenCalledWith('/api/tasks/task-1'));
    await waitFor(() => expect(push).toHaveBeenCalledWith('/papers/42'));
  });

  test('rejects a PDF larger than 20 MB before calling the backend', async () => {
    const { container } = render(<PapersPage />);
    fireEvent.click(await screen.findByRole('checkbox'));
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    const oversized = new File([new Uint8Array(20 * 1024 * 1024 + 1)], 'large.pdf', {
      type: 'application/pdf',
    });
    fireEvent.change(fileInput, { target: { files: [oversized] } });

    expect(await screen.findByText(/20 MB or smaller/i)).toBeInTheDocument();
    expect(mockApiPostForm).not.toHaveBeenCalled();
  });

  test('stops polling and keeps a retry action when the task fails', async () => {
    mockApiGet.mockImplementation(async (path) => {
      if (path === '/api/papers') return { summary: {}, papers: [] } as never;
      return { id: 'task-1', status: 'failed', result: { stage: 'generating' }, error: 'Generator failed' } as never;
    });
    const { container } = render(<PapersPage />);
    fireEvent.click(await screen.findByRole('checkbox'));
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(fileInput, {
      target: { files: [new File(['%PDF-1.4'], 'failed.pdf', { type: 'application/pdf' })] },
    });

    expect(await screen.findByText('Generator failed')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Retry upload' })).toBeInTheDocument();
    expect(push).not.toHaveBeenCalled();
  });
});
