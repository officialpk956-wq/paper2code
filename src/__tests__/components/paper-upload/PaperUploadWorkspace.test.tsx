import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const pushMock = vi.fn();

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: pushMock }),
}));

describe('PaperUploadWorkspace', () => {
  beforeEach(() => {
    pushMock.mockReset();
    global.fetch = vi.fn();
  });

  it('renders the upload form with title input and buttons', async () => {
    const { PaperUploadWorkspace } = await import('@/components/paper-upload/PaperUploadWorkspace');
    render(<PaperUploadWorkspace />);

    expect(screen.getByLabelText('Optional paper title')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /choose file/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /generate workspace/i })).toBeInTheDocument();
  });

  it('shows validation error when no file is selected', async () => {
    const { PaperUploadWorkspace } = await import('@/components/paper-upload/PaperUploadWorkspace');
    render(<PaperUploadWorkspace />);

    await userEvent.click(screen.getByRole('button', { name: /generate workspace/i }));
    expect(screen.getByText('Choose a PDF before uploading.')).toBeInTheDocument();
  });

  it('updates file summary when a file is selected', async () => {
    const { PaperUploadWorkspace } = await import('@/components/paper-upload/PaperUploadWorkspace');
    const user = userEvent.setup();
    const { container } = render(<PaperUploadWorkspace />);

    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(fileInput, new File(['%PDF-1.7'], 'myresearch.pdf', { type: 'application/pdf' }));

    expect(screen.getByText(/myresearch\.pdf/i)).toBeInTheDocument();
  });

  it('submits PDF and navigates to generated workspace on success', async () => {
    vi.mocked(global.fetch).mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          paper_id: 7,
          title: 'Uploaded Paper',
          report: { nodes: 10, edges: 9, modules: 4, figure_count: 2, equation_count: 5 },
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
    );

    const { PaperUploadWorkspace } = await import('@/components/paper-upload/PaperUploadWorkspace');
    const user = userEvent.setup();
    const { container } = render(<PaperUploadWorkspace />);

    await user.type(screen.getByLabelText('Optional paper title'), 'Uploaded Paper');
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(fileInput, new File(['%PDF-1.7'], 'upload.pdf', { type: 'application/pdf' }));
    await user.click(screen.getByRole('button', { name: /generate workspace/i }));

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        '/api/papers/upload',
        expect.objectContaining({ method: 'POST' }),
      );
      expect(pushMock).toHaveBeenCalledWith('/papers/upload/7');
    });
  });

  it('shows backend error message on non-ok response', async () => {
    vi.mocked(global.fetch).mockResolvedValueOnce(
      new Response(
        JSON.stringify({ error: 'File exceeds 20MB limit.' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } },
      ),
    );

    const { PaperUploadWorkspace } = await import('@/components/paper-upload/PaperUploadWorkspace');
    const user = userEvent.setup();
    const { container } = render(<PaperUploadWorkspace />);

    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(fileInput, new File(['%PDF-1.7'], 'huge.pdf', { type: 'application/pdf' }));
    await user.click(screen.getByRole('button', { name: /generate workspace/i }));

    await waitFor(() => {
      expect(screen.getByText('File exceeds 20MB limit.')).toBeInTheDocument();
    });
  });

  it('shows generic error message when fetch throws', async () => {
    vi.mocked(global.fetch).mockRejectedValueOnce(new Error('Network failure'));

    const { PaperUploadWorkspace } = await import('@/components/paper-upload/PaperUploadWorkspace');
    const user = userEvent.setup();
    const { container } = render(<PaperUploadWorkspace />);

    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(fileInput, new File(['%PDF-1.7'], 'test.pdf', { type: 'application/pdf' }));
    await user.click(screen.getByRole('button', { name: /generate workspace/i }));

    await waitFor(() => {
      expect(screen.getByText('Network failure')).toBeInTheDocument();
    });
  });

  it('disables the submit button while loading', async () => {
    let resolveUpload!: (v: Response) => void;
    vi.mocked(global.fetch).mockReturnValueOnce(
      new Promise<Response>((resolve) => { resolveUpload = resolve; }),
    );

    const { PaperUploadWorkspace } = await import('@/components/paper-upload/PaperUploadWorkspace');
    const user = userEvent.setup();
    const { container } = render(<PaperUploadWorkspace />);

    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(fileInput, new File(['%PDF-1.7'], 'test.pdf', { type: 'application/pdf' }));
    await user.click(screen.getByRole('button', { name: /generate workspace/i }));

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /processing/i })).toBeDisabled();
    });

    resolveUpload(
      new Response(
        JSON.stringify({ paper_id: 1, title: 'T' }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
    );
  });

  it('shows ingestion flow step indicators', async () => {
    const { PaperUploadWorkspace } = await import('@/components/paper-upload/PaperUploadWorkspace');
    render(<PaperUploadWorkspace />);

    expect(screen.getByText('Select a PDF and optional title')).toBeInTheDocument();
    expect(screen.getByText('Upload to the ingestion pipeline')).toBeInTheDocument();
    expect(screen.getByText('Extract text, figures, and equations')).toBeInTheDocument();
  });
});
