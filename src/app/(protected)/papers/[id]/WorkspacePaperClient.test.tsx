import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { describe, test, expect, beforeEach, vi, type MockedFunction } from 'vitest';
import WorkspacePaperClient from './WorkspacePaperClient';
import { apiGet, apiPost } from '@/lib/api';

// Mock the API calls
vi.mock('@/lib/api', () => ({
  apiGet: vi.fn(),
  apiPost: vi.fn(),
}));

const mockApiGet = apiGet as MockedFunction<typeof apiGet>;
const mockApiPost = apiPost as MockedFunction<typeof apiPost>;

describe('WorkspacePaperClient AI Tutor', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  test('tutor tab sends to /api/papers/{id}/ask for non-flagship paper', async () => {
    mockApiGet.mockImplementation(async (url: string) => {
      if (url === '/api/papers/5') {
        return {
          title: 'Test Paper', authors: 'Test', year: 2024, abstract: 'Abstract',
          color: '#000', contributions: [],
        };
      }
      return null;
    });

    mockApiPost.mockImplementation(async (url: string, _body?: unknown) => {
      if (url === '/api/papers/5/ask') {
        return { answer: 'Because...', referenced_papers: [] };
      }
      return null;
    });

    render(<WorkspacePaperClient id="5" />);

    // Wait for the summary fetch to complete and UI to render
    await waitFor(() => {
      expect(screen.queryByText(/Loading/i)).not.toBeInTheDocument();
      expect(screen.getByText('Test Paper')).toBeInTheDocument();
    });

    // Switch to tutor tab
    const tutorTab = screen.getByText('AI Tutor');
    fireEvent.click(tutorTab);

    // Type a question into the tutor input
    const input = screen.getByPlaceholderText(/Ask about this paper/i);
    fireEvent.change(input, { target: { value: 'Why?' } });
    
    // Click send
    const sendButton = screen.getByText('Send');
    fireEvent.click(sendButton);

    // Assert POST was called and chat shows answer
    await waitFor(() => {
      expect(mockApiPost).toHaveBeenCalledWith('/api/papers/5/ask', { question: 'Why?' });
      expect(screen.getByText('Because...')).toBeInTheDocument();
    });
  });

  test('tutor tab falls back gracefully when numericId not yet resolved', async () => {
    mockApiGet.mockImplementation(async (url: string) => {
      if (url === '/api/papers') {
        return { papers: [] }; // Can't resolve
      }
      return null;
    });

    mockApiPost.mockImplementation(async (url: string, _body?: unknown) => {
      if (url === '/api/tutor/ask') {
        return { answer: 'Fallback answer', session_id: '123' };
      }
      return null;
    });

    // We use attention-is-all-you-need, which is a flagship ID string
    render(<WorkspacePaperClient id="attention-is-all-you-need" />);

    // Wait for UI
    await waitFor(() => {
      expect(screen.queryByText(/Loading/i)).not.toBeInTheDocument();
    });

    // Switch to tutor tab
    const tutorTab = screen.getByText('AI Tutor');
    fireEvent.click(tutorTab);

    // Type a question
    const input = screen.getByPlaceholderText(/Ask about this paper/i);
    fireEvent.change(input, { target: { value: 'Why?' } });
    
    // Click send
    const sendButton = screen.getByText('Send');
    fireEvent.click(sendButton);

    // Assert fallback API was called
    await waitFor(() => {
      expect(mockApiPost).toHaveBeenCalledWith('/api/tutor/ask', expect.objectContaining({
        query: 'Why?',
        context_type: 'paper'
      }));
      expect(screen.getByText('Fallback answer')).toBeInTheDocument();
    });
  });
});
