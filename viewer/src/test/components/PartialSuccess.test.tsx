
import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import CreateJobDetail from '../../pages/CreateJobDetail';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import type { JobResponse } from '@/api/client';

// Mock API client
vi.mock('@/api/client', () => ({
  apiClient: {
    getJob: vi.fn(),
  }
}));

// Mock hooks
vi.mock('@/hooks/useJobCancellation', () => ({
  canCancelJob: vi.fn().mockReturnValue(false),
}));

vi.mock('@/hooks/useJobDeletion', () => ({
  canDeleteJob: vi.fn().mockReturnValue(false),
}));

// Setup QueryClient
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
    },
  },
});

const renderWithProviders = (ui: React.ReactNode) => {
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/create/jobs/job_123']}>
        <Routes>
          <Route path="/create/jobs/:jobId" element={ui} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  );
};

describe('CreateJobDetail - Partial Success', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('displays partial success warning when status is completed but has error_message', async () => {
    // Mock the API response for a partial success job
    const mockJob = {
      id: 'job_123',
      status: 'completed',
      error_message: 'Failed pipelines: speaker_diarization',
      video_filename: 'test_video.mp4',
      created_at: new Date().toISOString(),
      // Add other required fields if necessary
    };

    // We need to mock the useQuery hook or the apiClient.getJob
    // Since the component uses useQuery calling apiClient.getJob, mocking apiClient.getJob is cleaner
    const { apiClient } = await import('@/api/client');
    vi.mocked(apiClient.getJob).mockResolvedValueOnce(mockJob as unknown as JobResponse);

    renderWithProviders(<CreateJobDetail />);

    // Wait for the job to load
    const statusBadge = await screen.findByText('COMPLETED');
    expect(statusBadge).toBeInTheDocument();

    // Check for the partial success alert
    const alertMessage = await screen.findByText(/Partial Success:/i);
    expect(alertMessage).toBeInTheDocument();
    expect(screen.getByText(/Failed pipelines: speaker_diarization/i)).toBeInTheDocument();
    
    // Check if the badge has the orange color class (partial success)
    // Note: We can't easily check class names on the badge component directly without data-testid, 
    // but we can check if the alert is present which confirms the logic branch was taken.
  });

  it('displays normal success when status is completed and no error_message', async () => {
    const mockJob = {
      id: 'job_124',
      status: 'completed',
      error_message: null,
      video_filename: 'test_video.mp4',
    };

    const { apiClient } = await import('@/api/client');
    vi.mocked(apiClient.getJob).mockResolvedValueOnce(mockJob as unknown as JobResponse);

    renderWithProviders(<CreateJobDetail />);

    await screen.findByText('COMPLETED');

    // Should NOT show partial success alert
    const alertMessage = screen.queryByText(/Partial Success:/i);
    expect(alertMessage).not.toBeInTheDocument();
  });

  it('displays failure when status is failed', async () => {
    const mockJob = {
      id: 'job_125',
      status: 'failed',
      error_message: 'Everything exploded',
      video_filename: 'test_video.mp4',
    };

    const { apiClient } = await import('@/api/client');
    vi.mocked(apiClient.getJob).mockResolvedValueOnce(mockJob as unknown as JobResponse);

    renderWithProviders(<CreateJobDetail />);

    await screen.findByText('FAILED');

    // Should NOT show partial success alert (it's a full failure)
    const alertMessage = screen.queryByText(/Partial Success:/i);
    expect(alertMessage).not.toBeInTheDocument();
  });
});
