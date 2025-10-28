// Unit tests for JobCancelButton component
// Tests rendering, confirmation dialog, and disabled states

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { JobCancelButton } from '@/components/JobCancelButton';
import { apiClient } from '@/api/client';
import type { JobStatus } from '@/types/api';
import React from 'react';

// Mock the API client
vi.mock('@/api/client', () => ({
    apiClient: {
        cancelJob: vi.fn(),
    },
}));

// Mock the toast hook
vi.mock('@/hooks/use-toast', () => ({
    useToast: () => ({
        toast: vi.fn(),
    }),
}));

describe('JobCancelButton', () => {
    let queryClient: QueryClient;
    let user: ReturnType<typeof userEvent.setup>;

    beforeEach(() => {
        queryClient = new QueryClient({
            defaultOptions: {
                queries: { retry: false },
                mutations: { retry: false },
            },
        });
        user = userEvent.setup();
        vi.clearAllMocks();
    });

    const wrapper = ({ children }: { children: React.ReactNode }) => (
        <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );

    describe('conditional rendering', () => {
        it('should render for cancellable jobs (running)', () => {
            render(<JobCancelButton jobId="job123" jobStatus="running" />, { wrapper });

            const button = screen.getByRole('button', { name: /cancel/i });
            expect(button).toBeInTheDocument();
        });

        it('should render for cancellable jobs (pending)', () => {
            render(<JobCancelButton jobId="job123" jobStatus="pending" />, { wrapper });

            const button = screen.getByRole('button', { name: /cancel/i });
            expect(button).toBeInTheDocument();
        });

        it('should render for cancellable jobs (queued)', () => {
            render(<JobCancelButton jobId="job123" jobStatus="queued" />, { wrapper });

            const button = screen.getByRole('button', { name: /cancel/i });
            expect(button).toBeInTheDocument();
        });

        it('should not render for completed jobs', () => {
            const { container } = render(
                <JobCancelButton jobId="job123" jobStatus="completed" />,
                { wrapper }
            );

            expect(container.firstChild).toBeNull();
        });

        it('should not render for failed jobs', () => {
            const { container } = render(
                <JobCancelButton jobId="job123" jobStatus="failed" />,
                { wrapper }
            );

            expect(container.firstChild).toBeNull();
        });

        it('should not render for cancelled jobs', () => {
            const { container } = render(
                <JobCancelButton jobId="job123" jobStatus="cancelled" />,
                { wrapper }
            );

            expect(container.firstChild).toBeNull();
        });

        it('should not render for cancelling jobs', () => {
            const { container } = render(
                <JobCancelButton jobId="job123" jobStatus="cancelling" />,
                { wrapper }
            );

            expect(container.firstChild).toBeNull();
        });
    });

    describe('confirmation dialog', () => {
        it('should show confirmation dialog when clicked', async () => {
            render(<JobCancelButton jobId="job123" jobStatus="running" />, { wrapper });

            const button = screen.getByRole('button', { name: /cancel/i });
            await user.click(button);

            // Dialog should be visible
            expect(screen.getByRole('alertdialog')).toBeInTheDocument();
            expect(screen.getByRole('heading', { name: /cancel job/i })).toBeInTheDocument();
            expect(screen.getByText(/are you sure/i)).toBeInTheDocument();
        });

        it('should close dialog when "Go Back" is clicked', async () => {
            render(<JobCancelButton jobId="job123" jobStatus="running" />, { wrapper });

            const button = screen.getByRole('button', { name: /cancel/i });
            await user.click(button);

            const goBackButton = screen.getByRole('button', { name: /no.*keep running/i });
            await user.click(goBackButton);

            // Dialog should be closed
            await waitFor(() => {
                expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument();
            });
        });

        it('should call cancelJob when confirmed', async () => {
            const mockResponse = {
                job_id: 'job123',
                status: 'cancelling' as const,
                message: 'Cancellation in progress',
            };

            vi.mocked(apiClient.cancelJob).mockResolvedValueOnce(mockResponse);

            render(<JobCancelButton jobId="job123" jobStatus="running" />, { wrapper });

            const button = screen.getByRole('button', { name: /cancel/i });
            await user.click(button);

            const confirmButton = screen.getByRole('button', { name: /yes.*cancel/i });
            await user.click(confirmButton);

            await waitFor(() => {
                expect(apiClient.cancelJob).toHaveBeenCalledWith('job123', undefined);
            });
        });
    });

    describe('loading states', () => {
        it('should show loading spinner during cancellation', async () => {
            let resolveCancel: (value: any) => void;
            const cancelPromise = new Promise((resolve) => {
                resolveCancel = resolve;
            });

            vi.mocked(apiClient.cancelJob).mockReturnValueOnce(cancelPromise as any);

            render(<JobCancelButton jobId="job123" jobStatus="running" />, { wrapper });

            const button = screen.getByRole('button', { name: /cancel/i });
            await user.click(button);

            const confirmButton = screen.getByRole('button', { name: /yes.*cancel/i });
            await user.click(confirmButton);

            // Should show loading state
            await waitFor(() => {
                expect(screen.getByRole('button', { name: /cancelling/i })).toBeDisabled();
            });

            // Resolve the promise
            resolveCancel!({
                job_id: 'job123',
                status: 'cancelled',
                message: 'Cancelled',
            });
        });

        it('should disable button during cancellation', async () => {
            let resolveCancel: (value: any) => void;
            const cancelPromise = new Promise((resolve) => {
                resolveCancel = resolve;
            });

            vi.mocked(apiClient.cancelJob).mockReturnValueOnce(cancelPromise as any);

            render(<JobCancelButton jobId="job123" jobStatus="running" />, { wrapper });

            const button = screen.getByRole('button', { name: /cancel/i });
            await user.click(button);

            const confirmButton = screen.getByRole('button', { name: /yes.*cancel/i });
            await user.click(confirmButton);

            // Button should be disabled
            await waitFor(() => {
                const cancellingButton = screen.getByRole('button', { name: /cancelling/i });
                expect(cancellingButton).toBeDisabled();
            });

            // Resolve the promise
            resolveCancel!({
                job_id: 'job123',
                status: 'cancelled',
                message: 'Cancelled',
            });
        });
    });

    describe('error handling', () => {
        it('should handle cancellation errors gracefully', async () => {
            const error = new Error('Job already cancelled');
            vi.mocked(apiClient.cancelJob).mockRejectedValueOnce(error);

            render(<JobCancelButton jobId="job123" jobStatus="running" />, { wrapper });

            const button = screen.getByRole('button', { name: /cancel/i });
            await user.click(button);

            const confirmButton = screen.getByRole('button', { name: /yes.*cancel/i });
            await user.click(confirmButton);

            // Should show error state (button re-enabled)
            await waitFor(() => {
                const cancelButton = screen.getByRole('button', { name: /cancel/i });
                expect(cancelButton).not.toBeDisabled();
            });
        });
    });

    describe('props customization', () => {
        it('should apply custom variant', () => {
            render(
                <JobCancelButton jobId="job123" jobStatus="running" variant="outline" />,
                { wrapper }
            );

            const button = screen.getByRole('button', { name: /cancel/i });
            // Outline variant adds border classes
            expect(button.className).toContain('border');
        });

        it('should apply custom size', () => {
            render(
                <JobCancelButton jobId="job123" jobStatus="running" size="sm" />,
                { wrapper }
            );

            const button = screen.getByRole('button', { name: /cancel/i });
            // Small size has h-9
            expect(button.className).toContain('h-9');
        });

        it('should apply custom className', () => {
            render(
                <JobCancelButton
                    jobId="job123"
                    jobStatus="running"
                    className="custom-class"
                />,
                { wrapper }
            );

            const button = screen.getByRole('button', { name: /cancel/i });
            expect(button).toHaveClass('custom-class');
        });
    });
});
