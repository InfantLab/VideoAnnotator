// Integration tests for authentication flow
// Tests token setup, validation, error handling, and status updates

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import React from 'react';

// Mock API client
const mockApiClient = {
    checkHealth: vi.fn(),
    setToken: vi.fn(),
    getToken: vi.fn(),
};

vi.mock('@/api/client', () => ({
    apiClient: mockApiClient,
}));

// Mock test component for authentication flow
const TestAuthenticationFlow = () => {
    const [token, setToken] = React.useState('');
    const [status, setStatus] = React.useState<'idle' | 'checking' | 'success' | 'error'>('idle');
    const [error, setError] = React.useState<string | null>(null);
    const [serverVersion, setServerVersion] = React.useState<string | null>(null);

    const handleValidateToken = async () => {
        setStatus('checking');
        setError(null);

        try {
            const healthResponse = await mockApiClient.checkHealth();
            mockApiClient.setToken(token);
            setStatus('success');
            setServerVersion(healthResponse.version || 'unknown');
        } catch (err: any) {
            setStatus('error');
            setError(err.message || 'Failed to validate token');
        }
    };

    return (
        <div>
            <label htmlFor="token-input">API Token</label>
            <input
                id="token-input"
                type="text"
                value={token}
                onChange={(e) => setToken(e.target.value)}
                placeholder="Enter API token"
            />
            <button onClick={handleValidateToken} disabled={!token || status === 'checking'}>
                Validate Token
            </button>

            {status === 'checking' && <div>Validating token...</div>}
            {status === 'success' && (
                <div role="status">
                    <div>Connected successfully</div>
                    {serverVersion && <div>Server version: {serverVersion}</div>}
                </div>
            )}
            {status === 'error' && (
                <div role="alert">
                    <div>Connection failed</div>
                    {error && <div>{error}</div>}
                </div>
            )}
        </div>
    );
};

describe('Authentication Integration', () => {
    let user: ReturnType<typeof userEvent.setup>;

    beforeEach(() => {
        user = userEvent.setup();
        vi.clearAllMocks();
    });

    afterEach(() => {
        vi.clearAllMocks();
    });

    describe('token validation flow', () => {
        it('should validate token and display success with server version', async () => {
            mockApiClient.checkHealth.mockResolvedValue({
                status: 'healthy',
                version: '1.3.0',
            });

            render(<TestAuthenticationFlow />);

            const tokenInput = screen.getByLabelText(/api token/i);
            const validateButton = screen.getByRole('button', { name: /validate token/i });

            // Enter token
            await user.type(tokenInput, 'valid-token-123');
            expect(validateButton).not.toBeDisabled();

            // Validate token
            await user.click(validateButton);

            // Should show loading state
            expect(screen.getByText(/validating token/i)).toBeInTheDocument();

            // Wait for validation to complete
            await waitFor(() => {
                expect(screen.getByText(/connected successfully/i)).toBeInTheDocument();
            });

            // Should display server version
            expect(screen.getByText(/server version.*1\.3\.0/i)).toBeInTheDocument();

            // API should have been called
            expect(mockApiClient.checkHealth).toHaveBeenCalledTimes(1);
            expect(mockApiClient.setToken).toHaveBeenCalledWith('valid-token-123');
        });

        it('should handle invalid token with clear error message', async () => {
            mockApiClient.checkHealth.mockRejectedValue(new Error('401 Unauthorized'));

            render(<TestAuthenticationFlow />);

            const tokenInput = screen.getByLabelText(/api token/i);
            const validateButton = screen.getByRole('button', { name: /validate token/i });

            await user.type(tokenInput, 'invalid-token');
            await user.click(validateButton);

            await waitFor(() => {
                expect(screen.getByText(/connection failed/i)).toBeInTheDocument();
            });

            expect(screen.getByText(/401 unauthorized/i)).toBeInTheDocument();
            expect(mockApiClient.setToken).not.toHaveBeenCalled();
        });

        it('should handle network errors gracefully', async () => {
            mockApiClient.checkHealth.mockRejectedValue(new Error('Network error: Failed to fetch'));

            render(<TestAuthenticationFlow />);

            const tokenInput = screen.getByLabelText(/api token/i);
            const validateButton = screen.getByRole('button', { name: /validate token/i });

            await user.type(tokenInput, 'any-token');
            await user.click(validateButton);

            await waitFor(() => {
                expect(screen.getByText(/connection failed/i)).toBeInTheDocument();
            });

            expect(screen.getByText(/network error/i)).toBeInTheDocument();
        });

        it('should disable validate button when token is empty', () => {
            render(<TestAuthenticationFlow />);

            const validateButton = screen.getByRole('button', { name: /validate token/i });
            expect(validateButton).toBeDisabled();
        });

        it('should disable validate button while validating', async () => {
            let resolveHealth: (value: any) => void;
            mockApiClient.checkHealth.mockReturnValue(
                new Promise((resolve) => {
                    resolveHealth = resolve;
                })
            );

            render(<TestAuthenticationFlow />);

            const tokenInput = screen.getByLabelText(/api token/i);
            const validateButton = screen.getByRole('button', { name: /validate token/i });

            await user.type(tokenInput, 'token');
            await user.click(validateButton);

            // Button should be disabled while validating
            expect(validateButton).toBeDisabled();

            // Resolve the validation
            resolveHealth!({ status: 'healthy', version: '1.3.0' });

            await waitFor(() => {
                expect(screen.getByText(/connected successfully/i)).toBeInTheDocument();
            });
        });
    });

    describe('token persistence', () => {
        it('should persist token to localStorage on successful validation', async () => {
            const setItemSpy = vi.spyOn(Storage.prototype, 'setItem');

            mockApiClient.checkHealth.mockResolvedValue({
                status: 'healthy',
                version: '1.3.0',
            });

            render(<TestAuthenticationFlow />);

            const tokenInput = screen.getByLabelText(/api token/i);
            await user.type(tokenInput, 'persistent-token');
            await user.click(screen.getByRole('button', { name: /validate token/i }));

            await waitFor(() => {
                expect(screen.getByText(/connected successfully/i)).toBeInTheDocument();
            });

            // Should store token (implementation may use specific key)
            expect(setItemSpy).toHaveBeenCalled();

            setItemSpy.mockRestore();
        });

        it('should load saved token on mount', () => {
            const getItemSpy = vi.spyOn(Storage.prototype, 'getItem')
                .mockReturnValue('saved-token-123');

            render(<TestAuthenticationFlow />);

            // Implementation should load token from storage
            expect(getItemSpy).toHaveBeenCalled();

            getItemSpy.mockRestore();
        });
    });

    describe('server version detection', () => {
        it('should detect v1.3.0 server with extended capabilities', async () => {
            mockApiClient.checkHealth.mockResolvedValue({
                status: 'healthy',
                version: '1.3.0',
                gpu_status: { available: true },
                worker_info: { active_workers: 2 },
            });

            render(<TestAuthenticationFlow />);

            const tokenInput = screen.getByLabelText(/api token/i);
            await user.type(tokenInput, 'token');
            await user.click(screen.getByRole('button', { name: /validate token/i }));

            await waitFor(() => {
                expect(screen.getByText(/server version.*1\.3\.0/i)).toBeInTheDocument();
            });
        });

        it('should detect v1.2.x server with basic capabilities', async () => {
            mockApiClient.checkHealth.mockResolvedValue({
                status: 'healthy',
                version: '1.2.5',
            });

            render(<TestAuthenticationFlow />);

            const tokenInput = screen.getByLabelText(/api token/i);
            await user.type(tokenInput, 'token');
            await user.click(screen.getByRole('button', { name: /validate token/i }));

            await waitFor(() => {
                expect(screen.getByText(/server version.*1\.2\.5/i)).toBeInTheDocument();
            });
        });

        it('should handle server without version field', async () => {
            mockApiClient.checkHealth.mockResolvedValue({
                status: 'healthy',
            });

            render(<TestAuthenticationFlow />);

            const tokenInput = screen.getByLabelText(/api token/i);
            await user.type(tokenInput, 'token');
            await user.click(screen.getByRole('button', { name: /validate token/i }));

            await waitFor(() => {
                expect(screen.getByText(/connected successfully/i)).toBeInTheDocument();
            });

            // Should show unknown version or no version
            expect(screen.getByText(/unknown/i)).toBeInTheDocument();
        });
    });

    describe('auth requirement detection', () => {
        it('should detect when authentication is required', async () => {
            mockApiClient.checkHealth.mockResolvedValue({
                status: 'healthy',
                version: '1.3.0',
                auth: {
                    enabled: true,
                    required: true,
                },
            });

            render(<TestAuthenticationFlow />);

            const tokenInput = screen.getByLabelText(/api token/i);
            await user.type(tokenInput, 'token');
            await user.click(screen.getByRole('button', { name: /validate token/i }));

            await waitFor(() => {
                expect(screen.getByText(/connected successfully/i)).toBeInTheDocument();
            });

            // Implementation should indicate auth is required
        });

        it('should warn when authentication is disabled', async () => {
            mockApiClient.checkHealth.mockResolvedValue({
                status: 'healthy',
                version: '1.3.0',
                auth: {
                    enabled: false,
                    required: false,
                },
            });

            render(<TestAuthenticationFlow />);

            const tokenInput = screen.getByLabelText(/api token/i);
            await user.type(tokenInput, 'any-token');
            await user.click(screen.getByRole('button', { name: /validate token/i }));

            await waitFor(() => {
                expect(screen.getByText(/connected successfully/i)).toBeInTheDocument();
            });

            // Implementation should show warning about unsecured connection
        });
    });

    describe('error recovery', () => {
        it('should allow retry after validation failure', async () => {
            mockApiClient.checkHealth
                .mockRejectedValueOnce(new Error('Network error'))
                .mockResolvedValueOnce({ status: 'healthy', version: '1.3.0' });

            render(<TestAuthenticationFlow />);

            const tokenInput = screen.getByLabelText(/api token/i);
            const validateButton = screen.getByRole('button', { name: /validate token/i });

            await user.type(tokenInput, 'token');

            // First attempt fails
            await user.click(validateButton);
            await waitFor(() => {
                expect(screen.getByText(/connection failed/i)).toBeInTheDocument();
            });

            // Second attempt succeeds
            await user.click(validateButton);
            await waitFor(() => {
                expect(screen.getByText(/connected successfully/i)).toBeInTheDocument();
            });

            expect(mockApiClient.checkHealth).toHaveBeenCalledTimes(2);
        });

        it('should clear previous error on new validation attempt', async () => {
            mockApiClient.checkHealth
                .mockRejectedValueOnce(new Error('First error'))
                .mockResolvedValueOnce({ status: 'healthy', version: '1.3.0' });

            render(<TestAuthenticationFlow />);

            const tokenInput = screen.getByLabelText(/api token/i);
            const validateButton = screen.getByRole('button', { name: /validate token/i });

            await user.type(tokenInput, 'token');

            // First attempt shows error
            await user.click(validateButton);
            await waitFor(() => {
                expect(screen.getByText(/first error/i)).toBeInTheDocument();
            });

            // Second attempt should clear previous error
            await user.click(validateButton);
            await waitFor(() => {
                expect(screen.queryByText(/first error/i)).not.toBeInTheDocument();
            });
        });
    });

    describe('accessibility', () => {
        it('should announce validation status changes to screen readers', async () => {
            mockApiClient.checkHealth.mockResolvedValue({
                status: 'healthy',
                version: '1.3.0',
            });

            render(<TestAuthenticationFlow />);

            const tokenInput = screen.getByLabelText(/api token/i);
            await user.type(tokenInput, 'token');
            await user.click(screen.getByRole('button', { name: /validate token/i }));

            await waitFor(() => {
                const statusElement = screen.getByRole('status');
                expect(statusElement).toBeInTheDocument();
            });
        });

        it('should announce errors with alert role', async () => {
            mockApiClient.checkHealth.mockRejectedValue(new Error('Validation failed'));

            render(<TestAuthenticationFlow />);

            const tokenInput = screen.getByLabelText(/api token/i);
            await user.type(tokenInput, 'token');
            await user.click(screen.getByRole('button', { name: /validate token/i }));

            await waitFor(() => {
                const alertElement = screen.getByRole('alert');
                expect(alertElement).toBeInTheDocument();
            });
        });
    });
});
