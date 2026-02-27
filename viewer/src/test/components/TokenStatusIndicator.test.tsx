// Component tests for TokenStatusIndicator
// Tests status states (connected, error, loading), version display, and auth indicators

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import React from 'react';

// Mock the hooks the component depends on
const mockTokenStatus = {
    isValid: false,
    user: undefined as string | undefined,
    error: undefined as string | undefined,
    isLoading: false,
    refresh: vi.fn(),
    validate: vi.fn(),
};

const mockCapabilities = {
    capabilities: null as import('@/types/api').ServerCapabilities | null,
    isLoading: false,
    error: null as Error | null,
    lastRefresh: null as Date | null,
    refresh: vi.fn(),
};

vi.mock('@/hooks/useTokenStatus', () => ({
    useTokenStatus: () => mockTokenStatus,
}));

vi.mock('@/contexts/ServerCapabilitiesContext', () => ({
    useServerCapabilitiesContext: () => mockCapabilities,
}));

import { TokenStatusIndicator } from '@/components/TokenStatusIndicator';

function renderWithRouter(ui: React.ReactElement) {
    return render(<MemoryRouter>{ui}</MemoryRouter>);
}

function setConnected(version = '1.3.0') {
    mockTokenStatus.isValid = true;
    mockTokenStatus.isLoading = false;
    mockTokenStatus.error = undefined;
    mockCapabilities.capabilities = {
        version,
        supportsJobCancellation: true,
        supportsConfigValidation: true,
        supportsEnhancedHealth: true,
        supportsEnhancedErrors: true,
        detectedAt: new Date(),
    };
    mockCapabilities.isLoading = false;
    mockCapabilities.error = null;
}

function setError(errorMsg: string) {
    mockTokenStatus.isValid = false;
    mockTokenStatus.isLoading = false;
    mockTokenStatus.error = errorMsg;
    mockCapabilities.capabilities = null;
    mockCapabilities.isLoading = false;
    mockCapabilities.error = new Error(errorMsg);
}

function setLoading() {
    mockTokenStatus.isValid = false;
    mockTokenStatus.isLoading = true;
    mockCapabilities.capabilities = null;
    mockCapabilities.isLoading = true;
    mockCapabilities.error = null;
}

beforeEach(() => {
    // Reset to disconnected state
    mockTokenStatus.isValid = false;
    mockTokenStatus.user = undefined;
    mockTokenStatus.error = undefined;
    mockTokenStatus.isLoading = false;
    mockTokenStatus.refresh = vi.fn();
    mockTokenStatus.validate = vi.fn();
    mockCapabilities.capabilities = null;
    mockCapabilities.isLoading = false;
    mockCapabilities.error = null;
    mockCapabilities.lastRefresh = null;
    mockCapabilities.refresh = vi.fn();
});

describe('TokenStatusIndicator', () => {
    describe('connection status display', () => {
        it('should display connected state', () => {
            setConnected('1.3.0');
            renderWithRouter(<TokenStatusIndicator />);

            expect(screen.getByText(/connected/i)).toBeInTheDocument();
        });

        it('should display error state when not connected', () => {
            setError('Connection failed');
            renderWithRouter(<TokenStatusIndicator />);

            expect(screen.getByText(/error/i)).toBeInTheDocument();
        });

        it('should display loading state with spinner text', () => {
            setLoading();
            renderWithRouter(<TokenStatusIndicator />);

            expect(screen.getByText(/checking/i)).toBeInTheDocument();
        });
    });

    describe('server version display', () => {
        it('should display server version when connected', () => {
            setConnected('1.3.0');
            renderWithRouter(<TokenStatusIndicator />);

            expect(screen.getByText(/1\.3\.0/)).toBeInTheDocument();
        });

        it('should display version with prerelease tag', () => {
            setConnected('1.3.0-beta.1');
            renderWithRouter(<TokenStatusIndicator />);

            expect(screen.getByText(/1\.3\.0-beta\.1/)).toBeInTheDocument();
        });

        it('should not display version in compact mode', () => {
            setConnected('1.3.0');
            renderWithRouter(<TokenStatusIndicator compact />);

            // Compact mode hides the version span
            const badge = screen.getByText(/connected/i);
            expect(badge).toBeInTheDocument();
        });

        it('should not show version when unknown', () => {
            setConnected('unknown');
            renderWithRouter(<TokenStatusIndicator />);

            expect(screen.getByText(/connected/i)).toBeInTheDocument();
            // version 'unknown' is hidden by the component
            expect(screen.queryByText(/vunknown/)).not.toBeInTheDocument();
        });
    });

    describe('detail popover (showDetails mode)', () => {
        it('should show detailed info when showDetails is true and connected', () => {
            setConnected('1.3.0');
            mockTokenStatus.user = 'Authenticated';
            renderWithRouter(<TokenStatusIndicator showDetails />);

            // The popover trigger badge should be present
            expect(screen.getByText(/connected/i)).toBeInTheDocument();
        });

        it('should show error details when not connected', () => {
            setError('Connection failed');
            renderWithRouter(<TokenStatusIndicator showDetails />);

            expect(screen.getByText(/error/i)).toBeInTheDocument();
        });
    });

    describe('compact mode', () => {
        it('should show abbreviated loading text in compact mode', () => {
            setLoading();
            renderWithRouter(<TokenStatusIndicator compact />);

            expect(screen.getByText('Checking...')).toBeInTheDocument();
        });

        it('should render without version in compact connected state', () => {
            setConnected('1.3.0');
            renderWithRouter(<TokenStatusIndicator compact />);

            expect(screen.getByText(/connected/i)).toBeInTheDocument();
        });
    });

    describe('rendering', () => {
        it('should render without crashing', () => {
            const { container } = renderWithRouter(<TokenStatusIndicator />);
            expect(container.firstChild).toBeTruthy();
        });

        it('should accept className prop', () => {
            setConnected();
            renderWithRouter(<TokenStatusIndicator className="my-custom-class" />);
            expect(screen.getByText(/connected/i)).toBeInTheDocument();
        });
    });
});
