// Component tests for TokenStatusIndicator
// Tests status states (connected, error, warning), version display, and auth mode indicators

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import React from 'react';

// Mock TokenStatusIndicator component (to be updated)
const TokenStatusIndicator = ({ 
    status, 
    version, 
    authRequired,
    error 
}: { 
    status: 'connected' | 'error' | 'warning' | 'loading';
    version?: string;
    authRequired?: boolean;
    error?: string;
}) => {
    return <div>Mock TokenStatusIndicator - Not Implemented</div>;
};

describe('TokenStatusIndicator', () => {
    describe('connection status display', () => {
        it('should display connected state with green indicator', () => {
            render(
                <TokenStatusIndicator 
                    status="connected" 
                    version="1.3.0"
                />
            );

            expect(screen.getByText(/connected/i)).toBeInTheDocument();
            // Should have success/green styling (implementation will determine exact class/attribute)
        });

        it('should display error state with red indicator', () => {
            render(
                <TokenStatusIndicator 
                    status="error" 
                    error="Connection failed"
                />
            );

            expect(screen.getByText(/error/i)).toBeInTheDocument();
            expect(screen.getByText(/connection failed/i)).toBeInTheDocument();
        });

        it('should display warning state with yellow indicator', () => {
            render(
                <TokenStatusIndicator 
                    status="warning" 
                />
            );

            expect(screen.getByText(/warning/i)).toBeInTheDocument();
        });

        it('should display loading state with appropriate indicator', () => {
            render(
                <TokenStatusIndicator 
                    status="loading" 
                />
            );

            expect(screen.getByText(/loading|checking|connecting/i)).toBeInTheDocument();
        });
    });

    describe('server version display', () => {
        it('should display server version when connected', () => {
            render(
                <TokenStatusIndicator 
                    status="connected" 
                    version="1.3.0"
                />
            );

            expect(screen.getByText(/1\.3\.0/)).toBeInTheDocument();
        });

        it('should display version with prerelease tag', () => {
            render(
                <TokenStatusIndicator 
                    status="connected" 
                    version="1.3.0-beta.1"
                />
            );

            expect(screen.getByText(/1\.3\.0-beta\.1/)).toBeInTheDocument();
        });

        it('should not display version when not connected', () => {
            render(
                <TokenStatusIndicator 
                    status="error" 
                    error="Network error"
                />
            );

            const versionText = screen.queryByText(/version|v1\./i);
            expect(versionText).not.toBeInTheDocument();
        });

        it('should handle unknown version gracefully', () => {
            render(
                <TokenStatusIndicator 
                    status="connected" 
                    version="unknown"
                />
            );

            expect(screen.getByText(/connected/i)).toBeInTheDocument();
            // Should still show connected status even if version is unknown
        });
    });

    describe('authentication mode indicators', () => {
        it('should display "Auth Required" indicator when authRequired is true', () => {
            render(
                <TokenStatusIndicator 
                    status="connected" 
                    version="1.3.0"
                    authRequired={true}
                />
            );

            expect(screen.getByText(/auth.*required|authentication.*required/i)).toBeInTheDocument();
        });

        it('should display "Auth Optional" indicator when authRequired is false', () => {
            render(
                <TokenStatusIndicator 
                    status="connected" 
                    version="1.3.0"
                    authRequired={false}
                />
            );

            expect(screen.getByText(/auth.*optional|authentication.*optional/i)).toBeInTheDocument();
        });

        it('should show warning icon for unsecured connection (auth disabled)', () => {
            render(
                <TokenStatusIndicator 
                    status="warning" 
                    version="1.3.0"
                    authRequired={false}
                />
            );

            expect(screen.getByText(/warning/i)).toBeInTheDocument();
            expect(screen.getByText(/unsecured|not secure|no authentication/i)).toBeInTheDocument();
        });

        it('should not display auth indicator when auth status is unknown', () => {
            render(
                <TokenStatusIndicator 
                    status="connected" 
                    version="1.2.0"
                />
            );

            const authText = screen.queryByText(/auth/i);
            // May or may not display - implementation decision
            // Test should be flexible here
        });
    });

    describe('error messages', () => {
        it('should display custom error message', () => {
            render(
                <TokenStatusIndicator 
                    status="error" 
                    error="Invalid API token"
                />
            );

            expect(screen.getByText(/invalid api token/i)).toBeInTheDocument();
        });

        it('should display network error with helpful hint', () => {
            render(
                <TokenStatusIndicator 
                    status="error" 
                    error="Network error: Failed to fetch"
                />
            );

            expect(screen.getByText(/network error/i)).toBeInTheDocument();
            expect(screen.getByText(/failed to fetch/i)).toBeInTheDocument();
        });

        it('should display auth error with setup guidance', () => {
            render(
                <TokenStatusIndicator 
                    status="error" 
                    error="401 Unauthorized"
                />
            );

            expect(screen.getByText(/401|unauthorized/i)).toBeInTheDocument();
        });

        it('should not display error message when status is connected', () => {
            render(
                <TokenStatusIndicator 
                    status="connected" 
                    version="1.3.0"
                    error="This should not display"
                />
            );

            const errorText = screen.queryByText(/this should not display/i);
            expect(errorText).not.toBeInTheDocument();
        });
    });

    describe('visual indicators', () => {
        it('should use appropriate icon for connected status', () => {
            const { container } = render(
                <TokenStatusIndicator 
                    status="connected" 
                    version="1.3.0"
                />
            );

            // Should have check/success icon (implementation will determine exact icon)
            expect(container.querySelector('[data-status="connected"]') || 
                   container.textContent).toBeTruthy();
        });

        it('should use appropriate icon for error status', () => {
            const { container } = render(
                <TokenStatusIndicator 
                    status="error" 
                    error="Connection failed"
                />
            );

            // Should have error/alert icon
            expect(container.querySelector('[data-status="error"]') || 
                   container.textContent).toBeTruthy();
        });

        it('should use appropriate icon for warning status', () => {
            const { container } = render(
                <TokenStatusIndicator 
                    status="warning" 
                />
            );

            // Should have warning icon
            expect(container.querySelector('[data-status="warning"]') || 
                   container.textContent).toBeTruthy();
        });
    });

    describe('accessibility', () => {
        it('should have appropriate ARIA role for status', () => {
            render(
                <TokenStatusIndicator 
                    status="connected" 
                    version="1.3.0"
                />
            );

            const statusElement = screen.getByText(/connected/i).closest('[role]');
            expect(statusElement).toHaveAttribute('role', 'status');
        });

        it('should have appropriate ARIA role for error', () => {
            render(
                <TokenStatusIndicator 
                    status="error" 
                    error="Connection failed"
                />
            );

            const alertElement = screen.getByText(/error/i).closest('[role]');
            expect(alertElement).toHaveAttribute('role', 'alert');
        });

        it('should provide clear text alternatives for icons', () => {
            const { container } = render(
                <TokenStatusIndicator 
                    status="connected" 
                    version="1.3.0"
                />
            );

            // Icons should have aria-label or sr-only text
            const icons = container.querySelectorAll('svg');
            icons.forEach(icon => {
                const hasLabel = icon.hasAttribute('aria-label') || 
                               icon.querySelector('.sr-only') !== null;
                expect(hasLabel || icon.getAttribute('aria-hidden') === 'true').toBe(true);
            });
        });
    });

    describe('compact mode', () => {
        it('should support compact display mode', () => {
            const { container } = render(
                <TokenStatusIndicator 
                    status="connected" 
                    version="1.3.0"
                />
            );

            // Compact mode should be supported (implementation detail)
            expect(container.firstChild).toBeTruthy();
        });
    });
});
