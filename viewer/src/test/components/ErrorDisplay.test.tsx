/**
 * T044: Component tests for ErrorDisplay
 * Tests display of error messages, hints, technical details, and field errors
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ErrorDisplay } from '@/components/ErrorDisplay';
import type { ParsedError } from '@/types/api';

describe('ErrorDisplay', () => {
  describe('basic error display', () => {
    it('should render error message', () => {
      const error: ParsedError = {
        message: 'Job not found',
      };

      render(<ErrorDisplay error={error} />);

      expect(screen.getByText(/job not found/i)).toBeInTheDocument();
    });

    it('should render error with code', async () => {
      const user = userEvent.setup();
      const error: ParsedError = {
        message: 'Pipeline configuration invalid',
        code: 'CONFIG_INVALID',
      };

      render(<ErrorDisplay error={error} />);

      expect(screen.getByText(/pipeline configuration invalid/i)).toBeInTheDocument();
      
      // Code is in technical details section - need to expand
      const detailsButton = screen.getByRole('button', { name: /technical details/i });
      await user.click(detailsButton);
      
      expect(screen.getByText('CONFIG_INVALID')).toBeInTheDocument();
    });

    it('should render hint when present', () => {
      const error: ParsedError = {
        message: 'Video URL is invalid',
        hint: 'Use http:// or https:// scheme',
      };

      render(<ErrorDisplay error={error} />);

      expect(screen.getByText(/video url is invalid/i)).toBeInTheDocument();
      expect(screen.getByText(/use http/i)).toBeInTheDocument();
    });
  });

  describe('technical details section', () => {
    it('should have collapsible technical details', async () => {
      const user = userEvent.setup();
      const error: ParsedError = {
        message: 'Server error',
        code: 'INTERNAL_ERROR',
        requestId: 'req-12345',
      };

      render(<ErrorDisplay error={error} />);

      // Should have a button/trigger to expand details
      const detailsButton = screen.getByRole('button', { name: /technical details/i });
      expect(detailsButton).toBeInTheDocument();
      expect(detailsButton).toHaveAttribute('aria-expanded', 'false');

      // Request ID should not be visible initially
      expect(screen.queryByText('req-12345')).not.toBeInTheDocument();

      // Click to expand
      await user.click(detailsButton);

      // Now request ID should be visible
      expect(detailsButton).toHaveAttribute('aria-expanded', 'true');
      expect(screen.getByText('req-12345')).toBeInTheDocument();
    });

    it('should show request ID in technical details', async () => {
      const user = userEvent.setup();
      const error: ParsedError = {
        message: 'Error occurred',
        requestId: 'req-xyz789',
      };

      render(<ErrorDisplay error={error} />);

      const detailsButton = screen.getByRole('button', { name: /technical details/i });
      await user.click(detailsButton);

      expect(screen.getByText('req-xyz789')).toBeInTheDocument();
    });

    it('should hide technical details section if no technical info', () => {
      const error: ParsedError = {
        message: 'Simple error',
      };

      render(<ErrorDisplay error={error} />);

      expect(screen.queryByRole('button', { name: /technical details/i })).not.toBeInTheDocument();
    });
  });

  describe('field errors display', () => {
    it('should render field errors list', () => {
      const error: ParsedError = {
        message: '2 validation errors',
        code: 'VALIDATION_ERROR',
        fieldErrors: [
          { field: 'video_url', message: 'Must be a valid URL' },
          { field: 'pipeline_id', message: 'Pipeline not found' },
        ],
      };

      render(<ErrorDisplay error={error} />);

      expect(screen.getByText(/video_url/i)).toBeInTheDocument();
      expect(screen.getByText(/must be a valid url/i)).toBeInTheDocument();
      expect(screen.getByText(/pipeline_id/i)).toBeInTheDocument();
      expect(screen.getByText(/pipeline not found/i)).toBeInTheDocument();
    });

    it('should render field error hints', () => {
      const error: ParsedError = {
        message: 'Validation failed',
        fieldErrors: [
          {
            field: 'config.threshold',
            message: 'Must be between 0 and 1',
            hint: 'Typical values are 0.5 to 0.8',
          },
        ],
      };

      render(<ErrorDisplay error={error} />);

      expect(screen.getByText(/config\.threshold/i)).toBeInTheDocument();
      expect(screen.getByText(/must be between 0 and 1/i)).toBeInTheDocument();
      expect(screen.getByText(/typical values/i)).toBeInTheDocument();
    });

    it('should highlight field errors visually', () => {
      const error: ParsedError = {
        message: 'Validation failed',
        fieldErrors: [
          { field: 'email', message: 'Invalid format' },
        ],
      };

      const { container } = render(<ErrorDisplay error={error} />);

      // Field errors should be in a distinct section with data-testid
      const fieldErrorSection = container.querySelector('[data-testid="field-errors"]');
      expect(fieldErrorSection).toBeInTheDocument();
    });
  });

  describe('visual states', () => {
    it('should use alert/destructive styling', () => {
      const error: ParsedError = {
        message: 'Critical error',
      };

      const { container } = render(<ErrorDisplay error={error} />);

      // Should use destructive/error variant (check for appropriate classes or role)
      const alert = container.querySelector('[role="alert"]');
      expect(alert).toBeInTheDocument();
    });

    it('should have appropriate icon for error', () => {
      const error: ParsedError = {
        message: 'Error occurred',
      };

      render(<ErrorDisplay error={error} />);

      // Should have an error icon (AlertCircle, XCircle, etc.)
      // This will depend on the icon library used
      const icon = screen.getByTestId('error-icon');
      expect(icon).toBeInTheDocument();
    });
  });

  describe('accessibility', () => {
    it('should have role="alert" for error announcement', () => {
      const error: ParsedError = {
        message: 'Job failed',
      };

      const { container } = render(<ErrorDisplay error={error} />);

      expect(container.querySelector('[role="alert"]')).toBeInTheDocument();
    });

    it('should have accessible collapsible region', async () => {
      const user = userEvent.setup();
      const error: ParsedError = {
        message: 'Error',
        requestId: 'req-123',
      };

      render(<ErrorDisplay error={error} />);

      const detailsButton = screen.getByRole('button', { name: /technical details/i });
      
      // Button should have aria-expanded
      expect(detailsButton).toHaveAttribute('aria-expanded', 'false');

      await user.click(detailsButton);

      expect(detailsButton).toHaveAttribute('aria-expanded', 'true');
    });

    it('should have proper heading hierarchy', () => {
      const error: ParsedError = {
        message: 'Error occurred',
        fieldErrors: [
          { field: 'test', message: 'Invalid' },
        ],
      };

      render(<ErrorDisplay error={error} />);

      // AlertTitle acts as heading
      const heading = screen.getByText('Error');
      expect(heading).toBeInTheDocument();
    });
  });

  describe('copy functionality', () => {
    it('should allow copying technical details', async () => {
      const user = userEvent.setup();
      const error: ParsedError = {
        message: 'Error',
        code: 'TEST_ERROR',
        requestId: 'req-456',
      };

      render(<ErrorDisplay error={error} />);

      // Need to expand technical details first
      const detailsButton = screen.getByRole('button', { name: /technical details/i });
      await user.click(detailsButton);

      // Now copy button should be visible
      const copyButton = screen.getByRole('button', { name: /copy/i });
      expect(copyButton).toBeInTheDocument();
    });
  });
});
