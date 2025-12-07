
import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import App from '../App';

// Mock the API client to avoid network requests
vi.mock('@/api/client', () => ({
  apiClient: {
    getEnhancedHealth: vi.fn().mockResolvedValue({
      status: 'ok',
      version: '1.3.0',
      features: {}
    }),
    getPipelineCatalog: vi.fn().mockResolvedValue({
      catalog: { pipelines: [] },
      server: { version: '1.3.0', features: {} }
    }),
    getServerInfo: vi.fn().mockResolvedValue({
      version: '1.3.0',
      features: {}
    }),
    clearPipelineCache: vi.fn(),
    clearServerInfoCache: vi.fn(),
  }
}));

describe('App Rendering', () => {
  it('renders the welcome screen by default', async () => {
    render(<App />);
    
    // Check if "Video Annotation Viewer" text is present (from WelcomeScreen)
    await waitFor(() => {
      expect(screen.getAllByText(/Video Annotation Viewer/i).length).toBeGreaterThan(0);
    });
  });
});
