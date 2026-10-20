
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
  it('renders the home page by default', async () => {
    render(<App />);

    // Check if "Video Annotation Viewer" text is present (from Home page)
    await waitFor(() => {
      expect(screen.getAllByText(/Video Annotation Viewer/i).length).toBeGreaterThan(0);
    });
  });
});

describe('the nav (spec 029 FR-018, SC-004)', () => {
  it('has Home, Videos, Ask AI and Pipelines, then Help and Settings, and nothing else', async () => {
    render(<App />);
    const header = await screen.findByRole('banner');
    const nav = header.querySelector('nav')!;
    const entries = [...nav.querySelectorAll('a')].map((a) => a.getAttribute('href'));
    expect(entries).toEqual(['/home', '/videos', '/ask', '/pipelines']);
    const links = [...header.querySelectorAll('a')].map((a) => a.getAttribute('href'));
    expect(links).toEqual(expect.arrayContaining(['/help', '/settings']));
    for (const gone of ['/jobs', '/jobs/new', '/results', '/datasets']) expect(links).not.toContain(gone);
  });
});
