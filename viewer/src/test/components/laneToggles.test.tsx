/** Spec 028 US2: toggle from the lanes, not a card. */
import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { idbDel } from '@/lib/persistence/idbKv';
import { lanePrefsKey } from '@/lib/lanePrefs';
import { apiClient } from '@/api/client';
import { lanesLoaded, mockVideoApi, renderVideoPage, wideLayout } from '@/test/videoPageHarness';

const lane = (kind: string) => screen.getAllByRole('group').filter((g) => g.dataset.kind === kind);
const PREFS = () => lanePrefsKey(apiClient.baseURL || window.location.origin, null);

// jsdom has no PointerEvent; a MouseEvent carries the same coordinates.
if (!('PointerEvent' in window)) Object.defineProperty(window, 'PointerEvent', { value: MouseEvent });

beforeEach(async () => {
  await idbDel(PREFS());
  mockVideoApi();
  wideLayout();
});
afterEach(() => vi.restoreAllMocks());

describe('lanes as the controls', () => {
  it('collapses a lane to its label row when its label is clicked, and expands it again', async () => {
    renderVideoPage();
    await lanesLoaded();
    const [speakers] = lane('speakers');
    expect(within(speakers).getAllByTestId('row-label').length).toBeGreaterThan(0);
    fireEvent.click(within(speakers).getByTestId('lane-label'));
    expect(within(lane('speakers')[0]).queryAllByTestId('row-label')).toHaveLength(0);
    expect(within(lane('speakers')[0]).getByTestId('lane-label')).toHaveAttribute('aria-expanded', 'false');
    fireEvent.click(within(lane('speakers')[0]).getByTestId('lane-label'));
    expect(within(lane('speakers')[0]).getAllByTestId('row-label').length).toBeGreaterThan(0);
  });

  it('hides only that lane\'s overlay with its eye', async () => {
    renderVideoPage();
    await lanesLoaded();
    const eye = within(lane('speech')[0]).getByTestId('lane-eye');
    expect(eye).toHaveAttribute('aria-pressed', 'true');
    fireEvent.click(eye);
    expect(within(lane('speech')[0]).getByTestId('lane-eye')).toHaveAttribute('aria-pressed', 'false');
    expect(within(lane('speakers')[0]).getByTestId('lane-eye')).toHaveAttribute('aria-pressed', 'true');
    const faces = lane('faces').map((g) => within(g).getByTestId('lane-eye'));
    fireEvent.click(faces[0]);
    expect(faces.map((e) => e.getAttribute('aria-pressed')).sort()).toEqual(['false', 'true']);
  });

  it('has no eye on kinds that draw nothing on the video', async () => {
    renderVideoPage();
    await lanesLoaded();
    for (const kind of ['action_units', 'head_pose', 'gaze', 'person_area']) {
      expect(within(lane(kind)[0]).queryByTestId('lane-eye')).toBeNull();
    }
  });

  it('switches OpenFace 3\'s drawings from its Faces lane', async () => {
    renderVideoPage();
    await lanesLoaded();
    const openface = lane('faces').find((g) => g.textContent?.includes('OpenFace 3'))!;
    fireEvent.click(within(openface).getByRole('button', { name: 'OpenFace 3 drawings' }));
    const gaze = await screen.findByRole('checkbox', { name: 'Gaze' });
    expect(gaze).toHaveAttribute('data-state', 'unchecked');
    fireEvent.click(gaze);
    expect(screen.getByRole('checkbox', { name: 'Gaze' })).toHaveAttribute('data-state', 'checked');
  });

  it('shows and hides every lane and overlay at once from Layers', async () => {
    renderVideoPage();
    await lanesLoaded();
    fireEvent.click(screen.getByRole('button', { name: 'Layers' }));
    const menu = await screen.findByTestId('layers-menu');
    const [lanesNone, overlaysNone] = within(menu).getAllByRole('button', { name: 'none' });
    fireEvent.click(lanesNone);
    expect(screen.queryAllByTestId('row-label')).toHaveLength(0);
    fireEvent.click(overlaysNone);
    expect(screen.getAllByTestId('lane-eye').every((e) => e.getAttribute('aria-pressed') === 'false')).toBe(true);
    fireEvent.click(within(menu).getAllByRole('button', { name: 'all' })[0]);
    expect(screen.queryAllByTestId('row-label').length).toBeGreaterThan(0);
  });

  it('remembers the lanes and overlays after the page is opened again', async () => {
    const first = renderVideoPage();
    await lanesLoaded();
    await waitFor(() => expect(within(lane('scenes')[0]).getByTestId('lane-label')).toBeInTheDocument());
    // Preferences load from the browser store first; wait until a change is saved.
    await new Promise((r) => setTimeout(r, 50));
    fireEvent.click(within(lane('scenes')[0]).getByTestId('lane-label'));
    fireEvent.click(within(lane('speech')[0]).getByTestId('lane-eye'));
    await new Promise((r) => setTimeout(r, 50));
    first.unmount();

    renderVideoPage();
    await lanesLoaded();
    await waitFor(() =>
      expect(within(lane('scenes')[0]).getByTestId('lane-label')).toHaveAttribute('aria-expanded', 'false'),
    );
    expect(within(lane('speech')[0]).getByTestId('lane-eye')).toHaveAttribute('aria-pressed', 'false');
  });

  it('shows the Ask AI answer for this moment on the video, marked as sampled; its eye hides it', async () => {
    renderVideoPage();
    await lanesLoaded();
    // Samples every 5 s: at 6 s the answer is the one sampled at 5 s.
    fireEvent.click(within(lane('speakers')[0]).getByTestId(/^lane-body-/), { clientX: 300, clientY: 5 });
    expect(screen.getByTestId('vlm-overlay')).toHaveTextContent('Ask AI: NO_TOUCH');
    expect(screen.getByTestId('vlm-overlay')).toHaveTextContent('sampled at 0:05.0');
    fireEvent.click(within(lane('ask_ai')[0]).getByTestId('lane-eye'));
    expect(screen.queryByTestId('vlm-overlay')).toBeNull();
  });

  it('has no controls card', async () => {
    renderVideoPage();
    await lanesLoaded();
    expect(screen.queryByText(/Overlay Controls|Timeline Controls/i)).toBeNull();
    expect(screen.queryByRole('button', { name: /View data/i })).toBeNull();
  });

  it('lets the timeline be made taller by dragging its top edge', async () => {
    renderVideoPage();
    await lanesLoaded();
    const handle = screen.getByTestId('timeline-resize');
    expect(screen.getByTestId('timeline-area').style.height).toBe('200px');
    fireEvent.pointerDown(handle, { clientY: 500 });
    fireEvent.pointerMove(handle, { clientY: 400 });
    fireEvent.pointerUp(handle);
    expect(screen.getByTestId('timeline-area').style.height).toBe('300px');
    fireEvent.pointerDown(handle, { clientY: 400 });
    fireEvent.pointerMove(handle, { clientY: 900 });
    fireEvent.pointerUp(handle);
    expect(screen.getByTestId('timeline-area').style.height).toBe('120px');
  });
});
