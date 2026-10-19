/** Spec 028 US1: read the timeline like ELAN. */
import { fireEvent, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { lanesLoaded, mockVideoApi, renderVideoPage, wideLayout } from '@/test/videoPageHarness';

const readout = () => screen.getByTestId('readout').textContent ?? '';
const lane = (kind: string) => screen.getAllByRole('group').filter((g) => g.dataset.kind === kind);

beforeEach(() => {
  mockVideoApi();
  wideLayout();
});
afterEach(() => vi.restoreAllMocks());

describe('the Video page timeline', () => {
  it('has one lane per kind of result present, and none for kinds absent', async () => {
    renderVideoPage();
    await lanesLoaded();
    const groups = screen.getAllByRole('group').map((g) => g.dataset.kind);
    expect(new Set(groups)).toEqual(
      new Set(['speech', 'speakers', 'people', 'faces', 'emotion', 'scenes', 'ask_ai', 'action_units', 'head_pose', 'gaze', 'person_area', 'notes']),
    );
    // The researcher's own lane is always there, last.
    expect(groups[groups.length - 1]).toBe('notes');
  });

  it('labels every row', async () => {
    renderVideoPage();
    await lanesLoaded();
    const [speakers] = lane('speakers');
    expect(within(speakers).getByText('SPEAKER_00')).toBeInTheDocument();
    const [people] = lane('people');
    expect(within(people).getByText('person 1 (parent)')).toBeInTheDocument();
  });

  it('gives every lane and row label a tooltip saying what it shows and what made it', async () => {
    renderVideoPage();
    await lanesLoaded();
    const pipelineLanes = screen.getAllByRole('group').filter((g) => g.dataset.kind !== 'notes');
    for (const lane of pipelineLanes) {
      for (const label of [...within(lane).getAllByTestId('lane-label'), ...within(lane).getAllByTestId('row-label')]) {
        expect(label.getAttribute('title')).toMatch(/: .+\. (Made by|From) \w+/);
      }
    }
    const notes = screen.getAllByRole('group').find((g) => g.dataset.kind === 'notes')!;
    expect(within(notes).getByTestId('lane-label').getAttribute('title')).toMatch(/not pipeline output/);
  });

  it('gives a run with both face pipelines two Faces lanes, each naming its pipeline', async () => {
    renderVideoPage();
    await lanesLoaded();
    const labels = lane('faces').map((g) => within(g).getByTestId('lane-label').textContent);
    expect(labels.sort()).toEqual(['Faces · DeepFace', 'Faces · OpenFace 3']);
  });

  it('names the run each lane shows', async () => {
    renderVideoPage();
    await lanesLoaded();
    const runs = new Set(screen.getAllByTestId('lane-run').map((r) => r.textContent));
    expect(runs.size).toBe(1);
    // The day as the browser's locale writes it.
    expect([...runs][0]).toMatch(/^run (12 Oct|Oct 12)$/);
  });

  it('zooms: the range bar and the view follow', async () => {
    renderVideoPage();
    await lanesLoaded();
    expect(screen.getByTestId('view-range')).toHaveTextContent('0:00.0–0:20.0');
    fireEvent.click(screen.getByRole('button', { name: 'Zoom in' }));
    expect(screen.getByTestId('view-range')).toHaveTextContent('0:00.0–0:10.0');
    expect(screen.getByTestId('range-window').style.width).toBe('50%');
    fireEvent.keyDown(window, { key: '0' });
    expect(screen.getByTestId('view-range')).toHaveTextContent('0:00.0–0:20.0');
  });

  it('steps to the next and previous event of the selected lane, not another\'s', async () => {
    renderVideoPage();
    await lanesLoaded();
    fireEvent.click(within(lane('speakers')[0]).getByTestId('row-label'));
    fireEvent.keyDown(window, { key: ']' });
    // The first speaker turn (0.638 s), not the second speech cue.
    expect(readout()).toMatch(/^0:00\.6 /);
    fireEvent.keyDown(window, { key: ']' });
    expect(readout()).toMatch(/^0:02\.4 /);
    fireEvent.click(screen.getByRole('button', { name: 'Previous event' }));
    expect(readout()).toMatch(/^0:00\.6 /);
  });

  it('shows an event\'s text and times on hover', async () => {
    renderVideoPage();
    await lanesLoaded();
    const body = within(lane('speech')[0]).getByTestId(/^lane-body-/);
    // 1000 px for 20 s: 0.5 s is at 25 px, inside the first cue (0.0–1.34 s).
    fireEvent.mouseMove(body, { clientX: 25, clientY: 5 });
    const card = await screen.findByTestId('lane-hover');
    expect(card).toHaveTextContent('Ready, baby girl?');
    expect(card).toHaveTextContent('0:00.0–0:01.3');
  });

  it('seeks to an event when it is clicked, and to the time clicked elsewhere', async () => {
    renderVideoPage();
    await lanesLoaded();
    const body = within(lane('speakers')[0]).getByTestId(/^lane-body-/);
    fireEvent.click(body, { clientX: 50, clientY: 5 }); // 1.0 s, inside the turn at 0.638–1.431
    expect(readout()).toMatch(/^0:00\.6 /);
    fireEvent.click(body, { clientX: 300, clientY: 5 }); // 6 s: between turns
    expect(readout()).toMatch(/^0:06\.0 /);
  });

  it('shows the frame number from the results\' own frame rate', async () => {
    renderVideoPage();
    await lanesLoaded();
    fireEvent.click(within(lane('speakers')[0]).getByTestId(/^lane-body-/), { clientX: 300, clientY: 5 });
    // The demo clip is 14.26 fps (frame 14 at 0.98 s), not the parsers' default 30: frame 86 at 6 s.
    expect(screen.getByTestId('frame-number')).toHaveTextContent('frame 86');
  });

  it('ignores keys while typing in a box', async () => {
    renderVideoPage();
    await lanesLoaded();
    const input = document.createElement('input');
    document.body.appendChild(input);
    input.focus();
    fireEvent.keyDown(input, { key: ']' });
    fireEvent.keyDown(input, { key: '+' });
    expect(readout()).toMatch(/^0:00\.0 /);
    expect(screen.getByTestId('view-range')).toHaveTextContent('0:00.0–0:20.0');
    input.remove();
  });

  it('plays the video from the server\'s stream, with no folder question', async () => {
    const api = mockVideoApi();
    const { container } = renderVideoPage();
    await lanesLoaded();
    const video = container.querySelector('video')!;
    expect(video.getAttribute('src')).toContain('/api/v1/videos/stream?');
    expect(api.streamToken).toHaveBeenCalledWith('/data/pk/demo_clip.mp4');
  });
});
