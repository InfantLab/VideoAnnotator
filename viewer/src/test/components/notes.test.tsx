/** Spec 028 US6: notes, light touch. */
import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { VIDEO_PATH, lanesLoaded, mockVideoApi, renderVideoPage, wideLayout } from '@/test/videoPageHarness';
import type { VideoNote } from '@/types/video';

const lane = (kind: string) => screen.getAllByRole('group').filter((g) => g.dataset.kind === kind);
const body = (kind: string) => within(lane(kind)[0]).getByTestId(/^lane-body-/);
/** 1000 px for the 20 s demo clip: 50 px a second; the speaker lane is empty at 2 s and 6 s. */
const seekTo = (seconds: number) => fireEvent.click(body('speakers'), { clientX: seconds * 50, clientY: 5 });
const editorBox = () => within(screen.getByTestId('note-editor')).getByRole('textbox', { name: 'Note text' });

const NOTE = (over: Partial<VideoNote>): VideoNote => ({
  id: 'n1',
  video_path: VIDEO_PATH,
  start_sec: 4,
  end_sec: null,
  text: 'infant looks away',
  author_id: 'u1',
  author_name: 'alice',
  created_at: '2026-10-19T10:00:00Z',
  updated_at: null,
  updated_by: null,
  ...over,
});

let api: ReturnType<typeof mockVideoApi>;

beforeEach(() => {
  wideLayout();
});
afterEach(() => vi.restoreAllMocks());

describe('notes', () => {
  it('N opens a note at this moment, focused; Enter saves it', async () => {
    api = mockVideoApi();
    renderVideoPage();
    await lanesLoaded();
    seekTo(6);
    fireEvent.keyDown(window, { key: 'n' });
    expect(screen.getByTestId('note-editor')).toHaveTextContent('New note at 0:06.0');
    expect(document.activeElement).toBe(editorBox());
    fireEvent.change(editorBox(), { target: { value: 'infant looks away' } });
    fireEvent.keyDown(editorBox(), { key: 'Enter' });
    await waitFor(() => expect(api.createNote).toHaveBeenCalledWith({ path: VIDEO_PATH, start_sec: 6, end_sec: null, text: 'infant looks away' }));
    expect(screen.queryByTestId('note-editor')).toBeNull();
  });

  it('Esc on an empty note cancels it, and nothing is saved', async () => {
    api = mockVideoApi();
    renderVideoPage();
    await lanesLoaded();
    fireEvent.keyDown(window, { key: 'n' });
    fireEvent.keyDown(editorBox(), { key: 'Escape' });
    expect(screen.queryByTestId('note-editor')).toBeNull();
    expect(api.createNote).not.toHaveBeenCalled();
  });

  it.each([
    ['forwards', 2, 6],
    ['backwards', 6, 2],
  ])('Shift+N twice makes a span between the two times (%s)', async (_way, first, second) => {
    api = mockVideoApi();
    renderVideoPage();
    await lanesLoaded();
    seekTo(first);
    fireEvent.keyDown(window, { key: 'N', shiftKey: true });
    expect(screen.getByTestId('span-pending')).toHaveTextContent(`Span from 0:0${first}.0`);
    seekTo(second);
    fireEvent.keyDown(window, { key: 'N', shiftKey: true });
    expect(screen.getByTestId('note-editor')).toHaveTextContent('New note at 0:02.0–0:06.0');
    fireEvent.change(editorBox(), { target: { value: 'peekaboo game' } });
    fireEvent.keyDown(editorBox(), { key: 'Enter' });
    await waitFor(() => expect(api.createNote).toHaveBeenCalledWith({ path: VIDEO_PATH, start_sec: 2, end_sec: 6, text: 'peekaboo game' }));
  });

  it('typing "n" in the search box makes no note', async () => {
    api = mockVideoApi();
    renderVideoPage();
    await lanesLoaded();
    const search = screen.getByRole('textbox', { name: 'Search the rows' });
    search.focus();
    fireEvent.keyDown(search, { key: 'n' });
    expect(screen.queryByTestId('note-editor')).toBeNull();
  });

  it('shows notes on their own lane, last and outlined, and as note rows in the Data tab', async () => {
    api = mockVideoApi(undefined, [NOTE({}), NOTE({ id: 'n2', start_sec: 8, end_sec: 10, text: 'peekaboo' })]);
    renderVideoPage();
    await lanesLoaded();
    await waitFor(() => expect(within(lane('notes')[0]).getAllByTestId('row-label')).toHaveLength(1));
    const groups = screen.getAllByRole('group');
    const notesLane = groups[groups.length - 1];
    expect(notesLane.dataset.kind).toBe('notes');
    expect(notesLane.className).toContain('border-dashed');
    expect(within(notesLane).getByTestId('lane-label')).toHaveTextContent('Notes (yours)');
    // Never inside a pipeline's lane.
    expect(lane('speech')[0].textContent).not.toContain('peekaboo');
    // The Data tab draws only the rows near the playhead: find each note's row.
    const search = screen.getByRole('textbox', { name: 'Search the rows' });
    for (const text of ['infant looks away', 'peekaboo']) {
      fireEvent.change(search, { target: { value: text } });
      const [row] = screen.getAllByTestId('data-row');
      expect(row.dataset.note).toBe('true');
      expect(row).toHaveTextContent('Notes (yours)');
      expect(row).toHaveTextContent(text);
    }
  });

  it('opens a note from its lane to change or delete it', async () => {
    api = mockVideoApi(undefined, [NOTE({})]);
    renderVideoPage();
    await lanesLoaded();
    await waitFor(() => expect(within(lane('notes')[0]).getAllByTestId('row-label').length).toBe(1));
    fireEvent.click(body('notes'), { clientX: 200, clientY: 5 }); // 4 s
    expect(screen.getByTestId('note-editor')).toHaveTextContent('Note at 0:04.0');
    expect(editorBox()).toHaveValue('infant looks away');
    fireEvent.click(within(screen.getByTestId('note-editor')).getByRole('button', { name: 'Delete' }));
    await waitFor(() => expect(api.deleteNote).toHaveBeenCalledWith('n1'));
  });

  it('lists notes in the Notes tab, where they can be changed', async () => {
    api = mockVideoApi(undefined, [NOTE({})]);
    renderVideoPage();
    await lanesLoaded();
    fireEvent.mouseDown(screen.getByRole('tab', { name: 'Notes' }));
    const list = await screen.findByTestId('notes-list');
    expect(list).toHaveTextContent('0:04.0');
    expect(list).toHaveTextContent('infant looks away');
    expect(list).toHaveTextContent('alice');
    fireEvent.click(within(list).getByRole('button', { name: 'Change this note' }));
    fireEvent.change(editorBox(), { target: { value: 'looks back' } });
    fireEvent.keyDown(editorBox(), { key: 'Enter' });
    await waitFor(() => expect(api.updateNote).toHaveBeenCalledWith('n1', { text: 'looks back' }));
  });
});
