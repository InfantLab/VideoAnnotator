import { act, renderHook } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { APIClient } from '@/api/client';
import { useZipDownloader } from '@/hooks/useZipDownloader';

// The results page starts a download whenever the state is 'idle'. When a
// missing library folder sent the state back to 'idle', the page started again,
// reopened the picker (or was refused it without a click) and looped: the
// flickering "View jobs" dialog.
describe('useZipDownloader without a library folder', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    delete (window as { showDirectoryPicker?: unknown }).showDirectoryPicker;
  });

  const withPicker = (picker: () => Promise<never>) => {
    const spy = vi.fn(picker);
    Object.defineProperty(window, 'showDirectoryPicker', { value: spy, configurable: true, writable: true });
    return spy;
  };

  it('waits for the user instead of opening the picker without a click', async () => {
    const picker = withPicker(() => Promise.reject(new DOMException('Must be handling a user gesture', 'SecurityError')));
    const artifacts = vi.spyOn(APIClient.prototype, 'getJobArtifacts');
    const { result } = renderHook(() => useZipDownloader());

    await act(() => result.current.startDownload('job-1'));

    expect(result.current.state).toBe('needs_folder');
    expect(picker).not.toHaveBeenCalled();
    expect(artifacts).not.toHaveBeenCalled();
  });

  it('stays put when the user cancels the picker', async () => {
    const picker = withPicker(() => Promise.reject(new DOMException('The user aborted a request.', 'AbortError')));
    const { result } = renderHook(() => useZipDownloader());

    await act(() => result.current.startDownload('job-1', 'pick'));

    expect(picker).toHaveBeenCalledTimes(1);
    expect(result.current.state).toBe('needs_folder');
  });

  it('can view without saving', async () => {
    withPicker(() => Promise.reject(new Error('should not be asked')));
    const artifacts = vi.spyOn(APIClient.prototype, 'getJobArtifacts').mockRejectedValue(new Error('offline'));
    const { result } = renderHook(() => useZipDownloader());

    await act(() => result.current.startDownload('job-1', 'skip'));

    // The viewer plays the video from this zip, so it asks for it (spec 022).
    expect(artifacts).toHaveBeenCalledWith('job-1', { includeVideo: true });
    expect(result.current.state).toBe('error');
  });
});
