// Spec 024: in a container with nothing shared, the wizard says how to share a
// folder, in the server's words, and offers upload; it never lists a folder.

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { NoFolderAccess } from '@/components/ServerFolderPicker';
import type { IngestAccess } from '@/types/ingest';

const LAUNCHER_REASON =
  'VideoAnnotator can only see folders you share with it. To share one, run: videoannotator-start share';

const noShare: IngestAccess = {
  same_machine: true,
  can_read_in_place: false,
  reason: LAUNCHER_REASON,
  allowed_folders: [],
  results_root: { path: '/home/ada/VideoAnnotator', display_path: '/home/ada/VideoAnnotator' },
  can_open_folders: false,
  places: [],
  in_container: true,
  managed_by_launcher: true,
};

describe('NoFolderAccess', () => {
  it('shows the server’s reason, with the how-to on its own line, and no folders', () => {
    render(<NoFolderAccess access={noShare} onUpload={vi.fn()} />);
    expect(screen.getByText('VideoAnnotator can only see folders you share with it.')).toBeInTheDocument();
    expect(screen.getByText('To share one, run: videoannotator-start share')).toBeInTheDocument();
    expect(screen.queryByText(/\/root/)).not.toBeInTheDocument();
    expect(screen.queryByRole('checkbox')).not.toBeInTheDocument();
  });

  it('switches to upload', () => {
    const onUpload = vi.fn();
    render(<NoFolderAccess access={noShare} onUpload={onUpload} />);
    fireEvent.click(screen.getByRole('button', { name: /Upload videos instead/ }));
    expect(onUpload).toHaveBeenCalled();
  });

  it('shows a one-sentence reason whole', () => {
    render(<NoFolderAccess access={{ ...noShare, reason: 'Only an administrator can do this' }} onUpload={vi.fn()} />);
    expect(screen.getByText('Only an administrator can do this')).toBeInTheDocument();
  });
});
