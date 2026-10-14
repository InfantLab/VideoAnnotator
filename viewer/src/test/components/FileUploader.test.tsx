import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { FileUploader } from '@/components/FileUploader';

const select = (container: HTMLElement, files: File[]) => {
  const input = container.querySelector('input[type="file"]') as HTMLInputElement;
  fireEvent.change(input, { target: { files } });
};

describe('FileUploader', () => {
  it('labels each file by its content and lets a video + JSON results be opened', async () => {
    const { container } = render(<FileUploader onVideoLoad={vi.fn()} onAnnotationLoad={vi.fn()} />);
    select(container, [
      new File(['x'], 'clip.mp4', { type: 'video/mp4' }),
      new File(
        [JSON.stringify({ annotations: [{ keypoints: [1, 2, 2], bbox: [0, 0, 1, 1] }] })],
        'results.json',
        { type: 'application/json' },
      ),
    ]);

    expect(await screen.findByText('Person Tracking (COCO)')).toBeInTheDocument();
    expect(screen.getByText('Video File')).toBeInTheDocument();
    expect(screen.queryByText('Unknown File Type')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Process 2 Files/ })).toBeEnabled();
  });

  it('a video with only ELAN ground truth can be opened', async () => {
    const { container } = render(<FileUploader onVideoLoad={vi.fn()} onAnnotationLoad={vi.fn()} />);
    select(container, [
      new File(['x'], 'clip.mp4', { type: 'video/mp4' }),
      new File(['<ANNOTATION_DOCUMENT></ANNOTATION_DOCUMENT>'], 'truth.eaf'),
    ]);

    expect(await screen.findByText('ELAN Ground Truth (.eaf)')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Process 2 Files/ })).toBeEnabled();
  });
});
