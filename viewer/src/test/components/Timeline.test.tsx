import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { Timeline } from '@/components/Timeline';
import type { RTTMSegment, StandardAnnotationData, TimelineSettings } from '@/types/annotations';

const segment = (speaker_id: string, start_time: number): RTTMSegment => ({
  file_id: 'clip', start_time, duration: 1, end_time: start_time + 1, speaker_id,
  confidence: 1, pipeline: 'speaker_diarization', format: 'rttm',
});

const settings: TimelineSettings = {
  showSubtitles: false, showSpeakers: true, showScenes: false,
  showMotion: false, showFaces: false, showEmotions: false,
};

describe('Timeline', () => {
  it('draws each speaker in its own colour', () => {
    const data = {
      video_info: { filename: 'clip.mp4', duration: 10 },
      speaker_diarization: [segment('SPEAKER_00', 0), segment('SPEAKER_01', 2)],
    } as StandardAnnotationData;

    render(<Timeline annotationData={data} currentTime={0} duration={10} settings={settings} onSeek={() => {}} />);

    const a = screen.getByTitle('Speaker: SPEAKER_00').style.backgroundColor;
    const b = screen.getByTitle('Speaker: SPEAKER_01').style.backgroundColor;
    expect(a).not.toBe('');
    expect(a).not.toBe(b);
  });
});
