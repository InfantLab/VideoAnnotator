import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { Footer } from '../components/Footer'
import { FileViewer } from '../components/FileViewer'
import type { StandardAnnotationData } from '../types/annotations'

// Mock the version utils
vi.mock('../utils/version', () => ({
  VERSION: '0.2.0',
  GITHUB_URL: 'https://github.com/InfantLab/video-annotation-viewer',
  APP_NAME: 'Video Annotation Viewer'
}))

describe('Components', () => {
  describe('Footer', () => {
    it('should render version information', () => {
      render(<Footer />)

      expect(screen.getByText('Video Annotation Viewer')).toBeInTheDocument()
      expect(screen.getByText('v0.2.0')).toBeInTheDocument()
    })

    it('should render Source link', () => {
      render(<Footer />)

      const sourceLink = screen.getByRole('link', { name: /Source/i })
      expect(sourceLink).toBeInTheDocument()
      expect(sourceLink).toHaveAttribute('href', 'https://github.com/InfantLab/video-annotation-viewer')
      expect(sourceLink).toHaveAttribute('target', '_blank')
    })

    it('should render VideoAnnotator link', () => {
      render(<Footer />)

      const videoAnnotatorLink = screen.getByRole('link', { name: /videoannotator/i })
      expect(videoAnnotatorLink).toBeInTheDocument()
      expect(videoAnnotatorLink).toHaveAttribute('href', 'https://github.com/InfantLab/VideoAnnotator')
      expect(videoAnnotatorLink).toHaveAttribute('target', '_blank')
    })

    it('should render docs link and powered by text', () => {
      render(<Footer />)

      expect(screen.getByText(/Powered by/i)).toBeInTheDocument()
      expect(screen.getByRole('link', { name: /Docs/i })).toBeInTheDocument()
    })
  })

  describe('FileViewer', () => {
    const mockAnnotationData: StandardAnnotationData = {
      video_info: {
        filename: 'demo.mp4',
        duration: 60,
        width: 1920,
        height: 1080,
        frame_rate: 30
      },
      person_tracking: [
        {
          id: 1,
          image_id: 'frame_001',
          category_id: 1,
          keypoints: new Array(51).fill(0),
          num_keypoints: 17,
          bbox: [50, 100, 200, 300],
          area: 60000,
          iscrowd: 0,
          score: 0.95,
          track_id: 1,
          timestamp: 1.5,
          frame_number: 1,
          person_id: 'person_1',
          person_label: 'adult',
          label_confidence: 0.9,
          labeling_method: 'automatic'
        }
      ],
      speech_recognition: [
        {
          startTime: 0.0,
          endTime: 2.5,
          text: 'Hello world'
        }
      ],
      speaker_diarization: [
        {
          file_id: 'demo',
          start_time: 0.0,
          duration: 2.5,
          end_time: 2.5,
          speaker_id: 'SPEAKER_00',
          confidence: 1,
          pipeline: 'speaker_diarization',
          format: 'rttm'
        }
      ],
      scene_detection: [
        {
          id: 1,
          start_time: 0.0,
          end_time: 5.0,
          duration: 5.0,
          scene_type: 'indoor',
          score: 1,
          video_id: 'demo'
        }
      ],
      face_analysis: [],
      metadata: {
        created: '2025-09-18T00:00:00Z',
        version: 'v1.2.1',
        pipelines: ['person_tracking', 'speech_recognition'],
        source: 'videoannotator'
      }
    }

    it('should render file viewer trigger button', () => {
      const trigger = <button>View Data</button>
      render(<FileViewer annotationData={mockAnnotationData} trigger={trigger} />)

      expect(screen.getByRole('button', { name: 'View Data' })).toBeInTheDocument()
    })

    it('should open dialog when trigger is clicked', () => {
      const trigger = <button>View Data</button>
      render(<FileViewer annotationData={mockAnnotationData} trigger={trigger} />)

      const triggerButton = screen.getByRole('button', { name: 'View Data' })
      fireEvent.click(triggerButton)

      // Dialog should open and show data sections
      expect(screen.getByText('JSON Data Viewer')).toBeInTheDocument()
    })

    it('should display annotation data counts', () => {
      const trigger = <button>View Data</button>
      render(<FileViewer annotationData={mockAnnotationData} trigger={trigger} />)

      fireEvent.click(screen.getByRole('button', { name: 'View Data' }))

      const dataCountsHeading = screen.getByText('📈 Data Counts')
      const countsBlock = dataCountsHeading.nextElementSibling as HTMLElement | null

      expect(countsBlock).not.toBeNull()
      const countsText = countsBlock?.textContent ?? ''

      expect(countsText).toContain('"person_tracking": 1')
      expect(countsText).toContain('"speech_recognition": 1')
    })
  })
})

// Integration test helpers
export const createMockFile = (name: string, content: string, type?: string): File => {
  return new File([content], name, { type })
}

export const createMockAnnotationData = (overrides?: Partial<StandardAnnotationData>): StandardAnnotationData => {
  return {
    video_info: {
      filename: 'test.mp4',
      duration: 0,
      width: 1920,
      height: 1080,
      frame_rate: 30
    },
    person_tracking: [],
    speech_recognition: [],
    speaker_diarization: [],
    scene_detection: [],
    face_analysis: [],
    metadata: {
      created: new Date().toISOString(),
      version: 'test',
      pipelines: [],
      source: 'custom'
    },
    ...overrides
  }
}
