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

    it('should render GitHub link', () => {
      render(<Footer />)
      
      const githubLink = screen.getByRole('link', { name: /view on github/i })
      expect(githubLink).toBeInTheDocument()
      expect(githubLink).toHaveAttribute('href', 'https://github.com/InfantLab/video-annotation-viewer')
      expect(githubLink).toHaveAttribute('target', '_blank')
    })

    it('should render VideoAnnotator link', () => {
      render(<Footer />)
      
      const videoAnnotatorLink = screen.getByRole('link', { name: /videoannotator/i })
      expect(videoAnnotatorLink).toBeInTheDocument()
      expect(videoAnnotatorLink).toHaveAttribute('href', 'https://github.com/InfantLab/VideoAnnotator')
      expect(videoAnnotatorLink).toHaveAttribute('target', '_blank')
    })

    it('should render copyright text', () => {
      render(<Footer />)
      
      expect(screen.getByText(/© \d{4} InfantLab/)).toBeInTheDocument()
      expect(screen.getByText(/VideoAnnotator pipeline outputs/)).toBeInTheDocument()
    })
  })

  describe('FileViewer', () => {
    const mockAnnotationData: StandardAnnotationData = {
      person_tracking: [
        {
          id: 1,
          timestamp: 1.5,
          keypoints: [100, 200, 2, 150, 220, 2], // Sample keypoints
          bbox: [50, 100, 200, 300],
          track_id: 'person_1',
          score: 0.95
        }
      ],
      speech_recognition: [
        {
          id: '1',
          startTime: 0.0,
          endTime: 2.5,
          text: 'Hello world'
        }
      ],
      speaker_diarization: [
        {
          speaker: 'SPEAKER_00',
          startTime: 0.0,
          duration: 2.5,
          endTime: 2.5
        }
      ],
      scene_detection: [
        {
          startTime: 0.0,
          endTime: 5.0,
          sceneType: 'indoor'
        }
      ],
      face_analysis: [],
      metadata: {
        pipelines: ['person_tracking', 'speech_recognition'],
        filesProcessed: 2,
        warnings: []
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
      expect(screen.getByText('Annotation Data Viewer')).toBeInTheDocument()
    })

    it('should display annotation data counts', () => {
      const trigger = <button>View Data</button>
      render(<FileViewer annotationData={mockAnnotationData} trigger={trigger} />)
      
      fireEvent.click(screen.getByRole('button', { name: 'View Data' }))
      
      // Should show data counts for each pipeline
      expect(screen.getByText(/person_tracking.*1 item/)).toBeInTheDocument()
      expect(screen.getByText(/speech_recognition.*1 item/)).toBeInTheDocument()
    })
  })
})

// Integration test helpers
export const createMockFile = (name: string, content: string, type?: string): File => {
  return new File([content], name, { type })
}

export const createMockAnnotationData = (overrides?: Partial<StandardAnnotationData>): StandardAnnotationData => {
  return {
    person_tracking: [],
    speech_recognition: [],
    speaker_diarization: [],
    scene_detection: [],
    face_analysis: [],
    metadata: {
      pipelines: [],
      filesProcessed: 0,
      warnings: []
    },
    ...overrides
  }
}