import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { Footer } from '../components/Footer'
import type { StandardAnnotationData } from '../types/annotations'

// Mock the version utils
vi.mock('../utils/version', () => ({
  VERSION: '0.2.0',
  GITHUB_URL: 'https://github.com/InfantLab/VideoAnnotator',
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
      expect(sourceLink).toHaveAttribute('href', 'https://github.com/InfantLab/VideoAnnotator')
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
})
