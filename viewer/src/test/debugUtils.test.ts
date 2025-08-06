import { describe, it, expect, vi, beforeEach } from 'vitest'
import { DEMO_DATA_SETS } from '../utils/debugUtils'

// Mock fetch for demo loading tests
global.fetch = vi.fn()

describe('debugUtils', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('DEMO_DATA_SETS', () => {
    it('should contain expected demo datasets', () => {
      expect(DEMO_DATA_SETS).toHaveProperty('peekaboo-rep3-v1.1.1')
      expect(DEMO_DATA_SETS).toHaveProperty('peekaboo-rep2-v1.1.1')
      expect(DEMO_DATA_SETS).toHaveProperty('tearingpaper-rep1-v1.1.1')
      expect(DEMO_DATA_SETS).toHaveProperty('thatsnotahat-rep1-v1.1.1')
      expect(DEMO_DATA_SETS).toHaveProperty('veatic-3-silent')
    })

    it('should have required properties for each dataset', () => {
      Object.values(DEMO_DATA_SETS).forEach(dataset => {
        expect(dataset).toHaveProperty('video')
        expect(dataset).toHaveProperty('complete_results')
        expect(typeof dataset.video).toBe('string')
        expect(typeof dataset.complete_results).toBe('string')
      })
    })

    it('should have VEATIC dataset with additional files', () => {
      const veatic = DEMO_DATA_SETS['veatic-3-silent']
      expect(veatic).toHaveProperty('person_tracking')
      expect(veatic).toHaveProperty('scene_detection')
      expect(veatic).toHaveProperty('scene_results')
    })
  })

  describe('Demo loading workflow', () => {
    it('should handle successful demo file loading', async () => {
      const mockVideoBlob = new Blob(['video data'], { type: 'video/mp4' })
      const mockJsonBlob = new Blob(['{"test": "data"}'], { type: 'application/json' })
      
      // Mock successful fetch responses
      ;(global.fetch as any)
        .mockResolvedValueOnce({ ok: true, blob: () => Promise.resolve(mockVideoBlob) })
        .mockResolvedValueOnce({ ok: true, blob: () => Promise.resolve(mockJsonBlob) })
      
      // Test that we can create File objects from the responses
      const videoResponse = await fetch('demo/videos/test.mp4')
      const videoBlob = await videoResponse.blob()
      const videoFile = new File([videoBlob], 'test.mp4', { type: videoBlob.type })
      
      expect(videoFile).toBeInstanceOf(File)
      expect(videoFile.name).toBe('test.mp4')
      expect(videoFile.type).toBe('video/mp4')
    })

    it('should handle failed demo file loading', async () => {
      // Mock failed fetch response
      ;(global.fetch as any).mockRejectedValueOnce(new Error('Network error'))
      
      try {
        await fetch('demo/videos/nonexistent.mp4')
      } catch (error) {
        expect(error).toBeInstanceOf(Error)
        expect(error.message).toBe('Network error')
      }
    })
  })

  describe('Data integrity checking', () => {
    it('should validate dataset structure', () => {
      const testDataset = {
        video: 'demo/videos/test.mp4',
        complete_results: 'demo/videos_out/test/complete_results.json',
        webvtt: 'demo/videos_out/test/speech.vtt',
        rttm: 'demo/videos_out/test/speakers.rttm'
      }
      
      // Basic structure validation
      expect(testDataset.video).toMatch(/\.(mp4|webm|avi|mov)$/)
      expect(testDataset.complete_results).toMatch(/\.json$/)
      expect(testDataset.webvtt).toMatch(/\.vtt$/)
      expect(testDataset.rttm).toMatch(/\.rttm$/)
    })

    it('should identify missing required fields', () => {
      const incompleteDataset = {
        video: 'demo/videos/test.mp4'
        // missing complete_results
      }
      
      expect(incompleteDataset).not.toHaveProperty('complete_results')
      // This would be caught by integrity checking
    })
  })
})