import { describe, it, expect, beforeEach, vi } from 'vitest'
import { detectFileType as mergerDetect } from '../lib/parsers/merger'
import { detectFileType as fileUtilsDetect, detectJSONType } from '../lib/fileUtils'

// Mock fetch for file loading tests
global.fetch = vi.fn()

describe('Integration Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('File Detection Pipeline', () => {
    it('should detect COCO person tracking files', async () => {
      const cocoContent = JSON.stringify({
        info: { description: "COCO Export" },
        annotations: [
          {
            id: 1,
            image_id: "frame_001",
            keypoints: [100, 200, 2, 150, 220, 2], // x,y,visibility for 2 keypoints
            bbox: [50, 100, 200, 300],
            score: 0.95
          }
        ]
      })
      
      const file = new File([cocoContent], 'person_tracking.json', { type: 'application/json' })
      
      // Test detectJSONType first
      const jsonResult = await detectJSONType(file)
      expect(jsonResult.type).toBe('person_tracking')
      expect(jsonResult.confidence).toBe('high')
      
      // Test merger detection as fallback
      const mergerResult = await mergerDetect(file)
      expect(mergerResult.type).toBe('person_tracking')
      expect(mergerResult.confidence).toBeGreaterThan(0.7)
    })

    it('should detect scene detection files', async () => {
      const sceneContent = JSON.stringify([
        { start_time: 0.0, end_time: 5.2, scene_type: 'indoor' },
        { start_time: 5.2, end_time: 10.8, scene_type: 'outdoor' }
      ])
      
      const file = new File([sceneContent], 'scenes.json', { type: 'application/json' })
      
      const jsonResult = await detectJSONType(file)
      expect(jsonResult.type).toBe('scene_detection')
      expect(jsonResult.confidence).toBe('high')
    })

    it('should detect VideoAnnotator complete results', async () => {
      const completeResults = JSON.stringify({
        video_path: "/path/to/video.mp4",
        output_dir: "/path/to/output",
        start_time: "2025-08-06T10:00:00Z",
        total_duration: 120,
        config: {
          scene_detection: { enabled: true },
          person_tracking: { enabled: true }
        },
        pipeline_results: {
          person: { results: [] },
          scene: { results: [] }
        }
      })
      
      const file = new File([completeResults], 'complete_results.json', { type: 'application/json' })
      
      const mergerResult = await mergerDetect(file)
      expect(mergerResult.type).toBe('complete_results')
      expect(mergerResult.confidence).toBeGreaterThan(0.8)
    })

    it('should handle face analysis files', async () => {
      const faceContent = JSON.stringify([
        {
          id: 1,
          image_id: "frame_001", 
          category_id: 100, // Face category
          bbox: [250, 103, 110, 110],
          area: 12100,
          score: 0.85,
          emotions: { happy: 0.8, sad: 0.1 }
        }
      ])
      
      const file = new File([faceContent], 'face_analysis.json', { type: 'application/json' })
      
      const mergerResult = await mergerDetect(file)
      expect(mergerResult.type).toBe('face_analysis')
      expect(mergerResult.confidence).toBeGreaterThan(0.5)
    })
  })

  describe('File Processing Workflow', () => {
    it('should simulate the complete drag & drop detection flow', async () => {
      const unknownJsonContent = JSON.stringify({
        custom_field: "unknown format",
        data: [1, 2, 3]
      })
      
      const file = new File([unknownJsonContent], 'mystery.json', { type: 'application/json' })
      
      // Step 1: fileUtils detection
      let detected = fileUtilsDetect(file)
      expect(detected.type).toBe('unknown')
      expect(detected.extension).toBe('json')
      
      // Step 2: JSON content analysis 
      if (detected.type === 'unknown' && detected.extension === 'json') {
        detected = await detectJSONType(file)
        expect(detected.type).toBe('unknown') // Should still be unknown for this content
        
        // Step 3: Merger fallback
        if (detected.type === 'unknown') {
          const mergerResult = await mergerDetect(file)
          expect(mergerResult.type).toBe('unknown')
          expect(mergerResult.confidence).toBeLessThan(0.5) // Low confidence for unknown format
        }
      }
    })
  })

  describe('Error Handling', () => {
    it('should handle malformed JSON gracefully', async () => {
      const malformedJson = '{ "incomplete": json file'
      const file = new File([malformedJson], 'broken.json', { type: 'application/json' })
      
      // Should not throw errors
      const jsonResult = await detectJSONType(file)
      expect(jsonResult.type).toBe('unknown')
      expect(jsonResult.reason).toContain('parsing error')
    })

    it('should handle empty files', async () => {
      const file = new File([''], 'empty.json', { type: 'application/json' })
      
      const jsonResult = await detectJSONType(file)
      expect(jsonResult.type).toBe('unknown')
    })

    it('should handle binary files with json extension', async () => {
      const binaryData = new ArrayBuffer(100)
      const file = new File([binaryData], 'binary.json', { type: 'application/json' })
      
      const jsonResult = await detectJSONType(file)
      expect(jsonResult.type).toBe('unknown')
    })
  })
})
