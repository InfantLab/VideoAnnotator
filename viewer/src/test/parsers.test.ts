import { describe, it, expect } from 'vitest'
import { parseWebVTT } from '../lib/parsers/webvtt'
import { parseRTTM } from '../lib/parsers/rttm'
import { parseSceneDetection } from '../lib/parsers/scene'

// Helper to create mock files
const createMockFile = (content: string, name: string, type = 'text/plain') => {
  return new File([content], name, { type })
}

describe('Parsers', () => {
  describe('parseWebVTT', () => {
    it('should parse valid WebVTT content', async () => {
      const webvttContent = `WEBVTT

00:00:01.000 --> 00:00:03.500
Hello, world!

00:00:04.000 --> 00:00:06.200
How are you doing?`

      const file = createMockFile(webvttContent, 'test.vtt', 'text/vtt')
      const result = await parseWebVTT(file)
      
      expect(result).toHaveLength(2)
      expect(result[0]).toMatchObject({
        startTime: 1.0,
        endTime: 3.5,
        text: 'Hello, world!'
      })
      expect(result[1]).toMatchObject({
        startTime: 4.0,
        endTime: 6.2,
        text: 'How are you doing?'
      })
    })

    it('should handle empty WebVTT content', async () => {
      const file = createMockFile('WEBVTT\n\n', 'empty.vtt', 'text/vtt')
      const result = await parseWebVTT(file)
      expect(result).toHaveLength(0)
    })

    it('should throw for malformed WebVTT content', async () => {
      const file = createMockFile('Invalid content', 'bad.vtt', 'text/vtt')
      await expect(parseWebVTT(file)).rejects.toThrow('Failed to parse WebVTT file')
    })
  })

  describe('parseRTTM', () => {
    it('should parse valid RTTM content', async () => {
      const rttmContent = `SPEAKER file1 1 0.00 2.50 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER file1 1 3.00 1.50 <NA> <NA> SPEAKER_01 <NA> <NA>`

      const file = createMockFile(rttmContent, 'test.rttm')
      const result = await parseRTTM(file)
      
      expect(result).toHaveLength(2)
      expect(result[0]).toMatchObject({
        speaker_id: 'SPEAKER_00',
        start_time: 0.0,
        duration: 2.5,
        end_time: 2.5,
        pipeline: 'speaker_diarization',
        format: 'rttm'
      })
      expect(result[1]).toMatchObject({
        speaker_id: 'SPEAKER_01',
        start_time: 3.0,
        duration: 1.5,
        end_time: 4.5
      })
    })

    it('should handle empty RTTM content', async () => {
      const file = createMockFile('', 'empty.rttm')
      await expect(parseRTTM(file)).rejects.toThrow('Failed to parse RTTM file')
    })

    it('should skip malformed RTTM lines', async () => {
      const rttmContent = `SPEAKER file1 1 0.00 2.50 <NA> <NA> SPEAKER_00 <NA> <NA>
INVALID LINE
SPEAKER file1 1 3.00 1.50 <NA> <NA> SPEAKER_01 <NA> <NA>`

      const file = createMockFile(rttmContent, 'test.rttm')
      const result = await parseRTTM(file)
      expect(result).toHaveLength(2) // Should skip the invalid line
    })
  })

  describe('parseSceneDetection', () => {
    it('should parse scene detection array format', async () => {
      const sceneData = [
        { start_time: 0.0, end_time: 5.2, scene_type: 'intro' },
        { start_time: 5.2, end_time: 10.8, scene_type: 'action' }
      ]

      const file = createMockFile(JSON.stringify(sceneData), 'scenes.json', 'application/json')
      const result = await parseSceneDetection(file)
      
      expect(result).toHaveLength(2)
      expect(result[0]).toMatchObject({
        start_time: 0.0,
        end_time: 5.2,
        scene_type: 'intro'
      })
      expect(result[1]).toMatchObject({
        start_time: 5.2,
        end_time: 10.8,
        scene_type: 'action'
      })
    })

    it('should handle empty scene data', async () => {
      const file = createMockFile(JSON.stringify([]), 'empty.json', 'application/json')
      const result = await parseSceneDetection(file)
      expect(result).toHaveLength(0)
    })

    it('should handle COCO format scene annotations', async () => {
      const cocoData = {
        annotations: [
          {
            id: 1,
            image_id: 'frame_0001',
            category_id: 1,
            bbox: [0, 0, 100, 100],
            timestamp: 1.5,
            scene_type: 'outdoor'
          }
        ]
      }

      const file = createMockFile(JSON.stringify(cocoData), 'coco_scenes.json', 'application/json')
      await expect(parseSceneDetection(file)).rejects.toThrow('Failed to parse scene detection data')
    })
  })
})
