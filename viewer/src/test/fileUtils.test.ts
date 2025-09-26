import { describe, it, expect } from 'vitest'
import { 
  detectFileType, 
  getFileTypeDescription,
  validateFileSize,
  validateFileSet
} from '../lib/fileUtils'

describe('fileUtils', () => {
  describe('detectFileType', () => {
    it('should detect video files correctly', () => {
      const videoFile = new File([''], 'test.mp4', { type: 'video/mp4' })
      const result = detectFileType(videoFile)
      
      expect(result.type).toBe('video')
      expect(result.confidence).toBe('high')
      expect(result.extension).toBe('mp4')
    })

    it('should detect audio files correctly', () => {
      const audioFile = new File([''], 'test.wav', { type: 'audio/wav' })
      const result = detectFileType(audioFile)
      
      expect(result.type).toBe('audio')
      expect(result.confidence).toBe('high')
      expect(result.extension).toBe('wav')
    })

    it('should detect WebVTT files correctly', () => {
      const webvttFile = new File([''], 'subtitles.vtt', { type: 'text/vtt' })
      const result = detectFileType(webvttFile)
      
      expect(result.type).toBe('speech_recognition')
      expect(result.confidence).toBe('high')
      expect(result.extension).toBe('vtt')
    })

    it('should detect RTTM files correctly', () => {
      const rttmFile = new File([''], 'speakers.rttm')
      const result = detectFileType(rttmFile)
      
      expect(result.type).toBe('speaker_diarization')
      expect(result.confidence).toBe('high')
      expect(result.extension).toBe('rttm')
    })

    it('should mark JSON files as unknown (requiring content analysis)', () => {
      const jsonFile = new File(['{}'], 'data.json', { type: 'application/json' })
      const result = detectFileType(jsonFile)
      
      expect(result.type).toBe('unknown')
      expect(result.extension).toBe('json')
      expect(result.reason).toContain('content analysis')
    })

    it('should handle unknown file types', () => {
      const unknownFile = new File([''], 'file.xyz')
      const result = detectFileType(unknownFile)
      
      expect(result.type).toBe('unknown')
      expect(result.confidence).toBe('low')
    })
  })

  describe('getFileTypeDescription', () => {
    it('should return correct descriptions for all file types', () => {
      expect(getFileTypeDescription('video')).toBe('Video File')
      expect(getFileTypeDescription('audio')).toBe('Audio File')
      expect(getFileTypeDescription('person_tracking')).toBe('Person Tracking (COCO)')
      expect(getFileTypeDescription('speech_recognition')).toBe('Speech Recognition (WebVTT)')
      expect(getFileTypeDescription('speaker_diarization')).toBe('Speaker Diarization (RTTM)')
      expect(getFileTypeDescription('scene_detection')).toBe('Scene Detection (JSON)')
      expect(getFileTypeDescription('face_analysis')).toBe('Face Analysis (COCO)')
      expect(getFileTypeDescription('complete_results')).toBe('Complete Results (VideoAnnotator)')
      expect(getFileTypeDescription('unknown')).toBe('Unknown File Type')
    })
  })

  describe('validateFileSize', () => {
    it('should accept normal-sized video files', () => {
      const videoFile = new File([new ArrayBuffer(100 * 1024 * 1024)], 'test.mp4', { type: 'video/mp4' }) // 100MB
      const result = validateFileSize(videoFile)
      
      expect(result.valid).toBe(true)
    })

    it('should reject oversized video files', () => {
      const largeVideoFile = new File([new ArrayBuffer(600 * 1024 * 1024)], 'large.mp4', { type: 'video/mp4' }) // 600MB
      Object.defineProperty(largeVideoFile, 'size', { value: 600 * 1024 * 1024 })
      const result = validateFileSize(largeVideoFile)

      expect(result.valid).toBe(false)
      expect(result.error).toContain('Video file too large')
    })

    it('should accept normal-sized annotation files', () => {
      const jsonFile = new File([JSON.stringify({})], 'data.json', { type: 'application/json' })
      Object.defineProperty(jsonFile, 'size', { value: 5 * 1024 * 1024 }) // 5MB
      
      const result = validateFileSize(jsonFile)
      expect(result.valid).toBe(true)
    })
  })

  describe('validateFileSet', () => {
    it('should require at least one video file', () => {
      const jsonFile = new File([JSON.stringify({})], 'data.json')
      const result = validateFileSet([jsonFile])
      
      expect(result.valid).toBe(false)
      expect(result.missing).toContain('Video file (.mp4, .webm, .avi, .mov)')
    })

    it('should accept valid file combinations', () => {
      const videoFile = new File([''], 'test.mp4', { type: 'video/mp4' })
      const webvttFile = new File([''], 'speech.vtt', { type: 'text/vtt' })
      
      const result = validateFileSet([videoFile, webvttFile])
      expect(result.valid).toBe(true)
      expect(result.missing).toHaveLength(0)
    })

    it('should warn about unknown files', () => {
      const videoFile = new File([''], 'test.mp4', { type: 'video/mp4' })
      const unknownFile = new File([''], 'unknown.xyz')
      
      const result = validateFileSet([videoFile, unknownFile])
      expect(result.warnings).toContain('1 file(s) could not be identified. Check file formats.')
    })
  })
})
