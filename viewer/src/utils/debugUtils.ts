/**
 * Debug utilities for Phase 5 integration testing
 */

import { parseWebVTT } from '../lib/parsers/webvtt'
import { parseRTTM } from '../lib/parsers/rttm'
import { parseCOCOPersonData } from '../lib/parsers/coco'
import { parseSceneDetection } from '../lib/parsers/scene'
import { detectFileType, mergeAnnotationData } from '../lib/parsers/merger'
import type { StandardAnnotationData } from '../types/annotations'

export interface DemoDataPaths {
  video: string
  coco?: string
  webvtt?: string
  rttm?: string
  scene?: string
  audio?: string
}export const DEMO_DATA_SETS = {
  'peekaboo-rep3': {
    video: 'demo/videos/2UWdXP.joke1.rep3.take1.Peekaboo_h265.mp4',
    coco: 'demo/annotations/2UWdXP.joke1.rep3.take1.Peekaboo_h265/2UWdXP.joke1.rep3.take1.Peekaboo_h265_person_tracking.json',
    webvtt: 'demo/annotations/2UWdXP.joke1.rep3.take1.Peekaboo_h265/2UWdXP.joke1.rep3.take1.Peekaboo_h265_speech_recognition.vtt',
    rttm: 'demo/annotations/2UWdXP.joke1.rep3.take1.Peekaboo_h265/2UWdXP.joke1.rep3.take1.Peekaboo_h265_speaker_diarization.rttm',
    scene: 'demo/annotations/2UWdXP.joke1.rep3.take1.Peekaboo_h265/2UWdXP.joke1.rep3.take1.Peekaboo_h265_scene_detection.json',
    audio: 'demo/annotations/2UWdXP.joke1.rep3.take1.Peekaboo_h265/2UWdXP.joke1.rep3.take1.Peekaboo_h265_audio.wav'
  },
  'peekaboo-rep2': {
    video: 'demo/videos/2UWdXP.joke1.rep2.take1.Peekaboo_h265.mp4',
    coco: 'demo/annotations/2UWdXP.joke1.rep2.take1.Peekaboo_h265/2UWdXP.joke1.rep2.take1.Peekaboo_h265_person_tracking.json',
    rttm: 'demo/annotations/2UWdXP.joke1.rep2.take1.Peekaboo_h265/2UWdXP.joke1.rep2.take1.Peekaboo_h265_speaker_diarization.rttm',
    scene: 'demo/annotations/2UWdXP.joke1.rep2.take1.Peekaboo_h265/2UWdXP.joke1.rep2.take1.Peekaboo_h265_scene_detection.json',
    audio: 'demo/annotations/2UWdXP.joke1.rep2.take1.Peekaboo_h265/2UWdXP.joke1.rep2.take1.Peekaboo_h265_audio.wav'
    // Note: No WebVTT file for this dataset
  }
} as const

export async function loadDemoAnnotations(datasetName: keyof typeof DEMO_DATA_SETS): Promise<StandardAnnotationData | null> {
  const paths = DEMO_DATA_SETS[datasetName]

  try {
    console.log(`🔍 Loading demo dataset: ${datasetName}`)

    // Fetch all available files as File objects (including video for merger requirement)
    const files: File[] = []

    // Fetch video file first (required by merger for metadata extraction)
    const videoRes = await fetch(paths.video)
    const videoBlob = await videoRes.blob()
    const videoFileName = paths.video.split('/').pop() || 'demo.mp4'
    files.push(new File([videoBlob], videoFileName, { type: videoBlob.type }))

    // Fetch COCO file
    if (paths.coco) {
      const res = await fetch(paths.coco)
      const blob = await res.blob()
      files.push(new File([blob], 'person_tracking.json', { type: 'application/json' }))
    }

    // Fetch WebVTT file (only available in peekaboo-rep3)
    if ('webvtt' in paths && paths.webvtt) {
      const res = await fetch(paths.webvtt)
      const blob = await res.blob()
      files.push(new File([blob], 'speech_recognition.vtt', { type: 'text/vtt' }))
    }

    // Fetch RTTM file
    if (paths.rttm) {
      const res = await fetch(paths.rttm)
      const blob = await res.blob()
      files.push(new File([blob], 'speaker_diarization.rttm', { type: 'text/plain' }))
    }

    // Fetch Scene file
    if (paths.scene) {
      const res = await fetch(paths.scene)
      const blob = await res.blob()
      files.push(new File([blob], 'scene_detection.json', { type: 'application/json' }))
    }

    // Detect file types
    const detectedFiles = await Promise.all(
      files.map(async (file) => await detectFileType(file))
    )

    console.log('✅ Files detected:', detectedFiles.map(f => f.type).join(', '))

    // Merge annotation data using the actual merger (includes video metadata extraction)
    const parseResult = await mergeAnnotationData(detectedFiles)

    console.log('🎉 Demo data successfully loaded and merged:', parseResult.data)
    return parseResult.data

  } catch (error) {
    console.error('❌ Error loading demo data:', error)
    return null
  }
}

/**
 * Load demo video file separately for use in VideoAnnotationViewer
 */
export async function loadDemoVideo(datasetName: keyof typeof DEMO_DATA_SETS): Promise<File | null> {
  const paths = DEMO_DATA_SETS[datasetName]

  try {
    console.log(`🎬 Loading demo video: ${datasetName}`)

    const videoRes = await fetch(paths.video)
    const videoBlob = await videoRes.blob()
    const videoFileName = paths.video.split('/').pop() || 'demo.mp4'
    const videoFile = new File([videoBlob], videoFileName, { type: videoBlob.type })

    console.log('✅ Demo video loaded:', videoFileName)
    return videoFile

  } catch (error) {
    console.error('❌ Error loading demo video:', error)
    return null
  }
}

// Make utilities available globally for browser console testing
if (typeof window !== 'undefined') {
  (window as any).debugUtils = {
    loadDemoAnnotations,
    loadDemoVideo,
    DEMO_DATA_SETS
  }
}
