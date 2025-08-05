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
  complete_results?: string
  webvtt?: string
  rttm?: string
  audio?: string
}

// VideoAnnotator v1.1.1 Demo Datasets (from demo/videos_out)
export const DEMO_DATA_SETS = {
  'peekaboo-rep3-v1.1.1': {
    video: 'demo/videos/2UWdXP.joke1.rep3.take1.Peekaboo_h265.mp4',
    complete_results: 'demo/videos_out/2UWdXP.joke1.rep3.take1.Peekaboo_h265/complete_results.json',
    webvtt: 'demo/videos_out/2UWdXP.joke1.rep3.take1.Peekaboo_h265/2UWdXP.joke1.rep3.take1.Peekaboo_h265_speech_recognition.vtt',
    rttm: 'demo/videos_out/2UWdXP.joke1.rep3.take1.Peekaboo_h265/2UWdXP.joke1.rep3.take1.Peekaboo_h265_speaker_diarization.rttm',
    audio: 'demo/videos_out/2UWdXP.joke1.rep3.take1.Peekaboo_h265/2UWdXP.joke1.rep3.take1.Peekaboo_h265_audio.wav'
  },
  'peekaboo-rep2-v1.1.1': {
    video: 'demo/videos/2UWdXP.joke1.rep2.take1.Peekaboo_h265.mp4',
    complete_results: 'demo/videos_out/2UWdXP.joke1.rep2.take1.Peekaboo_h265/complete_results.json',
    webvtt: 'demo/videos_out/2UWdXP.joke1.rep2.take1.Peekaboo_h265/2UWdXP.joke1.rep2.take1.Peekaboo_h265_speech_recognition.vtt',
    rttm: 'demo/videos_out/2UWdXP.joke1.rep2.take1.Peekaboo_h265/2UWdXP.joke1.rep2.take1.Peekaboo_h265_speaker_diarization.rttm',
    audio: 'demo/videos_out/2UWdXP.joke1.rep2.take1.Peekaboo_h265/2UWdXP.joke1.rep2.take1.Peekaboo_h265_audio.wav'
  },
  'tearingpaper-rep1-v1.1.1': {
    video: 'demo/videos/3dC3SQ.joke1.rep1.take1.TearingPaper_h265.mp4',
    complete_results: 'demo/videos_out/3dC3SQ.joke1.rep1.take1.TearingPaper_h265/complete_results.json',
    webvtt: 'demo/videos_out/3dC3SQ.joke1.rep1.take1.TearingPaper_h265/3dC3SQ.joke1.rep1.take1.TearingPaper_h265_speech_recognition.vtt',
    rttm: 'demo/videos_out/3dC3SQ.joke1.rep1.take1.TearingPaper_h265/3dC3SQ.joke1.rep1.take1.TearingPaper_h265_speaker_diarization.rttm',
    audio: 'demo/videos_out/3dC3SQ.joke1.rep1.take1.TearingPaper_h265/3dC3SQ.joke1.rep1.take1.TearingPaper_h265_audio.wav'
  },
  'thatsnotahat-rep1-v1.1.1': {
    video: 'demo/videos/6c6MZQ.joke1.rep1.take1.ThatsNotAHat_h265.mp4',
    complete_results: 'demo/videos_out/6c6MZQ.joke1.rep1.take1.ThatsNotAHat_h265/complete_results.json',
    webvtt: 'demo/videos_out/6c6MZQ.joke1.rep1.take1.ThatsNotAHat_h265/6c6MZQ.joke1.rep1.take1.ThatsNotAHat_h265_speech_recognition.vtt',
    rttm: 'demo/videos_out/6c6MZQ.joke1.rep1.take1.ThatsNotAHat_h265/6c6MZQ.joke1.rep1.take1.ThatsNotAHat_h265_speaker_diarization.rttm',
    audio: 'demo/videos_out/6c6MZQ.joke1.rep1.take1.ThatsNotAHat_h265/6c6MZQ.joke1.rep1.take1.ThatsNotAHat_h265_audio.wav'
  }
} as const

export async function loadDemoAnnotations(datasetName: keyof typeof DEMO_DATA_SETS): Promise<StandardAnnotationData | null> {
  const paths = DEMO_DATA_SETS[datasetName]

  try {
    console.log(`🔍 Loading demo dataset: ${datasetName} (VideoAnnotator v1.1.1)`)

    // Fetch all available files as File objects (including video for merger requirement)
    const files: File[] = []

    // Fetch video file first (required by merger for metadata extraction)
    const videoRes = await fetch(paths.video)
    const videoBlob = await videoRes.blob()
    const videoFileName = paths.video.split('/').pop() || 'demo.mp4'
    files.push(new File([videoBlob], videoFileName, { type: videoBlob.type }))

    // Fetch complete results file (v1.1.1 unified format)
    if (paths.complete_results) {
      const res = await fetch(paths.complete_results)
      const blob = await res.blob()
      files.push(new File([blob], 'complete_results.json', { type: 'application/json' }))
    }

    // Fetch WebVTT file (speech recognition - separate from complete results)
    if (paths.webvtt) {
      const res = await fetch(paths.webvtt)
      const blob = await res.blob()
      files.push(new File([blob], 'speech_recognition.vtt', { type: 'text/vtt' }))
    }

    // Fetch RTTM file (speaker diarization - separate from complete results)
    if (paths.rttm) {
      const res = await fetch(paths.rttm)
      const blob = await res.blob()
      files.push(new File([blob], 'speaker_diarization.rttm', { type: 'text/plain' }))
    }

    // Fetch audio file
    if (paths.audio) {
      const res = await fetch(paths.audio)
      const blob = await res.blob()
      files.push(new File([blob], 'audio.wav', { type: 'audio/wav' }))
    }

    // Detect file types
    const detectedFiles = await Promise.all(
      files.map(async (file) => await detectFileType(file))
    )

    console.log('✅ Files detected:', detectedFiles.map(f => `${f.type} (${f.confidence.toFixed(2)})`).join(', '))

    // Merge annotation data using the actual merger (includes video metadata extraction)
    const parseResult = await mergeAnnotationData(detectedFiles)

    console.log('🎉 VideoAnnotator v1.1.1 demo data successfully loaded:', {
      pipelines: parseResult.data.metadata?.pipelines || [],
      person_tracking: parseResult.data.person_tracking?.length || 0,
      face_analysis: parseResult.data.face_analysis?.length || 0,
      speech_recognition: parseResult.data.speech_recognition?.length || 0,
      speaker_diarization: parseResult.data.speaker_diarization?.length || 0,
      scene_detection: parseResult.data.scene_detection?.length || 0
    })
    
    return parseResult.data

  } catch (error) {
    console.error('❌ Error loading VideoAnnotator v1.1.1 demo data:', error)
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
