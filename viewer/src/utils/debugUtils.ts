/**
 * Debug utilities for Phase 5 integration testing
 */

import { parseWebVTT } from '../lib/parsers/webvtt'
import { parseRTTM } from '../lib/parsers/rttm'
import { parseCOCOPersonData } from '../lib/parsers/coco'
import { parseSceneDetection } from '../lib/parsers/scene'
import { detectFileType, mergeAnnotationData } from '../lib/parsers/merger'
import { apiClient } from '../api/client'
import type { StandardAnnotationData } from '../types/annotations'

export interface DemoDataPaths {
  video: string
  complete_results?: string
  webvtt?: string
  rttm?: string
  audio?: string
  openface3?: string
}

// VideoAnnotator v1.1.1 Demo Datasets (from demo/videos_out)
export const DEMO_DATA_SETS = {
  'peekaboo-rep3-v1.1.1': {
    video: 'demo/videos/2UWdXP.joke1.rep3.take1.Peekaboo_h265.mp4',
    complete_results: 'demo/videos_out/2UWdXP.joke1.rep3.take1.Peekaboo_h265/complete_results.json',
    webvtt: 'demo/videos_out/2UWdXP.joke1.rep3.take1.Peekaboo_h265/2UWdXP.joke1.rep3.take1.Peekaboo_h265_speech_recognition.vtt',
    rttm: 'demo/videos_out/2UWdXP.joke1.rep3.take1.Peekaboo_h265/2UWdXP.joke1.rep3.take1.Peekaboo_h265_speaker_diarization.rttm',
    audio: 'demo/videos_out/2UWdXP.joke1.rep3.take1.Peekaboo_h265/2UWdXP.joke1.rep3.take1.Peekaboo_h265_audio.wav',
    openface3: 'demo/videos_out/2UWdXP.joke1.rep3.take1.Peekaboo_h265/2UWdXP.joke1.rep3.take1.Peekaboo_h265_openface3_analysis.json'
  },
  'peekaboo-rep2-v1.1.1': {
    video: 'demo/videos/2UWdXP.joke1.rep2.take1.Peekaboo_h265.mp4',
    complete_results: 'demo/videos_out/2UWdXP.joke1.rep2.take1.Peekaboo_h265/complete_results.json',
    webvtt: 'demo/videos_out/2UWdXP.joke1.rep2.take1.Peekaboo_h265/2UWdXP.joke1.rep2.take1.Peekaboo_h265_speech_recognition.vtt',
    rttm: 'demo/videos_out/2UWdXP.joke1.rep2.take1.Peekaboo_h265/2UWdXP.joke1.rep2.take1.Peekaboo_h265_speaker_diarization.rttm',
    audio: 'demo/videos_out/2UWdXP.joke1.rep2.take1.Peekaboo_h265/2UWdXP.joke1.rep2.take1.Peekaboo_h265_audio.wav',
    openface3: 'demo/videos_out/2UWdXP.joke1.rep2.take1.Peekaboo_h265/2UWdXP.joke1.rep2.take1.Peekaboo_h265_openface3_analysis.json'
  },
  'tearingpaper-rep1-v1.1.1': {
    video: 'demo/videos/3dC3SQ.joke1.rep1.take1.TearingPaper_h265.mp4',
    complete_results: 'demo/videos_out/3dC3SQ.joke1.rep1.take1.TearingPaper_h265/complete_results.json',
    webvtt: 'demo/videos_out/3dC3SQ.joke1.rep1.take1.TearingPaper_h265/3dC3SQ.joke1.rep1.take1.TearingPaper_h265_speech_recognition.vtt',
    rttm: 'demo/videos_out/3dC3SQ.joke1.rep1.take1.TearingPaper_h265/3dC3SQ.joke1.rep1.take1.TearingPaper_h265_speaker_diarization.rttm',
    audio: 'demo/videos_out/3dC3SQ.joke1.rep1.take1.TearingPaper_h265/3dC3SQ.joke1.rep1.take1.TearingPaper_h265_audio.wav',
    openface3: 'demo/videos_out/3dC3SQ.joke1.rep1.take1.TearingPaper_h265/3dC3SQ.joke1.rep1.take1.TearingPaper_h265_openface3_analysis.json'
  },
  'thatsnotahat-rep1-v1.1.1': {
    video: 'demo/videos/6c6MZQ.joke1.rep1.take1.ThatsNotAHat_h265.mp4',
    complete_results: 'demo/videos_out/6c6MZQ.joke1.rep1.take1.ThatsNotAHat_h265/complete_results.json',
    webvtt: 'demo/videos_out/6c6MZQ.joke1.rep1.take1.ThatsNotAHat_h265/6c6MZQ.joke1.rep1.take1.ThatsNotAHat_h265_speech_recognition.vtt',
    rttm: 'demo/videos_out/6c6MZQ.joke1.rep1.take1.ThatsNotAHat_h265/6c6MZQ.joke1.rep1.take1.ThatsNotAHat_h265_speaker_diarization.rttm',
    audio: 'demo/videos_out/6c6MZQ.joke1.rep1.take1.ThatsNotAHat_h265/6c6MZQ.joke1.rep1.take1.ThatsNotAHat_h265_audio.wav',
    openface3: 'demo/videos_out/6c6MZQ.joke1.rep1.take1.ThatsNotAHat_h265/6c6MZQ.joke1.rep1.take1.ThatsNotAHat_h265_openface3_analysis.json'
  },
  'veatic-3-silent': {
    video: 'demo/videos/3.mp4',
    complete_results: 'demo/videos_out/3/complete_results.json',
    webvtt: 'demo/videos_out/3/3_speech_recognition.vtt',
    rttm: 'demo/videos_out/3/3_speaker_diarization.rttm',
    audio: 'demo/videos_out/3/3_audio.wav',
    // Note: No OpenFace3 data available for this demo
    // Additional individual files for manual testing
    person_tracking: 'demo/videos_out/3/3_person_tracking.json',
    scene_detection: 'demo/videos_out/3/3_scene_detection.json',
    scene_results: 'demo/videos_out/3/scene_results.json'
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

    // Fetch OpenFace3 file (if available)
    if ('openface3' in paths && paths.openface3) {
      const res = await fetch(paths.openface3)
      const blob = await res.blob()
      const fileName = paths.openface3.split('/').pop() || 'openface3_analysis.json'
      files.push(new File([blob], fileName, { type: 'application/json' }))
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
      openface3_faces: parseResult.data.openface3_faces?.length || 0,
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

/**
 * Load demo dataset and switch to viewer (for console testing)
 */
export async function loadDemoDataset(datasetKey: keyof typeof DEMO_DATA_SETS): Promise<void> {
  console.log(`🔄 Loading demo dataset: ${datasetKey}`)

  try {
    const [videoFile, annotation] = await Promise.all([
      loadDemoVideo(datasetKey),
      loadDemoAnnotations(datasetKey)
    ])

    if (videoFile && annotation) {
      console.log('✅ Demo dataset loaded successfully:', {
        video: videoFile.name,
        pipelines: annotation.metadata?.pipelines?.length || 0,
        person_tracking: annotation.person_tracking?.length || 0,
        face_analysis: annotation.face_analysis?.length || 0,
        speech_recognition: annotation.speech_recognition?.length || 0,
        speaker_diarization: annotation.speaker_diarization?.length || 0,
        scene_detection: annotation.scene_detection?.length || 0
      })

      // Try to trigger the viewer (would need app integration)
      console.log('ℹ️ To view this dataset, use the "View Demo" button or FileUploader demo selection')

      return Promise.resolve()
    } else {
      throw new Error('Failed to load demo files')
    }
  } catch (error) {
    console.error('❌ Failed to load demo dataset:', error)
    throw error
  }
}

// API Testing utilities (from browser_debug_console.js)
const createVideoAnnotatorDebug = () => {
  // Use the same configuration as the actual API client for consistency
  const getApiConfig = () => {
    if (typeof window !== 'undefined') {
      return apiClient.getConfig()
    }
    return {
      baseURL: 'http://localhost:18011',
      token: 'dev-token'
    }
  }

  return {
    get apiBase() { return getApiConfig().baseURL },
    get defaultToken() { return getApiConfig().token },
    logRequests: true,

    async checkHealth() {
      console.log('🏥 Checking API health...')
      try {
        const response = await fetch(`${this.apiBase}/health`)
        const data = await response.json()

        console.log('✅ Basic Health:', response.status === 200 ? 'OK' : 'FAIL')
        console.table({
          'Status': data.status,
          'API Version': data.api_version,
          'Server': data.videoannotator_version
        })

        return data
      } catch (error) {
        console.error('❌ Health check failed:', error)
        return null
      }
    },

    async checkAuth(token = null) {
      const authToken = token || this.defaultToken
      console.log(`🔐 Testing authentication with token: ${authToken.substring(0, 10)}...`)

      try {
        const response = await fetch(`${this.apiBase}/api/v1/debug/token-info`, {
          headers: { 'Authorization': `Bearer ${authToken}` }
        })

        if (response.ok) {
          const data = await response.json()
          console.log('✅ Authentication: Valid')
          console.table({
            'User ID': data.token?.user_id,
            'Valid': data.token?.valid,
            'Permissions': data.token?.permissions?.join(', '),
            'Rate Limit': `${data.token?.rate_limit?.remaining_this_minute || '?'}/min remaining`
          })
          return data
        } else {
          console.error('❌ Authentication failed:', response.status, await response.text())
          return null
        }
      } catch (error) {
        console.error('❌ Auth check failed:', error)
        return null
      }
    },

    async runAllTests(token = null) {
      console.log('🧪 Running comprehensive API tests...')
      console.log('='.repeat(50))

      const results = {
        health: await this.checkHealth(),
        auth: await this.checkAuth(token)
      }

      console.log('='.repeat(50))
      console.log('📊 Test Results Summary:')
      Object.entries(results).forEach(([test, result]) => {
        console.log(`- ${test}: ${result ? '✅' : '❌'}`)
      })

      return results
    }
  }
}

// Make utilities available globally for browser console testing
if (typeof window !== 'undefined') {
  const windowWithDebug = window as unknown as {
    debugUtils?: unknown;
    VideoAnnotatorDebug?: unknown;
  };

  windowWithDebug.debugUtils = {
    loadDemoAnnotations,
    loadDemoVideo,
    loadDemoDataset,
    DEMO_DATA_SETS,

    // Helper for quick testing
    async testAllDatasets() {
      console.log('🧪 Testing all demo datasets...')
      const results = {}

      for (const [key, paths] of Object.entries(DEMO_DATA_SETS)) {
        try {
          console.log(`\n📋 Testing ${key}:`)
          await loadDemoDataset(key as keyof typeof DEMO_DATA_SETS)
          results[key] = '✅ SUCCESS'
        } catch (error) {
          console.error(`❌ ${key} failed:`, error)
          results[key] = '❌ FAILED'
        }
      }

      console.log('\n📊 Test Results Summary:', results)
      return results
    },

    // Check data integrity for a specific dataset
    async checkDataIntegrity(datasetKey: keyof typeof DEMO_DATA_SETS) {
      console.log(`🔍 Checking data integrity for ${datasetKey}`)
      const paths = DEMO_DATA_SETS[datasetKey]
      const issues = []

      try {
        // Test complete_results.json if it exists
        if (paths.complete_results) {
          console.log('  📄 Testing complete_results.json...')
          const response = await fetch(paths.complete_results)
          const text = await response.text()

          try {
            const data = JSON.parse(text)
            console.log('    ✅ JSON is valid')
            console.log('    📊 Structure:', {
              has_video_path: !!data.video_path,
              has_pipeline_results: !!data.pipeline_results,
              has_config: !!data.config,
              pipeline_keys: Object.keys(data.pipeline_results || {})
            })
          } catch (parseError) {
            console.error('    ❌ JSON parse error:', parseError)
            console.log('    🔍 First 300 characters:', text.substring(0, 300))
            issues.push(`Malformed complete_results.json: ${parseError.message}`)
          }
        }

        // Test other files
        const fileTests = [
          { name: 'Video', path: paths.video },
          { name: 'WebVTT', path: paths.webvtt },
          { name: 'RTTM', path: paths.rttm },
          { name: 'Audio', path: paths.audio }
        ]

        for (const test of fileTests) {
          if (test.path) {
            try {
              const response = await fetch(test.path)
              if (response.ok) {
                console.log(`    ✅ ${test.name} file accessible`)
              } else {
                console.log(`    ❌ ${test.name} file not accessible (${response.status})`)
                issues.push(`${test.name} file not accessible`)
              }
            } catch (error) {
              console.log(`    ❌ ${test.name} file failed:`, error)
              issues.push(`${test.name} file failed: ${error.message}`)
            }
          }
        }

        if (issues.length === 0) {
          console.log('  ✅ Dataset integrity check passed')
        } else {
          console.log('  ⚠️ Dataset has issues:', issues)
        }

        return { valid: issues.length === 0, issues }
      } catch (error) {
        console.error('  ❌ Integrity check failed:', error)
        return { valid: false, issues: [error.message] }
      }
    },

    // Quick dataset info
    listDatasets() {
      console.log('📋 Available demo datasets:')
      Object.entries(DEMO_DATA_SETS).forEach(([key, paths]) => {
        console.log(`  • ${key}:`)
        console.log(`    - Video: ${paths.video}`)
        console.log(`    - Data: ${paths.complete_results || 'No complete results'}`)
        console.log(`    - Speech: ${paths.webvtt || 'No WebVTT'}`)
        console.log(`    - Speakers: ${paths.rttm || 'No RTTM'}`)
      })
      return DEMO_DATA_SETS
    }
  }

    // Add VideoAnnotatorDebug for API testing
    ; windowWithDebug.VideoAnnotatorDebug = createVideoAnnotatorDebug()

  // Add help message
  console.log(`
🎯 VideoActionViewer Debug Utils Available:

📋 Demo Data Testing:
   window.debugUtils.listDatasets() - Show available datasets
   window.debugUtils.loadDemoDataset('peekaboo-rep3-v1.1.1') - Load specific dataset  
   window.debugUtils.testAllDatasets() - Test all datasets
   window.debugUtils.checkDataIntegrity('dataset-name') - Check file integrity
   window.debugUtils.DEMO_DATA_SETS - Raw dataset configuration

🔧 API Testing:
   VideoAnnotatorDebug.runAllTests() - Run comprehensive API tests
   VideoAnnotatorDebug.checkHealth() - Test API health endpoints
   VideoAnnotatorDebug.checkAuth() - Test authentication

Available datasets: ${Object.keys(DEMO_DATA_SETS).join(', ')}

⚠️  Known Issue: peekaboo-rep2-v1.1.1 has malformed complete_results.json
🔇 New: veatic-3-silent - longer duration video without speech/audio data
`)
}
