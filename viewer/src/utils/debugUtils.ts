/**
 * Debug utilities for Phase 5 integration testing
 */

import { parseWebVTT } from '../lib/parsers/webvtt'
import { parseRTTM } from '../lib/parsers/rttm'
import { parseCOCO } from '../lib/parsers/coco'
import { parseSceneDetection } from '../lib/parsers/scene'
import { mergeAnnotationData } from '../lib/parsers/merger'
import type { StandardAnnotationData } from '../types/annotations'

export interface DemoDataPaths {
    video: string
    coco?: string
    webvtt?: string
    rttm?: string
    scene?: string
    audio?: string
}

export const DEMO_DATA_SETS = {
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

        const loadPromises: Promise<any>[] = []
        const parsedData: any = {}

        // Load COCO person tracking
        if (paths.coco) {
            loadPromises.push(
                fetch(paths.coco)
                    .then(res => res.json())
                    .then(data => {
                        console.log('✅ COCO data loaded')
                        parsedData.coco = parseCOCO(JSON.stringify(data))
                    })
            )
        }

        // Load WebVTT speech recognition
        if (paths.webvtt) {
            loadPromises.push(
                fetch(paths.webvtt)
                    .then(res => res.text())
                    .then(data => {
                        console.log('✅ WebVTT data loaded')
                        parsedData.webvtt = parseWebVTT(data)
                    })
            )
        }

        // Load RTTM speaker diarization
        if (paths.rttm) {
            loadPromises.push(
                fetch(paths.rttm)
                    .then(res => res.text())
                    .then(data => {
                        console.log('✅ RTTM data loaded')
                        parsedData.rttm = parseRTTM(data)
                    })
            )
        }

        // Load scene detection
        if (paths.scene) {
            loadPromises.push(
                fetch(paths.scene)
                    .then(res => res.json())
                    .then(data => {
                        console.log('✅ Scene data loaded')
                        parsedData.scene = parseSceneDetection(JSON.stringify(data))
                    })
            )
        }

        await Promise.all(loadPromises)

        // Merge all data into StandardAnnotationData
        const mergedData = mergeAnnotationData(
            parsedData.coco || null,
            parsedData.webvtt || null,
            parsedData.rttm || null,
            parsedData.scene || null
        )

        console.log('🎉 Demo data successfully loaded and merged:', mergedData)
        return mergedData

    } catch (error) {
        console.error('❌ Error loading demo data:', error)
        return null
    }
}

/**
 * Test individual parsers with demo data
 */
export async function testParsers() {
    console.log('🧪 Testing individual parsers with demo data...')

    try {
        // Test COCO parser
        const cocoResponse = await fetch(DEMO_DATA_SETS['peekaboo-rep3'].coco!)
        const cocoData = await cocoResponse.json()
        const cocoResult = parseCOCO(JSON.stringify(cocoData))
        console.log('✅ COCO parser test:', cocoResult.persons.length, 'person detections')

        // Test WebVTT parser
        const webvttResponse = await fetch(DEMO_DATA_SETS['peekaboo-rep3'].webvtt!)
        const webvttData = await webvttResponse.text()
        const webvttResult = parseWebVTT(webvttData)
        console.log('✅ WebVTT parser test:', webvttResult.cues.length, 'speech cues')

        // Test RTTM parser
        const rttmResponse = await fetch(DEMO_DATA_SETS['peekaboo-rep3'].rttm!)
        const rttmData = await rttmResponse.text()
        const rttmResult = parseRTTM(rttmData)
        console.log('✅ RTTM parser test:', rttmResult.speakers.length, 'speaker segments')

        // Test Scene parser
        const sceneResponse = await fetch(DEMO_DATA_SETS['peekaboo-rep3'].scene!)
        const sceneData = await sceneResponse.json()
        const sceneResult = parseSceneDetection(JSON.stringify(sceneData))
        console.log('✅ Scene parser test:', sceneResult.scenes.length, 'scene boundaries')

        console.log('🎉 All parser tests completed successfully!')

    } catch (error) {
        console.error('❌ Parser test failed:', error)
    }
}

// Make utilities available globally for browser console testing
if (typeof window !== 'undefined') {
    (window as any).debugUtils = {
        loadDemoAnnotations,
        testParsers,
        DEMO_DATA_SETS
    }
}
