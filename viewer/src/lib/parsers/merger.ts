// Data merger utility for combining VideoAnnotator pipeline outputs
// Merges person tracking, speech recognition, speaker diarization, and scene detection
// into a unified StandardAnnotationData structure

import type {
    StandardAnnotationData,
    COCOPersonAnnotation,
    WebVTTCue,
    RTTMSegment,
    SceneAnnotation
} from '@/types/annotations';

import { parseWebVTT } from './webvtt';
import { parseRTTM } from './rttm';
import { parseCOCOPersonData } from './coco';
import { parseSceneDetection } from './scene';

/**
 * File type detection result
 */
export interface DetectedFile {
    file: File;
    type: 'video' | 'person_tracking' | 'speech_recognition' | 'speaker_diarization' | 'scene_detection' | 'audio' | 'unknown';
    pipeline?: string;
    confidence: number;
}

/**
 * Parsing progress callback
 */
export type ProgressCallback = (stage: string, progress: number, total: number) => void;

/**
 * Parsing result with metadata
 */
export interface ParseResult {
    data: StandardAnnotationData;
    metadata: {
        filesProcessed: number;
        pipelinesFound: string[];
        warnings: string[];
        processingTime: number;
    };
}

/**
 * Detects file type based on extension and content
 */
export async function detectFileType(file: File): Promise<DetectedFile> {
    const extension = file.name.toLowerCase().split('.').pop() || '';
    const mimeType = file.type.toLowerCase();

    // Video files
    if (['mp4', 'webm', 'avi', 'mov'].includes(extension) || mimeType.startsWith('video/')) {
        return { file, type: 'video', confidence: 0.95 };
    }

    // Audio files
    if (['wav', 'mp3', 'aac', 'ogg'].includes(extension) || mimeType.startsWith('audio/')) {
        return { file, type: 'audio', confidence: 0.95 };
    }

    // Text files - need content analysis
    if (extension === 'vtt' || file.name.includes('speech_recognition')) {
        const isValid = await isValidWebVTT(file);
        return {
            file,
            type: 'speech_recognition',
            pipeline: 'speech_recognition',
            confidence: isValid ? 0.9 : 0.3
        };
    }

    if (extension === 'rttm' || file.name.includes('speaker_diarization')) {
        const isValid = await isValidRTTM(file);
        return {
            file,
            type: 'speaker_diarization',
            pipeline: 'speaker_diarization',
            confidence: isValid ? 0.9 : 0.3
        };
    }

    // JSON files - need content analysis
    if (extension === 'json' || mimeType === 'application/json') {
        return await detectJSONType(file);
    }

    return { file, type: 'unknown', confidence: 0.0 };
}

/**
 * Detects JSON file type based on content structure
 */
async function detectJSONType(file: File): Promise<DetectedFile> {
    try {
        const sample = await file.slice(0, 5000).text();
        const data = JSON.parse(sample);

        // Check for person tracking (COCO format)
        if (await isValidCOCOPersonData(file)) {
            return {
                file,
                type: 'person_tracking',
                pipeline: 'person_tracking',
                confidence: 0.8
            };
        }

        // Check for scene detection
        if (await isValidSceneDetection(file)) {
            return {
                file,
                type: 'scene_detection',
                pipeline: 'scene_detection',
                confidence: 0.8
            };
        }

        // Check filename patterns
        if (file.name.includes('person') || file.name.includes('tracking')) {
            return { file, type: 'person_tracking', pipeline: 'person_tracking', confidence: 0.5 };
        }

        if (file.name.includes('scene')) {
            return { file, type: 'scene_detection', pipeline: 'scene_detection', confidence: 0.5 };
        }

        return { file, type: 'unknown', confidence: 0.2 };

    } catch {
        return { file, type: 'unknown', confidence: 0.0 };
    }
}

/**
 * Helper functions for file validation (imported from parsers)
 */
async function isValidWebVTT(file: File): Promise<boolean> {
    try {
        const firstLine = await file.slice(0, 100).text();
        return firstLine.trim().startsWith('WEBVTT');
    } catch {
        return false;
    }
}

async function isValidRTTM(file: File): Promise<boolean> {
    try {
        const firstLines = await file.slice(0, 1000).text();
        return firstLines.includes('SPEAKER');
    } catch {
        return false;
    }
}

async function isValidCOCOPersonData(file: File): Promise<boolean> {
    try {
        const sample = await file.slice(0, 2000).text();
        const data = JSON.parse(sample);

        if (Array.isArray(data)) {
            return data.length === 0 || (data[0] && 'keypoints' in data[0] && 'bbox' in data[0]);
        }

        return (data.annotations && Array.isArray(data.annotations)) ||
            (data.results && Array.isArray(data.results));
    } catch {
        return false;
    }
}

async function isValidSceneDetection(file: File): Promise<boolean> {
    try {
        const sample = await file.slice(0, 2000).text();
        const data = JSON.parse(sample);

        if (Array.isArray(data)) {
            return data.length === 0 || (data[0] && ('start_time' in data[0] || 'startTime' in data[0]));
        }

        return (data.results && Array.isArray(data.results)) ||
            (data.scenes && Array.isArray(data.scenes));
    } catch {
        return false;
    }
}

/**
 * Extracts video information from video file
 */
async function extractVideoInfo(videoFile: File): Promise<StandardAnnotationData['video_info']> {
    return new Promise((resolve, reject) => {
        const video = document.createElement('video');
        const url = URL.createObjectURL(videoFile);

        video.onloadedmetadata = () => {
            const info = {
                filename: videoFile.name,
                duration: video.duration,
                width: video.videoWidth,
                height: video.videoHeight,
                frame_rate: 30 // Default, could be extracted with more advanced techniques
            };

            URL.revokeObjectURL(url);
            resolve(info);
        };

        video.onerror = () => {
            URL.revokeObjectURL(url);
            reject(new Error('Failed to load video metadata'));
        };

        video.src = url;
    });
}

/**
 * Merges all pipeline outputs into unified annotation data
 */
export async function mergeAnnotationData(
    detectedFiles: DetectedFile[],
    onProgress?: ProgressCallback
): Promise<ParseResult> {
    const startTime = Date.now();
    const warnings: string[] = [];
    const pipelinesFound: string[] = [];

    let videoFile: File | undefined;
    let audioFile: File | undefined;
    let personTracking: COCOPersonAnnotation[] = [];
    let speechRecognition: WebVTTCue[] = [];
    let speakerDiarization: RTTMSegment[] = [];
    let sceneDetection: SceneAnnotation[] = [];

    const totalFiles = detectedFiles.length;
    let processedFiles = 0;

    for (const detectedFile of detectedFiles) {
        try {
            onProgress?.(`Processing ${detectedFile.file.name}`, processedFiles, totalFiles);

            switch (detectedFile.type) {
                case 'video':
                    videoFile = detectedFile.file;
                    break;

                case 'audio':
                    audioFile = detectedFile.file;
                    break;

                case 'person_tracking':
                    personTracking = await parseCOCOPersonData(detectedFile.file);
                    pipelinesFound.push('person_tracking');
                    break;

                case 'speech_recognition':
                    speechRecognition = await parseWebVTT(detectedFile.file);
                    pipelinesFound.push('speech_recognition');
                    break;

                case 'speaker_diarization':
                    speakerDiarization = await parseRTTM(detectedFile.file);
                    pipelinesFound.push('speaker_diarization');
                    break;

                case 'scene_detection':
                    sceneDetection = await parseSceneDetection(detectedFile.file);
                    pipelinesFound.push('scene_detection');
                    break;

                case 'unknown':
                    warnings.push(`Could not determine type of file: ${detectedFile.file.name}`);
                    break;
            }
        } catch (error) {
            warnings.push(`Failed to parse ${detectedFile.file.name}: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }

        processedFiles++;
    }

    // Extract video information
    if (!videoFile) {
        throw new Error('No video file provided');
    }

    onProgress?.('Extracting video metadata', processedFiles, totalFiles + 1);
    const video_info = await extractVideoInfo(videoFile);

    // Create unified annotation data
    const data: StandardAnnotationData = {
        video_info,
        metadata: {
            created: new Date().toISOString(),
            version: '1.0.0',
            pipelines: pipelinesFound,
            source: 'videoannotator'
        }
    };

    // Add pipeline data if available
    if (personTracking.length > 0) {
        data.person_tracking = personTracking;
    }

    if (speechRecognition.length > 0) {
        data.speech_recognition = speechRecognition;
    }

    if (speakerDiarization.length > 0) {
        data.speaker_diarization = speakerDiarization;
    }

    if (sceneDetection.length > 0) {
        data.scene_detection = sceneDetection;
    }

    if (audioFile) {
        data.audio_file = audioFile;
    }

    const processingTime = Date.now() - startTime;

    onProgress?.('Merging complete', totalFiles + 1, totalFiles + 1);

    return {
        data,
        metadata: {
            filesProcessed: processedFiles,
            pipelinesFound,
            warnings,
            processingTime
        }
    };
}

/**
 * Batch detect file types for multiple files
 */
export async function batchDetectFileTypes(files: File[]): Promise<DetectedFile[]> {
    const results = await Promise.all(
        files.map(file => detectFileType(file))
    );

    // Sort by confidence (highest first)
    return results.sort((a, b) => b.confidence - a.confidence);
}

/**
 * Validates that required files are present
 */
export function validateFileSet(detectedFiles: DetectedFile[]): {
    isValid: boolean;
    missing: string[];
    suggestions: string[];
} {
    const types = new Set(detectedFiles.map(f => f.type));
    const missing: string[] = [];
    const suggestions: string[] = [];

    if (!types.has('video')) {
        missing.push('Video file');
        suggestions.push('Add a video file (.mp4, .webm, .avi, .mov)');
    }

    const hasPipelineData = types.has('person_tracking') ||
        types.has('speech_recognition') ||
        types.has('speaker_diarization') ||
        types.has('scene_detection');

    if (!hasPipelineData) {
        missing.push('Pipeline data');
        suggestions.push('Add at least one annotation file (.json for tracking/scenes, .vtt for speech, .rttm for speakers)');
    }

    return {
        isValid: missing.length === 0,
        missing,
        suggestions
    };
}

/**
 * Gets summary of detected files for UI display
 */
export function getFilesSummary(detectedFiles: DetectedFile[]): {
    video?: string;
    audio?: string;
    pipelines: Array<{ name: string; file: string; confidence: number }>;
    unknown: string[];
} {
    const summary: {
        video?: string;
        audio?: string;
        pipelines: Array<{ name: string; file: string; confidence: number }>;
        unknown: string[];
    } = {
        pipelines: [],
        unknown: []
    };

    for (const detected of detectedFiles) {
        switch (detected.type) {
            case 'video':
                summary.video = detected.file.name;
                break;
            case 'audio':
                summary.audio = detected.file.name;
                break;
            case 'person_tracking':
            case 'speech_recognition':
            case 'speaker_diarization':
            case 'scene_detection':
                summary.pipelines.push({
                    name: detected.pipeline || detected.type,
                    file: detected.file.name,
                    confidence: detected.confidence
                });
                break;
            case 'unknown':
                summary.unknown.push(detected.file.name);
                break;
        }
    }

    return summary;
}
