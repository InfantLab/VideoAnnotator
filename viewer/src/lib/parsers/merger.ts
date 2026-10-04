// Data merger utility for combining VideoAnnotator pipeline outputs
// Merges person tracking, speech recognition, speaker diarization, and scene detection
// into a unified StandardAnnotationData structure

import type {
    StandardAnnotationData,
    COCOPersonAnnotation,
    WebVTTCue,
    RTTMSegment,
    SceneAnnotation,
    VLMFrameAnnotation,
    ElanTierAnnotation,
    LAIONFaceAnnotation,
    StandardFaceAnnotation,
    VideoAnnotatorCompleteResults
} from '@/types/annotations';

import { parseWebVTT } from './webvtt';
import { parseRTTM } from './rttm';
import { parseCOCOPersonData } from './coco';
import { parseSceneDetection } from './scene';
import { parseVlmAnnotations } from './vlm';
import { parseElanFile } from './elan';
import { parseCOCOOpenFace3Data } from './cocoOpenface3';
// import { parseFaceAnalysis } from './face'; // Using local implementation

import type { DetectedFile } from '../fileDetection';

// Detection lives in lib/fileDetection.ts; re-exported for existing importers.
export { detectFileType, detectJSONStructure, type DetectedFile } from '../fileDetection';

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
 * Parses VideoAnnotator v1.1.1 complete results format
 */
async function parseCompleteResults(file: File): Promise<{
    personTracking: COCOPersonAnnotation[];
    faceAnalysis: LAIONFaceAnnotation[];
    sceneDetection: SceneAnnotation[];
    config: VideoAnnotatorCompleteResults['config'];
    processingTime: number;
    totalDuration: number;
}> {
    const text = await file.text();
    const data: VideoAnnotatorCompleteResults = JSON.parse(text);

    // DEBUG: Log raw data structure
    console.log('🔍 parseCompleteResults: Raw data structure');
    console.log('  - pipeline_results keys:', Object.keys(data.pipeline_results || {}));
    console.log('  - person section exists:', !!data.pipeline_results.person);
    console.log('  - person results count:', data.pipeline_results.person?.results?.length || 0);

    const personTracking = data.pipeline_results.person?.results || [];
    const faceAnalysis = data.pipeline_results.face?.results || [];
    const sceneDetection = data.pipeline_results.scene?.results || [];

    console.log('🔍 parseCompleteResults: Parsed arrays');
    console.log('  - personTracking length:', personTracking.length);
    console.log('  - faceAnalysis length:', faceAnalysis.length);
    console.log('  - sceneDetection length:', sceneDetection.length);

    return {
        personTracking,
        faceAnalysis,
        sceneDetection,
        config: data.config,
        processingTime: Object.values(data.pipeline_results).reduce((sum, result) => sum + (result?.processing_time || 0), 0),
        totalDuration: data.total_duration
    };
}

/**
 * Parses face analysis file (LAION format)
 */
async function parseFaceAnalysis(file: File): Promise<LAIONFaceAnnotation[]> {
    const text = await file.text();
    const data = JSON.parse(text);

    if (Array.isArray(data)) {
        return data;
    }

    if (data.annotations && Array.isArray(data.annotations)) {
        return data.annotations;
    }

    if (data.results && Array.isArray(data.results)) {
        return data.results;
    }

    return [];
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
    // DEBUG: Log incoming files
    console.log('🔍 mergeAnnotationData called with', detectedFiles.length, 'detected files:');
    detectedFiles.forEach((df, i) => {
        console.log(`  ${i}: ${df.file.name} -> type: ${df.type}, pipeline: ${df.pipeline}, confidence: ${df.confidence}`);
    });

    const startTime = Date.now();
    const warnings: string[] = [];
    const pipelinesFound: string[] = [];

    let videoFile: File | undefined;
    let audioFile: File | undefined;
    let personTracking: COCOPersonAnnotation[] = [];
    let speechRecognition: WebVTTCue[] = [];
    let speakerDiarization: RTTMSegment[] = [];
    let sceneDetection: SceneAnnotation[] = [];
    let vlmAnnotations: VLMFrameAnnotation[] = [];
    let elanGroundTruth: ElanTierAnnotation[] = [];
    let faceAnalysis: LAIONFaceAnnotation[] = [];
    let openface3Faces: StandardFaceAnnotation[] = []; // OpenFace3 faces data

    // Processing metadata from VideoAnnotator v1.1.1
    let processingConfig: VideoAnnotatorCompleteResults['config'] | undefined;
    let processingTime: number | undefined;
    let totalDuration: number | undefined;

    const totalFiles = detectedFiles.length;
    let processedFiles = 0;

    // Check for complete results file first (highest priority)
    const completeResultsFile = detectedFiles.find(f => f.type === 'complete_results');
    
    if (completeResultsFile) {
        try {
            onProgress?.(`Processing ${completeResultsFile.file.name}`, processedFiles, totalFiles);
            
            const completeResults = await parseCompleteResults(completeResultsFile.file);
            personTracking = completeResults.personTracking;
            faceAnalysis = completeResults.faceAnalysis;
            sceneDetection = completeResults.sceneDetection;
            processingConfig = completeResults.config;
            
            // DEBUG: Log parsing results
            console.log('🔍 Complete results parsed:');
            console.log('  - Person tracking entries:', personTracking?.length || 0);
            console.log('  - Face analysis entries:', faceAnalysis?.length || 0);
            console.log('  - Scene detection entries:', sceneDetection?.length || 0);
            if (personTracking && personTracking.length > 0) {
                console.log('  - First person entry:', personTracking[0]);
            }
            processingTime = completeResults.processingTime;
            totalDuration = completeResults.totalDuration;

            if (personTracking.length > 0) pipelinesFound.push('person_tracking');
            if (faceAnalysis.length > 0) pipelinesFound.push('face_analysis');
            if (sceneDetection.length > 0) pipelinesFound.push('scene_detection');

            processedFiles++;
        } catch (error) {
            warnings.push(`Failed to parse complete results ${completeResultsFile.file.name}: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
    }

    // Process remaining files (for speech/speaker data or if no complete results)
    for (const detectedFile of detectedFiles) {
        if (detectedFile === completeResultsFile) continue; // Skip already processed

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
                    if (personTracking.length === 0) { // Only if not from complete results
                        personTracking = await parseCOCOPersonData(detectedFile.file);
                        pipelinesFound.push('person_tracking');
                    }
                    break;

                case 'face_analysis':
                    if (faceAnalysis.length === 0) { // Only if not from complete results
                        faceAnalysis = await parseFaceAnalysis(detectedFile.file);
                        pipelinesFound.push('face_analysis');
                    }
                    break;

                case 'openface3_faces':
                    if (openface3Faces.length === 0) {
                        // Check if it's COCO+OpenFace3 format or native OpenFace3 format
                        const fileContent = await detectedFile.file.text();
                        const data = JSON.parse(fileContent);
                        
                        if (data.info && data.images && data.annotations && data.annotations[0]?.openface3) {
                            // COCO+OpenFace3 format (VideoAnnotator export)
                            openface3Faces = await parseCOCOOpenFace3Data(detectedFile.file);
                        } else {
                            // Native OpenFace3 format
                            const { OpenFace3Parser } = await import('./openface3Parser');
                            const parser = OpenFace3Parser.getInstance();
                            openface3Faces = parser.parseOpenFace3Data(data);
                        }
                        pipelinesFound.push('openface3');
                    }
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
                    if (sceneDetection.length === 0) { // Only if not from complete results
                        sceneDetection = await parseSceneDetection(detectedFile.file);
                        pipelinesFound.push('scene_detection');
                    }
                    break;

                case 'vlm_annotation':
                    if (vlmAnnotations.length === 0) {
                        vlmAnnotations = await parseVlmAnnotations(detectedFile.file);
                        pipelinesFound.push('vlm_annotation');
                    }
                    break;

                case 'elan_ground_truth':
                    if (elanGroundTruth.length === 0) {
                        elanGroundTruth = await parseElanFile(detectedFile.file);
                        pipelinesFound.push('elan_ground_truth');
                    }
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
            version: '1.1.1',
            pipelines: pipelinesFound,
            source: 'videoannotator',
            // NEW v1.1.1 metadata
            processing_config: processingConfig,
            processing_time: processingTime,
            total_duration: totalDuration
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

    if (vlmAnnotations.length > 0) {
        data.vlm_annotations = vlmAnnotations;
    }

    if (elanGroundTruth.length > 0) {
        data.elan_ground_truth = elanGroundTruth;
    }

    if (faceAnalysis.length > 0) {
        data.face_analysis = faceAnalysis;
    }

    if (openface3Faces.length > 0) {
        data.openface3_faces = openface3Faces;
    }

    if (audioFile) {
        data.audio_file = audioFile;
    }

    const totalProcessingTime = Date.now() - startTime;

    onProgress?.('Merging complete', totalFiles + 1, totalFiles + 1);

    return {
        data,
        metadata: {
            filesProcessed: processedFiles,
            pipelinesFound,
            warnings,
            processingTime: totalProcessingTime
        }
    };
}
