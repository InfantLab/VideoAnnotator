// Validation schemas for VideoAnnotator standard formats
// Using Zod for runtime validation and type safety
// Reference: https://github.com/InfantLab/VideoAnnotator

import { z } from 'zod';
import type {
    COCOPersonAnnotation,
    WebVTTCue,
    RTTMSegment,
    SceneAnnotation,
    StandardAnnotationData,
    PipelineResult,
    SupportedFileType
} from '@/types/annotations';

// =============================================================================
// COCO FORMAT VALIDATION
// =============================================================================

export const COCOPersonAnnotationSchema = z.object({
    id: z.number(),
    image_id: z.string(),
    category_id: z.number(),
    keypoints: z.array(z.number()).length(51), // 17 keypoints * 3 (x, y, visibility)
    num_keypoints: z.number().min(0).max(17),
    bbox: z.tuple([z.number(), z.number(), z.number(), z.number()]),
    area: z.number().nonnegative(),
    iscrowd: z.union([z.literal(0), z.literal(1)]),
    score: z.number().min(0).max(1),
    track_id: z.number().optional(),
    timestamp: z.number().nonnegative(),
    frame_number: z.number().nonnegative()
});

// =============================================================================
// WEBVTT FORMAT VALIDATION
// =============================================================================

export const WebVTTCueSchema = z.object({
    id: z.string().optional(),
    startTime: z.number().nonnegative(),
    endTime: z.number().nonnegative(),
    text: z.string(),
    settings: z.string().optional()
}).refine(data => data.endTime > data.startTime, {
    message: "End time must be greater than start time"
});

// =============================================================================
// RTTM FORMAT VALIDATION
// =============================================================================

export const RTTMSegmentSchema = z.object({
    file_id: z.string(),
    start_time: z.number().nonnegative(),
    duration: z.number().positive(),
    end_time: z.number().nonnegative(),
    speaker_id: z.string(),
    confidence: z.number().min(0).max(1),
    pipeline: z.literal('speaker_diarization'),
    format: z.literal('rttm')
}).refine(data => Math.abs((data.start_time + data.duration) - data.end_time) < 0.001, {
    message: "End time must equal start time plus duration"
});

// =============================================================================
// SCENE DETECTION VALIDATION
// =============================================================================

export const SceneAnnotationSchema = z.object({
    id: z.number(),
    video_id: z.string(),
    timestamp: z.number().nonnegative(),
    start_time: z.number().nonnegative(),
    end_time: z.number().nonnegative(),
    duration: z.number().positive(),
    scene_type: z.string(),
    bbox: z.tuple([z.number(), z.number(), z.number(), z.number()]),
    score: z.number().min(0).max(1),
    frame_start: z.number().nonnegative(),
    frame_end: z.number().nonnegative(),
    all_scores: z.record(z.string(), z.number())
});

// =============================================================================
// PIPELINE RESULT VALIDATION
// =============================================================================

export const PipelineResultSchema = <T>(dataSchema: z.ZodSchema<T>) => z.object({
    pipeline: z.string(),
    format: z.string(),
    data: z.array(dataSchema),
    metadata: z.object({
        pipeline_name: z.string(),
        output_format: z.string(),
        processed_segments: z.number().nonnegative()
    })
});

// =============================================================================
// UNIFIED ANNOTATION DATA VALIDATION
// =============================================================================

export const StandardAnnotationDataSchema = z.object({
    video_info: z.object({
        filename: z.string(),
        duration: z.number().positive(),
        width: z.number().positive(),
        height: z.number().positive(),
        frame_rate: z.number().positive().optional()
    }),
    person_tracking: z.array(COCOPersonAnnotationSchema).optional(),
    speech_recognition: z.array(WebVTTCueSchema).optional(),
    speaker_diarization: z.array(RTTMSegmentSchema).optional(),
    scene_detection: z.array(SceneAnnotationSchema).optional(),
    audio_file: z.instanceof(File).optional(),
    metadata: z.object({
        created: z.string(),
        version: z.string(),
        pipelines: z.array(z.string()),
        source: z.enum(['videoannotator', 'custom'])
    }).optional()
});

// =============================================================================
// FILE TYPE VALIDATION
// =============================================================================

export const SupportedFileTypeSchema = z.enum([
    'video/mp4',
    'video/webm',
    'video/avi',
    'video/mov',
    'text/vtt',
    'text/plain',
    'application/json',
    'audio/wav',
    'audio/mpeg'
]);

// =============================================================================
// VALIDATION FUNCTIONS
// =============================================================================

export class ValidationError extends Error {
    constructor(
        message: string,
        public readonly field?: string,
        public readonly value?: unknown
    ) {
        super(message);
        this.name = 'ValidationError';
    }
}

/**
 * Validates COCO person tracking data
 */
export function validateCOCOPersonData(data: unknown[]): COCOPersonAnnotation[] {
    try {
        return z.array(COCOPersonAnnotationSchema).parse(data);
    } catch (error) {
        if (error instanceof z.ZodError) {
            throw new ValidationError(
                `Invalid COCO person tracking data: ${error.errors[0]?.message}`,
                error.errors[0]?.path.join('.'),
                error.errors[0]?.received
            );
        }
        throw error;
    }
}

/**
 * Validates WebVTT subtitle data
 */
export function validateWebVTTData(data: unknown[]): WebVTTCue[] {
    try {
        return z.array(WebVTTCueSchema).parse(data);
    } catch (error) {
        if (error instanceof z.ZodError) {
            throw new ValidationError(
                `Invalid WebVTT data: ${error.errors[0]?.message}`,
                error.errors[0]?.path.join('.'),
                error.errors[0]?.received
            );
        }
        throw error;
    }
}

/**
 * Validates RTTM speaker diarization data
 */
export function validateRTTMData(data: unknown[]): RTTMSegment[] {
    try {
        return z.array(RTTMSegmentSchema).parse(data);
    } catch (error) {
        if (error instanceof z.ZodError) {
            throw new ValidationError(
                `Invalid RTTM data: ${error.errors[0]?.message}`,
                error.errors[0]?.path.join('.'),
                error.errors[0]?.received
            );
        }
        throw error;
    }
}

/**
 * Validates scene detection data
 */
export function validateSceneData(data: unknown[]): SceneAnnotation[] {
    try {
        return z.array(SceneAnnotationSchema).parse(data);
    } catch (error) {
        if (error instanceof z.ZodError) {
            throw new ValidationError(
                `Invalid scene detection data: ${error.errors[0]?.message}`,
                error.errors[0]?.path.join('.'),
                error.errors[0]?.received
            );
        }
        throw error;
    }
}

/**
 * Validates complete annotation data structure
 */
export function validateStandardAnnotationData(data: unknown): StandardAnnotationData {
    try {
        return StandardAnnotationDataSchema.parse(data);
    } catch (error) {
        if (error instanceof z.ZodError) {
            throw new ValidationError(
                `Invalid annotation data: ${error.errors[0]?.message}`,
                error.errors[0]?.path.join('.'),
                error.errors[0]?.received
            );
        }
        throw error;
    }
}

/**
 * Validates file type
 */
export function validateFileType(mimeType: string): SupportedFileType {
    try {
        return SupportedFileTypeSchema.parse(mimeType);
    } catch (error) {
        throw new ValidationError(
            `Unsupported file type: ${mimeType}`,
            'fileType',
            mimeType
        );
    }
}

/**
 * Gets user-friendly error message for validation errors
 */
export function getValidationErrorMessage(error: ValidationError): string {
    const baseMessage = error.message;

    if (error.field) {
        return `${baseMessage} (field: ${error.field})`;
    }

    return baseMessage;
}

/**
 * Checks if data matches expected VideoAnnotator output format
 */
export function isVideoAnnotatorFormat(data: unknown): boolean {
    try {
        if (Array.isArray(data)) {
            // Check if it's a pipeline result array
            return data.every(item =>
                typeof item === 'object' &&
                item !== null &&
                'pipeline' in item &&
                'format' in item &&
                'data' in item
            );
        }

        if (typeof data === 'object' && data !== null) {
            // Check if it's a single pipeline result or complete results
            return 'pipeline_results' in data || 'pipeline' in data;
        }

        return false;
    } catch {
        return false;
    }
}
