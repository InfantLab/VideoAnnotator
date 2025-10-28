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
    frame_number: z.number().nonnegative(),
    // NEW v1.1.1 fields
    person_id: z.string(),
    person_label: z.string(),
    label_confidence: z.number(),
    labeling_method: z.string()
});

// =============================================================================
// WEBVTT FORMAT VALIDATION
// =============================================================================

export const WebVTTCueSchema = z.object({
    id: z.string().optional(),
    startTime: z.number().nonnegative().optional(),
    endTime: z.number().nonnegative().optional(),
    text: z.string().optional(),
    settings: z.string().optional()
});

// =============================================================================
// RTTM FORMAT VALIDATION
// =============================================================================

export const RTTMSegmentSchema = z.object({
    file_id: z.string().optional(),
    start_time: z.number().nonnegative().optional(),
    duration: z.number().positive().optional(),
    end_time: z.number().nonnegative().optional(),
    speaker_id: z.string().optional(),
    confidence: z.number().min(0).max(1).optional(),
    pipeline: z.literal('speaker_diarization').optional(),
    format: z.literal('rttm').optional()
});

// =============================================================================
// SCENE DETECTION VALIDATION
// =============================================================================

export const SceneAnnotationSchema = z.object({
    id: z.number().optional(),
    image_id: z.string().optional(),
    category_id: z.number().optional(),
    bbox: z.tuple([z.number(), z.number(), z.number(), z.number()]).optional(),
    area: z.number().nonnegative().optional(),
    iscrowd: z.union([z.literal(0), z.literal(1)]).optional(),
    score: z.number().min(0).max(1).optional(),
    video_id: z.string().optional(),
    timestamp: z.number().nonnegative().optional(),
    start_time: z.number().nonnegative().optional(),
    end_time: z.number().nonnegative().optional(),
    duration: z.number().positive().optional(),
    scene_type: z.string().optional(),
    frame_start: z.number().nonnegative().optional(),
    frame_end: z.number().nonnegative().optional(),
    all_scores: z.record(z.string(), z.number()).optional()
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
        filename: z.string().optional(),
        duration: z.number().positive().optional(),
        width: z.number().positive().optional(),
        height: z.number().positive().optional(),
        frame_rate: z.number().positive().optional()
    }).optional(),
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
                (error.errors[0] as any)?.received
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
                (error.errors[0] as any)?.received
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
                (error.errors[0] as any)?.received
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
                (error.errors[0] as any)?.received
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
                (error.errors[0] as any)?.received
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

// =============================================================================
// VIDEOANNOTATOR v1.3.0 API VALIDATION
// =============================================================================

/**
 * Field-level error from config validation
 */
export const FieldErrorSchema = z.object({
    field: z.string(),
    message: z.string(),
    error_code: z.string().optional(),
    hint: z.string().optional(),
}).strict();

/**
 * Error envelope (v1.3.0 standard format)
 */
export const ErrorEnvelopeSchema = z.object({
    error: z.union([z.string(), z.array(FieldErrorSchema)]),
    error_code: z.string().optional(),
    request_id: z.string().optional(),
    hint: z.string().optional(),
});

/**
 * Job cancellation response
 */
export const JobCancellationResponseSchema = z.object({
    job_id: z.string(),
    status: z.enum(['cancelled', 'cancelling']),
    message: z.string(),
    cancelled_at: z.string().optional(),
});

/**
 * Validation issue
 */
export const ValidationIssueSchema = z.object({
    field: z.string(),
    message: z.string(),
    severity: z.enum(['error', 'warning', 'info']),
    error_code: z.string().optional(),
    hint: z.string().optional(),
    suggested_value: z.unknown().optional(),
});

/**
 * Config validation result
 */
export const ConfigValidationResultSchema = z.object({
    valid: z.boolean(),
    errors: z.array(ValidationIssueSchema),
    warnings: z.array(ValidationIssueSchema),
    validated_config: z.record(z.unknown()).optional(),
});

/**
 * GPU status
 */
export const GpuStatusSchema = z.object({
    available: z.boolean(),
    device_name: z.string().optional(),
    cuda_version: z.string().optional(),
    memory_total: z.number().optional(),
    memory_used: z.number().optional(),
    memory_free: z.number().optional(),
});

/**
 * Worker info
 */
export const WorkerInfoSchema = z.object({
    active_jobs: z.number(),
    queued_jobs: z.number(),
    max_concurrent_jobs: z.number(),
    worker_status: z.enum(['idle', 'busy', 'overloaded']),
});

/**
 * System diagnostics
 */
export const SystemDiagnosticsSchema = z.object({
    database: z.object({
        status: z.enum(['healthy', 'degraded', 'unhealthy']),
        message: z.string().optional(),
    }),
    storage: z.object({
        status: z.enum(['healthy', 'degraded', 'unhealthy']),
        disk_usage_percent: z.number().optional(),
        message: z.string().optional(),
    }),
    ffmpeg: z.object({
        available: z.boolean(),
        version: z.string().optional(),
    }),
});

/**
 * Enhanced health response (v1.3.0)
 */
export const EnhancedHealthResponseSchema = z.object({
    status: z.enum(['healthy', 'degraded', 'unhealthy']),
    version: z.string(),
    api_version: z.string(),
    uptime_seconds: z.number(),
    gpu_status: GpuStatusSchema.optional(),
    worker_info: WorkerInfoSchema.optional(),
    diagnostics: SystemDiagnosticsSchema.optional(),
    message: z.string().optional(),
});

/**
 * Legacy health response (v1.2.x)
 */
export const LegacyHealthResponseSchema = z.object({
    status: z.string(),
    api_version: z.string().optional(),
    videoannotator_version: z.string().optional(),
    message: z.string().optional(),
});

/**
 * Union health response schema
 */
export const HealthResponseSchema = z.union([
    EnhancedHealthResponseSchema,
    LegacyHealthResponseSchema,
]);
