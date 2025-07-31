/**
 * File utility functions for VideoAnnotator format detection and handling
 */

export interface FileTypeInfo {
    type: 'video' | 'audio' | 'person_tracking' | 'speech_recognition' | 'speaker_diarization' | 'scene_detection' | 'unknown';
    extension: string;
    mimeType?: string;
    confidence: 'high' | 'medium' | 'low';
    reason: string;
}

/**
 * Detect file type based on name, extension, and content analysis
 */
export function detectFileType(file: File): FileTypeInfo {
    const name = file.name.toLowerCase();
    const extension = name.split('.').pop() || '';
    const mimeType = file.type;

    // Video files
    if (mimeType.startsWith('video/') || ['mp4', 'webm', 'avi', 'mov', 'mkv'].includes(extension)) {
        return {
            type: 'video',
            extension,
            mimeType,
            confidence: 'high',
            reason: 'Video MIME type or extension'
        };
    }

    // Audio files
    if (mimeType.startsWith('audio/') || ['wav', 'mp3', 'aac', 'ogg'].includes(extension)) {
        return {
            type: 'audio',
            extension,
            mimeType,
            confidence: 'high',
            reason: 'Audio MIME type or extension'
        };
    }

    // WebVTT files (speech recognition)
    if (extension === 'vtt' || mimeType === 'text/vtt') {
        return {
            type: 'speech_recognition',
            extension,
            mimeType,
            confidence: 'high',
            reason: 'WebVTT format'
        };
    }

    // RTTM files (speaker diarization)
    if (extension === 'rttm') {
        return {
            type: 'speaker_diarization',
            extension,
            mimeType,
            confidence: 'high',
            reason: 'RTTM format'
        };
    }

    // JSON files - need content analysis
    if (extension === 'json' || mimeType === 'application/json') {
        // Will be determined by content analysis in detectJSONType
        return {
            type: 'unknown',
            extension,
            mimeType,
            confidence: 'low',
            reason: 'JSON file requires content analysis'
        };
    }

    return {
        type: 'unknown',
        extension,
        mimeType,
        confidence: 'low',
        reason: 'Unrecognized file type'
    };
}

/**
 * Analyze JSON content to determine annotation type
 */
export async function detectJSONType(file: File): Promise<FileTypeInfo> {
    try {
        const text = await file.text();
        const data = JSON.parse(text);

        // COCO format detection (person tracking)
        if (data.annotations && Array.isArray(data.annotations) &&
            data.annotations.length > 0 && data.annotations[0].keypoints) {
            return {
                type: 'person_tracking',
                extension: 'json',
                mimeType: 'application/json',
                confidence: 'high',
                reason: 'COCO format with keypoints detected'
            };
        }

        // Scene detection format
        if (Array.isArray(data) && data.length > 0 &&
            data[0].start_time !== undefined && data[0].end_time !== undefined) {
            return {
                type: 'scene_detection',
                extension: 'json',
                mimeType: 'application/json',
                confidence: 'high',
                reason: 'Scene detection format with timestamps'
            };
        }

        // Other JSON structures that might indicate specific types
        if (data.scenes && Array.isArray(data.scenes)) {
            return {
                type: 'scene_detection',
                extension: 'json',
                mimeType: 'application/json',
                confidence: 'medium',
                reason: 'JSON with scenes array'
            };
        }

        if (data.people && Array.isArray(data.people)) {
            return {
                type: 'person_tracking',
                extension: 'json',
                mimeType: 'application/json',
                confidence: 'medium',
                reason: 'JSON with people array'
            };
        }

        return {
            type: 'unknown',
            extension: 'json',
            mimeType: 'application/json',
            confidence: 'low',
            reason: 'Unknown JSON structure'
        };

    } catch (error) {
        return {
            type: 'unknown',
            extension: 'json',
            mimeType: 'application/json',
            confidence: 'low',
            reason: 'Invalid JSON or parsing error'
        };
    }
}

/**
 * Get human-readable file type description
 */
export function getFileTypeDescription(type: FileTypeInfo['type']): string {
    switch (type) {
        case 'video': return 'Video File';
        case 'audio': return 'Audio File';
        case 'person_tracking': return 'Person Tracking (COCO)';
        case 'speech_recognition': return 'Speech Recognition (WebVTT)';
        case 'speaker_diarization': return 'Speaker Diarization (RTTM)';
        case 'scene_detection': return 'Scene Detection (JSON)';
        default: return 'Unknown File Type';
    }
}

/**
 * Validate file size constraints
 */
export function validateFileSize(file: File): { valid: boolean; error?: string } {
    const maxSizes = {
        video: 500 * 1024 * 1024, // 500MB
        audio: 100 * 1024 * 1024, // 100MB
        annotation: 10 * 1024 * 1024 // 10MB for all annotation files
    };

    const typeInfo = detectFileType(file);

    if (typeInfo.type === 'video' && file.size > maxSizes.video) {
        return { valid: false, error: 'Video file too large (max 500MB)' };
    }

    if (typeInfo.type === 'audio' && file.size > maxSizes.audio) {
        return { valid: false, error: 'Audio file too large (max 100MB)' };
    }

    if (['person_tracking', 'speech_recognition', 'speaker_diarization', 'scene_detection'].includes(typeInfo.type)
        && file.size > maxSizes.annotation) {
        return { valid: false, error: 'Annotation file too large (max 10MB)' };
    }

    return { valid: true };
}

/**
 * Check if file set contains required components
 */
export function validateFileSet(files: File[]): { valid: boolean; missing: string[]; warnings: string[] } {
    const detectedTypes = files.map(f => detectFileType(f).type);
    const missing: string[] = [];
    const warnings: string[] = [];

    // Require at least one video file
    if (!detectedTypes.includes('video')) {
        missing.push('Video file (.mp4, .webm, .avi, .mov)');
    }

    // Warn if no annotation files
    const annotationTypes: Array<FileTypeInfo['type']> = ['person_tracking', 'speech_recognition', 'speaker_diarization', 'scene_detection'];
    const hasAnnotations = annotationTypes.some(type => detectedTypes.includes(type));

    if (!hasAnnotations) {
        warnings.push('No annotation files detected. Consider adding tracking, speech, or scene data.');
    }

    // Warn about unknown files
    const unknownCount = detectedTypes.filter(type => type === 'unknown').length;
    if (unknownCount > 0) {
        warnings.push(`${unknownCount} file(s) could not be identified. Check file formats.`);
    }

    return {
        valid: missing.length === 0,
        missing,
        warnings
    };
}

/**
 * Generate file processing summary
 */
export function generateFilesSummary(files: File[]): string {
    const typeInfo = files.map(f => detectFileType(f));
    const typeCounts = typeInfo.reduce((acc, info) => {
        acc[info.type] = (acc[info.type] || 0) + 1;
        return acc;
    }, {} as Record<FileTypeInfo['type'], number>);

    const parts: string[] = [];

    if (typeCounts.video) parts.push(`${typeCounts.video} video file(s)`);
    if (typeCounts.audio) parts.push(`${typeCounts.audio} audio file(s)`);
    if (typeCounts.person_tracking) parts.push(`${typeCounts.person_tracking} person tracking file(s)`);
    if (typeCounts.speech_recognition) parts.push(`${typeCounts.speech_recognition} speech recognition file(s)`);
    if (typeCounts.speaker_diarization) parts.push(`${typeCounts.speaker_diarization} speaker diarization file(s)`);
    if (typeCounts.scene_detection) parts.push(`${typeCounts.scene_detection} scene detection file(s)`);
    if (typeCounts.unknown) parts.push(`${typeCounts.unknown} unknown file(s)`);

    return parts.length > 0 ? parts.join(', ') : 'No files';
}
