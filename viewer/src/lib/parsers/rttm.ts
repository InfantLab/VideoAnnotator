// RTTM parser for speaker diarization data
// Reference: https://github.com/nryant/dscore#rttm
// VideoAnnotator produces RTTM files for speaker diarization

import type { RTTMSegment } from '@/types/annotations';
import { validateRTTMData } from '@/lib/validation';

/**
 * RTTM format specification:
 * SPEAKER <file> <channel> <start> <duration> <ortho> <stype> <name> <conf> <slat>
 * 
 * Fields:
 * - type: Always "SPEAKER" for speaker diarization
 * - file: File identifier
 * - channel: Channel number (usually 1)
 * - start: Start time in seconds
 * - duration: Duration in seconds
 * - ortho: Orthographic transcription (usually <NA>)
 * - stype: Speaker type (usually <NA>)
 * - name: Speaker ID
 * - conf: Confidence score (usually <NA> or number)
 * - slat: Speaker language (usually <NA>)
 */

/**
 * Parses a single RTTM line
 */
function parseRTTMLine(line: string, lineNumber: number): RTTMSegment | null {
    const parts = line.trim().split(/\s+/);

    if (parts.length < 9) {
        throw new Error(`Invalid RTTM line ${lineNumber}: expected at least 9 fields, got ${parts.length}`);
    }

    const [type, file_id, channel, startStr, durationStr, ortho, stype, speaker_id, confStr] = parts;

    // Validate type
    if (type !== 'SPEAKER') {
        // Skip non-speaker lines (might be other types in extended RTTM)
        return null;
    }

    // Parse numeric values
    const start_time = parseFloat(startStr);
    const duration = parseFloat(durationStr);

    if (isNaN(start_time) || isNaN(duration)) {
        throw new Error(`Invalid RTTM line ${lineNumber}: start_time and duration must be numbers`);
    }

    if (start_time < 0 || duration <= 0) {
        throw new Error(`Invalid RTTM line ${lineNumber}: start_time must be >= 0 and duration must be > 0`);
    }

    // Parse confidence (might be <NA> or a number)
    let confidence = 1.0; // Default confidence
    if (confStr && confStr !== '<NA>' && confStr !== '<na>') {
        const parsedConf = parseFloat(confStr);
        if (!isNaN(parsedConf)) {
            confidence = Math.max(0, Math.min(1, parsedConf)); // Clamp to [0, 1]
        }
    }

    const end_time = start_time + duration;

    return {
        file_id,
        start_time,
        duration,
        end_time,
        speaker_id,
        confidence,
        pipeline: 'speaker_diarization',
        format: 'rttm'
    };
}

/**
 * Parses RTTM file content and returns array of speaker segments
 */
export async function parseRTTM(file: File): Promise<RTTMSegment[]> {
    try {
        const content = await file.text();

        // Split into lines and filter out empty lines and comments
        const lines = content
            .split('\n')
            .map(line => line.trim())
            .filter(line => line.length > 0 && !line.startsWith('#'));

        if (lines.length === 0) {
            throw new Error('RTTM file is empty or contains no valid data');
        }

        // Parse each line
        const segments: RTTMSegment[] = [];

        for (let i = 0; i < lines.length; i++) {
            try {
                const segment = parseRTTMLine(lines[i], i + 1);
                if (segment) {
                    segments.push(segment);
                }
            } catch (error) {
                console.warn(`Skipping invalid RTTM line ${i + 1}: ${error}`);
                // Continue parsing other lines instead of failing completely
            }
        }

        if (segments.length === 0) {
            throw new Error('No valid speaker segments found in RTTM file');
        }

        // Validate parsed data
        const validatedSegments = validateRTTMData(segments);

        // Sort by start time
        validatedSegments.sort((a, b) => a.start_time - b.start_time);

        return validatedSegments;

    } catch (error) {
        throw new Error(`Failed to parse RTTM file: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
}

/**
 * Converts RTTM segments to timeline events for the viewer
 */
export function rttmToTimelineEvents(segments: RTTMSegment[]) {
    return segments.map((segment, index) => ({
        id: `speaker_${index}`,
        type: 'speaker' as const,
        label: segment.speaker_id,
        startTime: segment.start_time,
        endTime: segment.end_time,
        confidence: segment.confidence,
        metadata: {
            duration: segment.duration,
            file_id: segment.file_id
        }
    }));
}

/**
 * Gets active speakers at a specific time
 */
export function getActiveSpeakersAtTime(segments: RTTMSegment[], currentTime: number): string[] {
    return segments
        .filter(segment =>
            currentTime >= segment.start_time &&
            currentTime <= segment.end_time
        )
        .map(segment => segment.speaker_id);
}

/**
 * Gets speaker timeline for visualization
 */
export function getSpeakerTimeline(segments: RTTMSegment[]): Array<{
    speaker: string;
    segments: Array<{ start: number; end: number; confidence: number }>;
}> {
    const speakerMap = new Map<string, Array<{ start: number; end: number; confidence: number }>>();

    for (const segment of segments) {
        if (!speakerMap.has(segment.speaker_id)) {
            speakerMap.set(segment.speaker_id, []);
        }

        speakerMap.get(segment.speaker_id)!.push({
            start: segment.start_time,
            end: segment.end_time,
            confidence: segment.confidence
        });
    }

    return Array.from(speakerMap.entries()).map(([speaker, segments]) => ({
        speaker,
        segments: segments.sort((a, b) => a.start - b.start)
    }));
}

/**
 * Validates RTTM file format without fully parsing
 */
export async function isValidRTTM(file: File): Promise<boolean> {
    try {
        const firstLines = await file.slice(0, 1000).text();
        const lines = firstLines.split('\n').filter(line => line.trim().length > 0);

        // Check if at least one line starts with SPEAKER
        return lines.some(line => line.trim().startsWith('SPEAKER'));
    } catch {
        return false;
    }
}

/**
 * Merges overlapping speaker segments from the same speaker
 */
export function mergeOverlappingSegments(segments: RTTMSegment[], threshold = 0.1): RTTMSegment[] {
    if (segments.length === 0) return segments;

    const sortedSegments = [...segments].sort((a, b) => a.start_time - b.start_time);
    const merged: RTTMSegment[] = [];

    for (const segment of sortedSegments) {
        const lastMerged = merged[merged.length - 1];

        if (
            lastMerged &&
            lastMerged.speaker_id === segment.speaker_id &&
            segment.start_time <= lastMerged.end_time + threshold
        ) {
            // Merge segments
            lastMerged.end_time = Math.max(lastMerged.end_time, segment.end_time);
            lastMerged.duration = lastMerged.end_time - lastMerged.start_time;
            lastMerged.confidence = Math.max(lastMerged.confidence, segment.confidence);
        } else {
            merged.push({ ...segment });
        }
    }

    return merged;
}
