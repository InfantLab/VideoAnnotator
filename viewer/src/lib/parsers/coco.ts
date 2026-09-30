// COCO format parser for person tracking data
// Reference: https://cocodataset.org/#format-data
// VideoAnnotator produces COCO format JSON for person tracking with keypoints

import type { COCOPersonAnnotation } from '@/types/annotations';
import { validateCOCOPersonData } from '@/lib/validation';
import { COCO_KEYPOINT_NAMES, COCO_SKELETON_CONNECTIONS, COCOVisibility } from '@/types/annotations';

/**
 * Parsed keypoint with semantic information
 */
export interface ParsedKeypoint {
    name: string;
    x: number;
    y: number;
    visibility: COCOVisibility;
    confidence: number; // Derived from visibility
}

/**
 * Parses COCO keypoints array into structured format
 */
function parseKeypoints(keypoints: number[]): ParsedKeypoint[] {
    if (keypoints.length !== 51) { // 17 keypoints * 3 values each
        throw new Error(`Invalid keypoints array: expected 51 values, got ${keypoints.length}`);
    }

    const parsed: ParsedKeypoint[] = [];

    for (let i = 0; i < 17; i++) {
        const x = keypoints[i * 3];
        const y = keypoints[i * 3 + 1];
        const visibility = keypoints[i * 3 + 2] as COCOVisibility;

        parsed.push({
            name: COCO_KEYPOINT_NAMES[i],
            x,
            y,
            visibility,
            confidence: visibility === COCOVisibility.LABELED_VISIBLE ? 1.0 :
                visibility === COCOVisibility.LABELED_NOT_VISIBLE ? 0.5 : 0.0
        });
    }

    return parsed;
}

/**
 * Parses COCO person tracking JSON file
 */
export async function parseCOCOPersonData(file: File): Promise<COCOPersonAnnotation[]> {
    try {
        const content = await file.text();
        const data = JSON.parse(content);

        // Handle different COCO JSON structures
        let annotations: unknown[];

        if (Array.isArray(data)) {
            // Direct array of annotations
            annotations = data;
        } else if (data.annotations && Array.isArray(data.annotations)) {
            // Standard COCO format with annotations array
            annotations = data.annotations;
        } else if (data.results && Array.isArray(data.results)) {
            // VideoAnnotator format with results array
            annotations = data.results;
        } else {
            throw new Error('Invalid COCO format: expected array of annotations or object with annotations/results array');
        }

        if (annotations.length === 0) {
            console.warn('COCO file contains no annotations');
            return [];
        }

        // Validate and parse annotations
        const validatedData = validateCOCOPersonData(annotations);

        // Sort by timestamp for efficient lookup
        validatedData.sort((a, b) => a.timestamp - b.timestamp);

        return validatedData;

    } catch (error) {
        if (error instanceof SyntaxError) {
            throw new Error('Invalid JSON in COCO person tracking file');
        }
        throw new Error(`Failed to parse COCO person data: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
}

/**
 * Groups COCO annotations by timestamp for efficient lookup
 */
export function groupByTimestamp(annotations: COCOPersonAnnotation[]): Map<number, COCOPersonAnnotation[]> {
    const groups = new Map<number, COCOPersonAnnotation[]>();

    for (const annotation of annotations) {
        const timestamp = Math.round(annotation.timestamp * 1000) / 1000; // Round to milliseconds

        if (!groups.has(timestamp)) {
            groups.set(timestamp, []);
        }

        groups.get(timestamp)!.push(annotation);
    }

    return groups;
}

/**
 * Gets person annotations for a specific time with interpolation
 */
export function getPersonsAtTime(
    annotations: COCOPersonAnnotation[],
    currentTime: number,
    tolerance = 0.1
): COCOPersonAnnotation[] {
    // Find annotations within tolerance
    return annotations.filter(
        annotation => Math.abs(annotation.timestamp - currentTime) <= tolerance
    );
}

/**
 * Converts COCO annotation to legacy PersonPose format for backward compatibility
 */
export function cocoToLegacyPose(annotation: COCOPersonAnnotation) {
    const keypoints = parseKeypoints(annotation.keypoints);

    return {
        id: annotation.track_id || annotation.id,
        keypoints: keypoints.map(kp => ({
            x: kp.x,
            y: kp.y,
            confidence: kp.confidence
        })),
        connections: COCO_SKELETON_CONNECTIONS,
        confidence: annotation.score
    };
}

/**
 * Gets visible keypoints for rendering
 */
export function getVisibleKeypoints(annotation: COCOPersonAnnotation): ParsedKeypoint[] {
    const keypoints = parseKeypoints(annotation.keypoints);
    return keypoints.filter(kp => kp.visibility === COCOVisibility.LABELED_VISIBLE);
}

/**
 * Gets skeleton connections that should be drawn
 */
export function getDrawableConnections(annotation: COCOPersonAnnotation): Array<{
    from: ParsedKeypoint;
    to: ParsedKeypoint;
    confidence: number;
}> {
    const keypoints = parseKeypoints(annotation.keypoints);
    const connections: Array<{ from: ParsedKeypoint; to: ParsedKeypoint; confidence: number }> = [];

    for (const [fromIdx, toIdx] of COCO_SKELETON_CONNECTIONS) {
        const fromKp = keypoints[fromIdx];
        const toKp = keypoints[toIdx];

        // Only draw connection if both keypoints are visible
        if (fromKp.visibility === COCOVisibility.LABELED_VISIBLE &&
            toKp.visibility === COCOVisibility.LABELED_VISIBLE) {
            connections.push({
                from: fromKp,
                to: toKp,
                confidence: Math.min(fromKp.confidence, toKp.confidence)
            });
        }
    }

    return connections;
}

/**
 * Gets bounding box with padding for person
 */
export function getBoundingBoxWithPadding(annotation: COCOPersonAnnotation, padding = 10) {
    const [x, y, width, height] = annotation.bbox;
    return {
        x: x - padding,
        y: y - padding,
        width: width + padding * 2,
        height: height + padding * 2
    };
}

/**
 * Validates COCO person tracking file format without fully parsing
 */
export async function isValidCOCOPersonData(file: File): Promise<boolean> {
    try {
        const sample = await file.slice(0, 2000).text();
        const data = JSON.parse(sample);

        // Check if it looks like COCO person data
        if (Array.isArray(data)) {
            return data.length === 0 || (
                data[0] &&
                typeof data[0] === 'object' &&
                'keypoints' in data[0] &&
                'bbox' in data[0]
            );
        }

        return (
            (data.annotations && Array.isArray(data.annotations)) ||
            (data.results && Array.isArray(data.results))
        );
    } catch {
        return false;
    }
}

/**
 * Gets track statistics for analysis
 */
export function getTrackStatistics(annotations: COCOPersonAnnotation[]): Array<{
    trackId: number;
    frameCount: number;
    duration: number;
    avgConfidence: number;
    avgKeypoints: number;
}> {
    const trackMap = new Map<number, COCOPersonAnnotation[]>();

    // Group by track ID
    for (const annotation of annotations) {
        const trackId = annotation.track_id || annotation.id;
        if (!trackMap.has(trackId)) {
            trackMap.set(trackId, []);
        }
        trackMap.get(trackId)!.push(annotation);
    }

    // Calculate statistics
    return Array.from(trackMap.entries()).map(([trackId, track]) => {
        const sortedTrack = track.sort((a, b) => a.timestamp - b.timestamp);
        const duration = sortedTrack.length > 1 ?
            sortedTrack[sortedTrack.length - 1].timestamp - sortedTrack[0].timestamp : 0;

        const avgConfidence = track.reduce((sum, ann) => sum + ann.score, 0) / track.length;
        const avgKeypoints = track.reduce((sum, ann) => sum + ann.num_keypoints, 0) / track.length;

        return {
            trackId,
            frameCount: track.length,
            duration,
            avgConfidence,
            avgKeypoints
        };
    });
}
