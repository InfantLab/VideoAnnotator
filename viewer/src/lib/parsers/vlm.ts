// VLM frame annotation parser for VideoAnnotator's vlm_annotation pipeline
// Reference: https://github.com/InfantLab/VideoAnnotator
// Output is COCO-shaped JSON (same export path as scene_detection) with
// VLM-specific extension fields (label, reasoning, sampling_mode, ...).

import type { VLMFrameAnnotation } from '@/types/annotations';
import { validateVlmAnnotationData } from '@/lib/validation';

/**
 * Parses a vlm_annotation pipeline output JSON file.
 */
export async function parseVlmAnnotations(file: File): Promise<VLMFrameAnnotation[]> {
    try {
        const content = await file.text();
        const data = JSON.parse(content);

        let annotations: unknown[];

        if (Array.isArray(data)) {
            annotations = data;
        } else if (data.annotations && Array.isArray(data.annotations)) {
            // COCO-shaped export (the pipeline's actual output format)
            annotations = data.annotations;
        } else if (data.results && Array.isArray(data.results)) {
            annotations = data.results;
        } else {
            throw new Error('Invalid VLM annotation format: expected array or object with annotations/results array');
        }

        if (annotations.length === 0) {
            console.warn('VLM annotation file contains no annotations');
            return [];
        }

        const validatedData = validateVlmAnnotationData(annotations);
        validatedData.sort((a, b) => a.timestamp_sec - b.timestamp_sec);

        return validatedData;
    } catch (error) {
        if (error instanceof SyntaxError) {
            throw new Error('Invalid JSON in VLM annotation file');
        }
        throw new Error(`Failed to parse VLM annotation data: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
}

/**
 * Gets the VLM annotation nearest the current time, within tolerance
 * (annotations are point samples, not intervals, so "active at time" means
 * "closest sample point within tolerance seconds").
 */
export function getVlmAnnotationAtTime(
    annotations: VLMFrameAnnotation[],
    currentTime: number,
    tolerance = 2.5
): VLMFrameAnnotation | null {
    let closest: VLMFrameAnnotation | null = null;
    let closestDelta = Infinity;

    for (const annotation of annotations) {
        const delta = Math.abs(annotation.timestamp_sec - currentTime);
        if (delta < closestDelta) {
            closest = annotation;
            closestDelta = delta;
        }
    }

    return closest && closestDelta <= tolerance ? closest : null;
}

/**
 * Validates VLM annotation file format without fully parsing (used by the
 * merger's file-type detector).
 */
export async function isValidVlmAnnotations(file: File): Promise<boolean> {
    try {
        const sampleSize = Math.min(5000, file.size);
        const sample = await file.slice(0, sampleSize).text();

        // Signature keys unique to this pipeline's output — "reasoning" +
        // "sampling_mode" together don't collide with any other pipeline's
        // COCO extension fields (scene_detection uses "scene_type", not these).
        if (sample.includes('"sampling_mode"') && sample.includes('"reasoning"')) {
            return true;
        }

        if (file.name.includes('vlm_annotation')) {
            return true;
        }

        return false;
    } catch {
        return false;
    }
}
