// Scene detection parser for VideoAnnotator scene detection data
// Reference: https://github.com/InfantLab/VideoAnnotator
// VideoAnnotator produces simple JSON arrays for scene detection

import type { SceneAnnotation } from '@/types/annotations';
import { validateSceneData } from '@/lib/validation';

/**
 * Scene transition for timeline visualization
 */
export interface SceneTransition {
    id: number;
    timestamp: number;
    fromScene?: SceneAnnotation;
    toScene: SceneAnnotation;
    type: 'start' | 'transition' | 'end';
}

/**
 * Parses scene detection JSON file
 */
export async function parseSceneDetection(file: File): Promise<SceneAnnotation[]> {
    try {
        const content = await file.text();
        const data = JSON.parse(content);

        // Handle different scene detection data structures
        let scenes: unknown[];

        if (Array.isArray(data)) {
            // Direct array of scene annotations
            scenes = data;
        } else if (data.results && Array.isArray(data.results)) {
            // VideoAnnotator format with results array
            scenes = data.results;
        } else if (data.scenes && Array.isArray(data.scenes)) {
            // Alternative format with scenes array
            scenes = data.scenes;
        } else {
            throw new Error('Invalid scene detection format: expected array of scenes or object with scenes/results array');
        }

        if (scenes.length === 0) {
            console.warn('Scene detection file contains no scenes');
            return [];
        }

        // Validate and parse scenes
        const validatedData = validateSceneData(scenes);

        // Sort by start time
        validatedData.sort((a, b) => a.start_time - b.start_time);

        return validatedData;

    } catch (error) {
        if (error instanceof SyntaxError) {
            throw new Error('Invalid JSON in scene detection file');
        }
        throw new Error(`Failed to parse scene detection data: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
}

/**
 * Converts scene annotations to timeline events for the viewer
 */
export function scenesToTimelineEvents(scenes: SceneAnnotation[]) {
    return scenes.map((scene, index) => ({
        id: `scene_${scene.id}`,
        type: 'scene' as const,
        label: scene.scene_type || 'Unknown Scene',
        startTime: scene.start_time,
        endTime: scene.end_time,
        confidence: scene.score,
        metadata: {
            sceneId: scene.id,
            videoId: scene.video_id,
            duration: scene.duration,
            frameStart: scene.frame_start,
            frameEnd: scene.frame_end,
            bbox: scene.bbox,
            allScores: scene.all_scores
        }
    }));
}

/**
 * Gets the active scene at a specific time
 */
export function getSceneAtTime(scenes: SceneAnnotation[], currentTime: number): SceneAnnotation | null {
    return scenes.find(
        scene => currentTime >= scene.start_time && currentTime <= scene.end_time
    ) || null;
}

/**
 * Gets scene transitions for timeline visualization
 */
export function getSceneTransitions(scenes: SceneAnnotation[]): SceneTransition[] {
    if (scenes.length === 0) return [];

    const sortedScenes = [...scenes].sort((a, b) => a.start_time - b.start_time);
    const transitions: SceneTransition[] = [];

    // Add start transition for first scene
    transitions.push({
        id: 0,
        timestamp: sortedScenes[0].start_time,
        toScene: sortedScenes[0],
        type: 'start'
    });

    // Add transitions between scenes
    for (let i = 1; i < sortedScenes.length; i++) {
        const prevScene = sortedScenes[i - 1];
        const currentScene = sortedScenes[i];

        // Check for gap between scenes
        if (currentScene.start_time > prevScene.end_time) {
            // Add end transition for previous scene
            transitions.push({
                id: transitions.length,
                timestamp: prevScene.end_time,
                fromScene: prevScene,
                toScene: currentScene,
                type: 'transition'
            });
        }

        // Add start transition for current scene
        transitions.push({
            id: transitions.length,
            timestamp: currentScene.start_time,
            fromScene: prevScene,
            toScene: currentScene,
            type: 'transition'
        });
    }

    // Add end transition for last scene
    const lastScene = sortedScenes[sortedScenes.length - 1];
    transitions.push({
        id: transitions.length,
        timestamp: lastScene.end_time,
        fromScene: lastScene,
        toScene: lastScene,
        type: 'end'
    });

    return transitions;
}

/**
 * Analyzes scene distribution and statistics
 */
export function analyzeScenes(scenes: SceneAnnotation[]): {
    totalScenes: number;
    totalDuration: number;
    averageDuration: number;
    sceneTypes: Array<{ type: string; count: number; totalDuration: number }>;
    confidence: { min: number; max: number; average: number };
} {
    if (scenes.length === 0) {
        return {
            totalScenes: 0,
            totalDuration: 0,
            averageDuration: 0,
            sceneTypes: [],
            confidence: { min: 0, max: 0, average: 0 }
        };
    }

    const totalDuration = scenes.reduce((sum, scene) => sum + scene.duration, 0);
    const averageDuration = totalDuration / scenes.length;

    // Analyze scene types
    const typeMap = new Map<string, { count: number; totalDuration: number }>();

    for (const scene of scenes) {
        const type = scene.scene_type || 'unknown';
        if (!typeMap.has(type)) {
            typeMap.set(type, { count: 0, totalDuration: 0 });
        }
        const typeData = typeMap.get(type)!;
        typeData.count++;
        typeData.totalDuration += scene.duration;
    }

    const sceneTypes = Array.from(typeMap.entries()).map(([type, data]) => ({
        type,
        count: data.count,
        totalDuration: data.totalDuration
    }));

    // Analyze confidence scores
    const scores = scenes.map(scene => scene.score);
    const confidence = {
        min: Math.min(...scores),
        max: Math.max(...scores),
        average: scores.reduce((sum, score) => sum + score, 0) / scores.length
    };

    return {
        totalScenes: scenes.length,
        totalDuration,
        averageDuration,
        sceneTypes,
        confidence
    };
}

/**
 * Finds scene boundaries (potential cut points)
 */
export function findSceneBoundaries(scenes: SceneAnnotation[]): Array<{
    timestamp: number;
    type: 'cut' | 'fade' | 'transition';
    confidence: number;
    fromScene?: string;
    toScene?: string;
}> {
    const boundaries: Array<{
        timestamp: number;
        type: 'cut' | 'fade' | 'transition';
        confidence: number;
        fromScene?: string;
        toScene?: string;
    }> = [];

    const sortedScenes = [...scenes].sort((a, b) => a.start_time - b.start_time);

    for (let i = 0; i < sortedScenes.length - 1; i++) {
        const currentScene = sortedScenes[i];
        const nextScene = sortedScenes[i + 1];

        // Determine boundary type based on timing
        const gap = nextScene.start_time - currentScene.end_time;
        let type: 'cut' | 'fade' | 'transition';

        if (gap <= 0.1) {
            type = 'cut'; // Sharp transition
        } else if (gap <= 0.5) {
            type = 'fade'; // Short fade
        } else {
            type = 'transition'; // Longer transition
        }

        boundaries.push({
            timestamp: currentScene.end_time,
            type,
            confidence: Math.min(currentScene.score, nextScene.score),
            fromScene: currentScene.scene_type,
            toScene: nextScene.scene_type
        });
    }

    return boundaries;
}

/**
 * Validates scene detection file format without fully parsing
 */
export async function isValidSceneDetection(file: File): Promise<boolean> {
    try {
        const sample = await file.slice(0, 2000).text();
        const data = JSON.parse(sample);

        // Check if it looks like scene detection data
        if (Array.isArray(data)) {
            return data.length === 0 || (
                data[0] &&
                typeof data[0] === 'object' &&
                ('start_time' in data[0] || 'startTime' in data[0]) &&
                ('end_time' in data[0] || 'endTime' in data[0])
            );
        }

        return (
            (data.results && Array.isArray(data.results)) ||
            (data.scenes && Array.isArray(data.scenes))
        );
    } catch {
        return false;
    }
}

/**
 * Merges adjacent scenes of the same type
 */
export function mergeAdjacentScenes(scenes: SceneAnnotation[], threshold = 0.1): SceneAnnotation[] {
    if (scenes.length === 0) return scenes;

    const sortedScenes = [...scenes].sort((a, b) => a.start_time - b.start_time);
    const merged: SceneAnnotation[] = [];

    for (const scene of sortedScenes) {
        const lastMerged = merged[merged.length - 1];

        if (
            lastMerged &&
            lastMerged.scene_type === scene.scene_type &&
            scene.start_time <= lastMerged.end_time + threshold
        ) {
            // Merge scenes
            lastMerged.end_time = Math.max(lastMerged.end_time, scene.end_time);
            lastMerged.duration = lastMerged.end_time - lastMerged.start_time;
            lastMerged.frame_end = Math.max(lastMerged.frame_end, scene.frame_end);
            lastMerged.score = Math.max(lastMerged.score, scene.score);

            // Merge bbox (take larger area)
            if (scene.bbox[2] * scene.bbox[3] > lastMerged.bbox[2] * lastMerged.bbox[3]) {
                lastMerged.bbox = scene.bbox;
            }

            // Merge all_scores
            Object.assign(lastMerged.all_scores, scene.all_scores);
        } else {
            merged.push({ ...scene });
        }
    }

    return merged;
}
