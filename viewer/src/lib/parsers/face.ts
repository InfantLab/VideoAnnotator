// Face analysis parser for LAION emotion recognition and facial landmarks
// Parses VideoAnnotator v1.1.1 LAION face analysis output
// Reference: https://github.com/InfantLab/VideoAnnotator

import type { LAIONFaceAnnotation } from '@/types/annotations';

/**
 * Face analysis parsing result with metadata
 */
export interface FaceAnalysisResult {
    faces: LAIONFaceAnnotation[];
    metadata: {
        total_faces: number;
        frames_with_faces: number;
        emotions_detected: string[];
        confidence_stats: {
            min: number;
            max: number;
            average: number;
        };
        backend: string;
        model_info?: {
            model_size: string;
            embedding_dim: number;
        };
    };
}

/**
 * Parses LAION face analysis file
 */
export async function parseFaceAnalysis(file: File): Promise<LAIONFaceAnnotation[]> {
    const text = await file.text();
    const data = JSON.parse(text);

    let faces: LAIONFaceAnnotation[] = [];

    // Handle different formats: direct array, COCO-style, or results wrapper
    if (Array.isArray(data)) {
        faces = data;
    } else if (data.annotations && Array.isArray(data.annotations)) {
        faces = data.annotations;
    } else if (data.results && Array.isArray(data.results)) {
        faces = data.results;
    } else {
        throw new Error('Invalid face analysis format: expected array of face annotations');
    }

    // Validate and filter valid face annotations
    const validFaces = faces.filter(face => {
        return face && 
               typeof face.face_id === 'number' &&
               Array.isArray(face.bbox) &&
               face.bbox.length === 4 &&
               face.timestamp !== undefined &&
               face.attributes &&
               face.attributes.emotions;
    });

    if (validFaces.length === 0 && faces.length > 0) {
        throw new Error('No valid face annotations found in file');
    }

    return validFaces;
}

/**
 * Parses face analysis file with detailed metadata analysis
 */
export async function parseFaceAnalysisWithMetadata(file: File): Promise<FaceAnalysisResult> {
    const faces = await parseFaceAnalysis(file);

    if (faces.length === 0) {
        return {
            faces: [],
            metadata: {
                total_faces: 0,
                frames_with_faces: 0,
                emotions_detected: [],
                confidence_stats: { min: 0, max: 0, average: 0 },
                backend: 'unknown'
            }
        };
    }

    // Collect emotions from all faces
    const allEmotions = new Set<string>();
    const confidenceScores: number[] = [];
    const frameNumbers = new Set<number>();
    let backend = 'unknown';
    let modelInfo: { model_size: string; embedding_dim: number } | undefined;

    for (const face of faces) {
        // Collect emotions
        if (face.attributes?.emotions) {
            Object.keys(face.attributes.emotions).forEach(emotion => {
                allEmotions.add(emotion);
            });
        }

        // Collect confidence scores
        if (typeof face.score === 'number') {
            confidenceScores.push(face.score);
        }

        // Track frames with faces
        if (typeof face.frame_number === 'number') {
            frameNumbers.add(face.frame_number);
        }

        // Extract backend and model info
        if (face.backend) {
            backend = face.backend;
        }

        if (face.attributes?.model_info && !modelInfo) {
            modelInfo = face.attributes.model_info;
        }
    }

    // Calculate confidence statistics
    const confidenceStats = {
        min: confidenceScores.length > 0 ? Math.min(...confidenceScores) : 0,
        max: confidenceScores.length > 0 ? Math.max(...confidenceScores) : 0,
        average: confidenceScores.length > 0 ? 
            confidenceScores.reduce((sum, score) => sum + score, 0) / confidenceScores.length : 0
    };

    return {
        faces,
        metadata: {
            total_faces: faces.length,
            frames_with_faces: frameNumbers.size,
            emotions_detected: Array.from(allEmotions).sort(),
            confidence_stats: confidenceStats,
            backend,
            model_info: modelInfo
        }
    };
}

/**
 * Gets faces at a specific timestamp with tolerance
 */
export function getFacesAtTime(
    faces: LAIONFaceAnnotation[], 
    currentTime: number, 
    tolerance: number = 0.1
): LAIONFaceAnnotation[] {
    return faces.filter(face => 
        Math.abs(face.timestamp - currentTime) <= tolerance
    );
}

/**
 * Gets the dominant emotion for a face
 */
export function getDominantEmotion(face: LAIONFaceAnnotation): {
    emotion: string;
    score: number;
    rank: number;
} | null {
    if (!face.attributes?.emotions) return null;

    const emotions = Object.entries(face.attributes.emotions);
    if (emotions.length === 0) return null;

    // Find emotion with rank 1 (highest)
    const dominantEmotion = emotions.find(([_, data]) => data.rank === 1);
    
    if (dominantEmotion) {
        const [emotion, data] = dominantEmotion;
        return {
            emotion,
            score: data.score,
            rank: data.rank
        };
    }

    // Fallback: highest score
    const sortedEmotions = emotions.sort((a, b) => b[1].score - a[1].score);
    const [emotion, data] = sortedEmotions[0];
    
    return {
        emotion,
        score: data.score,
        rank: data.rank
    };
}

/**
 * Gets all emotions for a face sorted by rank
 */
export function getFaceEmotions(face: LAIONFaceAnnotation): Array<{
    emotion: string;
    score: number;
    rank: number;
    raw_score: number;
}> {
    if (!face.attributes?.emotions) return [];

    return Object.entries(face.attributes.emotions)
        .map(([emotion, data]) => ({
            emotion,
            score: data.score,
            rank: data.rank,
            raw_score: data.raw_score
        }))
        .sort((a, b) => a.rank - b.rank); // Sort by rank (1 = highest)
}

/**
 * Validates face analysis file format
 */
export async function validateFaceAnalysisFile(file: File): Promise<{
    isValid: boolean;
    error?: string;
    warnings?: string[];
}> {
    try {
        const sample = await file.slice(0, 5000).text();
        const data: unknown = JSON.parse(sample);

        let faces: unknown[] = [];

        if (Array.isArray(data)) {
            faces = data;
        } else if (data && typeof data === 'object') {
            const record = data as Record<string, unknown>;
            if (Array.isArray(record.annotations)) {
                faces = record.annotations;
            } else if (Array.isArray(record.results)) {
                faces = record.results;
            } else {
                return {
                    isValid: false,
                    error: 'Invalid format: expected array or object with annotations/results field'
                };
            }
        } else {
            return {
                isValid: false,
                error: 'Invalid format: expected array or object with annotations/results field'
            };
        }

        if (faces.length === 0) {
            return {
                isValid: true,
                warnings: ['File contains no face annotations']
            };
        }

        const firstFace = faces[0];
        const warnings: string[] = [];

        // Check required fields
        if (!('face_id' in firstFace)) {
            return { isValid: false, error: 'Missing required field: face_id' };
        }

        if (!('bbox' in firstFace) || !Array.isArray(firstFace.bbox) || firstFace.bbox.length !== 4) {
            return { isValid: false, error: 'Invalid or missing bbox field' };
        }

        if (!('timestamp' in firstFace)) {
            return { isValid: false, error: 'Missing required field: timestamp' };
        }

        if (!('attributes' in firstFace) || !firstFace.attributes.emotions) {
            warnings.push('Face has no emotion analysis data');
        }

        return {
            isValid: true,
            warnings: warnings.length > 0 ? warnings : undefined
        };

    } catch (error) {
        return {
            isValid: false,
            error: `Failed to parse file: ${error instanceof Error ? error.message : 'Unknown error'}`
        };
    }
}

/**
 * Gets face analysis timeline data for visualization
 */
export function getFaceAnalysisTimeline(faces: LAIONFaceAnnotation[]): Array<{
    timestamp: number;
    face_count: number;
    dominant_emotions: string[];
    average_confidence: number;
}> {
    if (faces.length === 0) return [];

    // Group faces by timestamp
    const facesByTime = new Map<number, LAIONFaceAnnotation[]>();
    
    for (const face of faces) {
        const time = face.timestamp;
        if (!facesByTime.has(time)) {
            facesByTime.set(time, []);
        }
        facesByTime.get(time)!.push(face);
    }

    // Create timeline entries
    const timeline = Array.from(facesByTime.entries()).map(([timestamp, facesAtTime]) => {
        const emotions = new Map<string, number>();
        let totalConfidence = 0;

        for (const face of facesAtTime) {
            const dominantEmotion = getDominantEmotion(face);
            if (dominantEmotion) {
                emotions.set(dominantEmotion.emotion, (emotions.get(dominantEmotion.emotion) || 0) + 1);
            }
            totalConfidence += face.score || 0;
        }

        const dominantEmotions = Array.from(emotions.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, 3)
            .map(([emotion]) => emotion);

        return {
            timestamp,
            face_count: facesAtTime.length,
            dominant_emotions: dominantEmotions,
            average_confidence: facesAtTime.length > 0 ? totalConfidence / facesAtTime.length : 0
        };
    });

    return timeline.sort((a, b) => a.timestamp - b.timestamp);
}