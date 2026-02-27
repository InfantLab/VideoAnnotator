import { describe, it, expect } from 'vitest';
import {
    parseFaceAnalysis,
    parseFaceAnalysisWithMetadata,
    getFacesAtTime,
    getDominantEmotion,
    getFaceEmotions,
    validateFaceAnalysisFile,
    getFaceAnalysisTimeline,
} from '@/lib/parsers/face';
import type { LAIONFaceAnnotation } from '@/types/annotations';

const createMockFile = (content: string, name: string, type = 'application/json') => {
    return new File([content], name, { type });
};

function makeFaceAnnotation(overrides: Partial<LAIONFaceAnnotation> = {}): LAIONFaceAnnotation {
    return {
        id: 1,
        image_id: 'frame_001',
        category_id: 1,
        bbox: [10, 20, 50, 60] as [number, number, number, number],
        area: 3000,
        iscrowd: 0,
        score: 0.92,
        face_id: 1,
        timestamp: 1.0,
        frame_number: 30,
        backend: 'laion',
        person_id: 'person_001',
        person_label: 'parent',
        person_label_confidence: 0.85,
        person_labeling_method: 'automatic',
        attributes: {
            emotions: {
                happiness: { score: 0.8, rank: 1, raw_score: 0.75 },
                neutral: { score: 0.15, rank: 2, raw_score: 0.12 },
                sadness: { score: 0.05, rank: 3, raw_score: 0.04 },
            },
            model_info: {
                model_size: 'small',
                embedding_dim: 1152,
            },
        },
        ...overrides,
    } as LAIONFaceAnnotation;
}

describe('Face Parser', () => {
    describe('parseFaceAnalysis', () => {
        it('should parse direct array of face annotations', async () => {
            const data = [makeFaceAnnotation()];
            const file = createMockFile(JSON.stringify(data), 'faces.json');
            const result = await parseFaceAnalysis(file);
            expect(result).toHaveLength(1);
            expect(result[0].face_id).toBe(1);
        });

        it('should parse COCO format with annotations array', async () => {
            const data = { annotations: [makeFaceAnnotation()] };
            const file = createMockFile(JSON.stringify(data), 'faces.json');
            const result = await parseFaceAnalysis(file);
            expect(result).toHaveLength(1);
        });

        it('should parse results wrapper format', async () => {
            const data = { results: [makeFaceAnnotation()] };
            const file = createMockFile(JSON.stringify(data), 'faces.json');
            const result = await parseFaceAnalysis(file);
            expect(result).toHaveLength(1);
        });

        it('should throw on invalid format', async () => {
            const file = createMockFile(JSON.stringify({ foo: 'bar' }), 'bad.json');
            await expect(parseFaceAnalysis(file)).rejects.toThrow('Invalid face analysis format');
        });

        it('should filter out invalid face entries', async () => {
            const data = [
                makeFaceAnnotation(),
                { invalid: true }, // missing face_id, bbox, etc.
            ];
            const file = createMockFile(JSON.stringify(data), 'faces.json');
            const result = await parseFaceAnalysis(file);
            expect(result).toHaveLength(1);
        });

        it('should throw when all faces are invalid', async () => {
            const data = [{ invalid: true }, { also_invalid: true }];
            const file = createMockFile(JSON.stringify(data), 'faces.json');
            await expect(parseFaceAnalysis(file)).rejects.toThrow('No valid face annotations');
        });

        it('should return empty for empty array', async () => {
            const file = createMockFile('[]', 'empty.json');
            const result = await parseFaceAnalysis(file);
            expect(result).toEqual([]);
        });
    });

    describe('parseFaceAnalysisWithMetadata', () => {
        it('should compute metadata from face annotations', async () => {
            const data = [
                makeFaceAnnotation({ face_id: 1, score: 0.9, frame_number: 1 }),
                makeFaceAnnotation({ face_id: 2, score: 0.8, frame_number: 2 }),
            ];
            const file = createMockFile(JSON.stringify(data), 'faces.json');
            const result = await parseFaceAnalysisWithMetadata(file);

            expect(result.faces).toHaveLength(2);
            expect(result.metadata.total_faces).toBe(2);
            expect(result.metadata.frames_with_faces).toBe(2);
            expect(result.metadata.emotions_detected).toContain('happiness');
            expect(result.metadata.confidence_stats.min).toBe(0.8);
            expect(result.metadata.confidence_stats.max).toBe(0.9);
            expect(result.metadata.confidence_stats.average).toBeCloseTo(0.85);
            expect(result.metadata.backend).toBe('laion');
            expect(result.metadata.model_info).toEqual({ model_size: 'small', embedding_dim: 1152 });
        });

        it('should return empty metadata for empty input', async () => {
            const file = createMockFile('[]', 'empty.json');
            const result = await parseFaceAnalysisWithMetadata(file);
            expect(result.faces).toEqual([]);
            expect(result.metadata.total_faces).toBe(0);
            expect(result.metadata.backend).toBe('unknown');
        });
    });

    describe('getFacesAtTime', () => {
        const faces = [
            makeFaceAnnotation({ face_id: 1, timestamp: 1.0 }),
            makeFaceAnnotation({ face_id: 2, timestamp: 1.05 }),
            makeFaceAnnotation({ face_id: 3, timestamp: 5.0 }),
        ];

        it('should find faces within default tolerance', () => {
            const result = getFacesAtTime(faces, 1.0);
            expect(result).toHaveLength(2);
        });

        it('should return empty when no faces match', () => {
            expect(getFacesAtTime(faces, 10.0)).toEqual([]);
        });

        it('should support custom tolerance', () => {
            expect(getFacesAtTime(faces, 1.0, 0.01)).toHaveLength(1);
        });
    });

    describe('getDominantEmotion', () => {
        it('should return emotion with rank 1', () => {
            const face = makeFaceAnnotation();
            const result = getDominantEmotion(face);
            expect(result).toEqual({ emotion: 'happiness', score: 0.8, rank: 1 });
        });

        it('should fall back to highest score when no rank 1', () => {
            const face = makeFaceAnnotation({
                attributes: {
                    emotions: {
                        neutral: { score: 0.3, rank: 2, raw_score: 0.2 },
                        sadness: { score: 0.7, rank: 3, raw_score: 0.6 },
                    },
                    model_info: { model_size: 'small', embedding_dim: 1152 },
                },
            } as LAIONFaceAnnotation['attributes'] extends infer T ? { attributes: T } : never);
            const result = getDominantEmotion(face);
            expect(result?.emotion).toBe('sadness');
            expect(result?.score).toBe(0.7);
        });

        it('should return null for face without emotions', () => {
            const face = makeFaceAnnotation();
            face.attributes = {} as typeof face.attributes;
            expect(getDominantEmotion(face)).toBeNull();
        });

        it('should return null for empty emotions object', () => {
            const face = makeFaceAnnotation();
            face.attributes = { emotions: {}, model_info: { model_size: 'small', embedding_dim: 1152 } } as typeof face.attributes;
            expect(getDominantEmotion(face)).toBeNull();
        });
    });

    describe('getFaceEmotions', () => {
        it('should return sorted emotions by rank', () => {
            const face = makeFaceAnnotation();
            const emotions = getFaceEmotions(face);
            expect(emotions).toHaveLength(3);
            expect(emotions[0].emotion).toBe('happiness');
            expect(emotions[0].rank).toBe(1);
            expect(emotions[1].rank).toBe(2);
            expect(emotions[2].rank).toBe(3);
        });

        it('should return empty for face without emotions', () => {
            const face = makeFaceAnnotation();
            face.attributes = {} as typeof face.attributes;
            expect(getFaceEmotions(face)).toEqual([]);
        });
    });

    describe('validateFaceAnalysisFile', () => {
        it('should validate correct face analysis file', async () => {
            const data = [makeFaceAnnotation()];
            const file = createMockFile(JSON.stringify(data), 'faces.json');
            const result = await validateFaceAnalysisFile(file);
            expect(result.isValid).toBe(true);
        });

        it('should report missing face_id', async () => {
            const data = [{ bbox: [0, 0, 10, 10], timestamp: 1.0 }];
            const file = createMockFile(JSON.stringify(data), 'faces.json');
            const result = await validateFaceAnalysisFile(file);
            expect(result.isValid).toBe(false);
            expect(result.error).toContain('face_id');
        });

        it('should report invalid bbox', async () => {
            const data = [{ face_id: 1, bbox: [0, 0], timestamp: 1.0 }];
            const file = createMockFile(JSON.stringify(data), 'faces.json');
            const result = await validateFaceAnalysisFile(file);
            expect(result.isValid).toBe(false);
            expect(result.error).toContain('bbox');
        });

        it('should report missing timestamp', async () => {
            const data = [{ face_id: 1, bbox: [0, 0, 10, 10] }];
            const file = createMockFile(JSON.stringify(data), 'faces.json');
            const result = await validateFaceAnalysisFile(file);
            expect(result.isValid).toBe(false);
            expect(result.error).toContain('timestamp');
        });

        it('should warn on missing emotion data', async () => {
            const data = [{ face_id: 1, bbox: [0, 0, 10, 10], timestamp: 1.0 }];
            const file = createMockFile(JSON.stringify(data), 'faces.json');
            const result = await validateFaceAnalysisFile(file);
            expect(result.isValid).toBe(true);
            expect(result.warnings).toContain('Face has no emotion analysis data');
        });

        it('should handle empty array with warning', async () => {
            const file = createMockFile('[]', 'empty.json');
            const result = await validateFaceAnalysisFile(file);
            expect(result.isValid).toBe(true);
            expect(result.warnings).toContain('File contains no face annotations');
        });

        it('should reject invalid JSON', async () => {
            const file = createMockFile('not json', 'bad.json');
            const result = await validateFaceAnalysisFile(file);
            expect(result.isValid).toBe(false);
            expect(result.error).toContain('Failed to parse');
        });

        it('should reject non-array non-object formats', async () => {
            const file = createMockFile('"string"', 'bad.json');
            const result = await validateFaceAnalysisFile(file);
            expect(result.isValid).toBe(false);
        });
    });

    describe('getFaceAnalysisTimeline', () => {
        it('should group faces by timestamp and compute stats', () => {
            const faces = [
                makeFaceAnnotation({ face_id: 1, timestamp: 1.0, score: 0.9 }),
                makeFaceAnnotation({ face_id: 2, timestamp: 1.0, score: 0.8 }),
                makeFaceAnnotation({ face_id: 3, timestamp: 2.0, score: 0.7 }),
            ];
            const timeline = getFaceAnalysisTimeline(faces);
            expect(timeline).toHaveLength(2);
            expect(timeline[0].timestamp).toBe(1.0);
            expect(timeline[0].face_count).toBe(2);
            expect(timeline[0].average_confidence).toBeCloseTo(0.85);
            expect(timeline[1].timestamp).toBe(2.0);
            expect(timeline[1].face_count).toBe(1);
        });

        it('should return empty for no faces', () => {
            expect(getFaceAnalysisTimeline([])).toEqual([]);
        });

        it('should sort timeline by timestamp', () => {
            const faces = [
                makeFaceAnnotation({ face_id: 1, timestamp: 3.0 }),
                makeFaceAnnotation({ face_id: 2, timestamp: 1.0 }),
            ];
            const timeline = getFaceAnalysisTimeline(faces);
            expect(timeline[0].timestamp).toBe(1.0);
            expect(timeline[1].timestamp).toBe(3.0);
        });
    });
});
