import { describe, it, expect } from 'vitest';
import {
    parseCOCOPersonData,
    groupByTimestamp,
    getPersonsAtTime,
    getVisibleKeypoints,
    getDrawableConnections,
    getBoundingBoxWithPadding,
    isValidCOCOPersonData,
    getTrackStatistics,
    cocoToLegacyPose,
} from '@/lib/parsers/coco';
import type { COCOPersonAnnotation } from '@/types/annotations';

const createMockFile = (content: string, name: string, type = 'application/json') => {
    return new File([content], name, { type });
};

// Helper: generate a valid 51-element keypoints array (17 keypoints * 3)
// All visible by default
function makeKeypoints(visibility = 2): number[] {
    const kps: number[] = [];
    for (let i = 0; i < 17; i++) {
        kps.push(i * 10, i * 10 + 5, visibility); // x, y, visibility
    }
    return kps;
}

function makeAnnotation(overrides: Partial<COCOPersonAnnotation> = {}): COCOPersonAnnotation {
    return {
        id: 1,
        image_id: 'frame_001',
        category_id: 1,
        keypoints: makeKeypoints(),
        num_keypoints: 17,
        bbox: [10, 20, 100, 200] as [number, number, number, number],
        area: 20000,
        iscrowd: 0,
        score: 0.95,
        track_id: 1,
        timestamp: 1.0,
        frame_number: 30,
        person_id: 'person_001',
        person_label: 'parent',
        label_confidence: 0.9,
        labeling_method: 'automatic_size_based',
        ...overrides,
    };
}

describe('COCO Parser', () => {
    describe('parseCOCOPersonData', () => {
        it('should parse direct array of annotations', async () => {
            const data = [makeAnnotation()];
            const file = createMockFile(JSON.stringify(data), 'test.json');
            const result = await parseCOCOPersonData(file);
            expect(result).toHaveLength(1);
            expect(result[0].person_id).toBe('person_001');
        });

        it('should parse COCO format with annotations array', async () => {
            const data = { annotations: [makeAnnotation()] };
            const file = createMockFile(JSON.stringify(data), 'test.json');
            const result = await parseCOCOPersonData(file);
            expect(result).toHaveLength(1);
        });

        it('should parse VideoAnnotator format with results array', async () => {
            const data = { results: [makeAnnotation()] };
            const file = createMockFile(JSON.stringify(data), 'test.json');
            const result = await parseCOCOPersonData(file);
            expect(result).toHaveLength(1);
        });

        it('should sort results by timestamp', async () => {
            const data = [
                makeAnnotation({ id: 2, timestamp: 3.0 }),
                makeAnnotation({ id: 1, timestamp: 1.0 }),
                makeAnnotation({ id: 3, timestamp: 2.0 }),
            ];
            const file = createMockFile(JSON.stringify(data), 'test.json');
            const result = await parseCOCOPersonData(file);
            expect(result.map(a => a.timestamp)).toEqual([1.0, 2.0, 3.0]);
        });

        it('should return empty array for empty annotations', async () => {
            const file = createMockFile('[]', 'empty.json');
            const result = await parseCOCOPersonData(file);
            expect(result).toEqual([]);
        });

        it('should throw on invalid JSON', async () => {
            const file = createMockFile('not json', 'bad.json');
            await expect(parseCOCOPersonData(file)).rejects.toThrow('Invalid JSON');
        });

        it('should throw on invalid COCO structure', async () => {
            const file = createMockFile(JSON.stringify({ foo: 'bar' }), 'bad.json');
            await expect(parseCOCOPersonData(file)).rejects.toThrow('Invalid COCO format');
        });

        it('should throw on invalid keypoints length', async () => {
            const badAnnotation = makeAnnotation({ keypoints: [1, 2, 3] }); // only 3 values
            const file = createMockFile(JSON.stringify([badAnnotation]), 'bad.json');
            await expect(parseCOCOPersonData(file)).rejects.toThrow();
        });
    });

    describe('groupByTimestamp', () => {
        it('should group annotations by rounded timestamp', () => {
            const annotations = [
                makeAnnotation({ id: 1, timestamp: 1.0 }),
                makeAnnotation({ id: 2, timestamp: 1.0 }),
                makeAnnotation({ id: 3, timestamp: 2.0 }),
            ];
            const groups = groupByTimestamp(annotations);
            expect(groups.size).toBe(2);
            expect(groups.get(1.0)).toHaveLength(2);
            expect(groups.get(2.0)).toHaveLength(1);
        });

        it('should handle empty annotations', () => {
            expect(groupByTimestamp([])).toEqual(new Map());
        });
    });

    describe('getPersonsAtTime', () => {
        it('should return annotations within tolerance', () => {
            const annotations = [
                makeAnnotation({ id: 1, timestamp: 1.0 }),
                makeAnnotation({ id: 2, timestamp: 1.05 }),
                makeAnnotation({ id: 3, timestamp: 2.0 }),
            ];
            const result = getPersonsAtTime(annotations, 1.0);
            expect(result).toHaveLength(2);
        });

        it('should return empty for no matches', () => {
            const annotations = [makeAnnotation({ timestamp: 5.0 })];
            expect(getPersonsAtTime(annotations, 1.0)).toEqual([]);
        });

        it('should use custom tolerance', () => {
            const annotations = [
                makeAnnotation({ id: 1, timestamp: 1.0 }),
                makeAnnotation({ id: 2, timestamp: 1.5 }),
            ];
            expect(getPersonsAtTime(annotations, 1.0, 0.01)).toHaveLength(1);
            expect(getPersonsAtTime(annotations, 1.0, 1.0)).toHaveLength(2);
        });
    });

    describe('getVisibleKeypoints', () => {
        it('should return only visible keypoints', () => {
            const annotation = makeAnnotation({ keypoints: makeKeypoints(2) }); // all visible
            const visible = getVisibleKeypoints(annotation);
            expect(visible).toHaveLength(17);
            expect(visible[0].name).toBe('nose');
        });

        it('should filter out non-visible keypoints', () => {
            const kps = makeKeypoints(0); // all not labeled
            const annotation = makeAnnotation({ keypoints: kps });
            expect(getVisibleKeypoints(annotation)).toHaveLength(0);
        });
    });

    describe('getDrawableConnections', () => {
        it('should return connections between visible keypoints', () => {
            const annotation = makeAnnotation({ keypoints: makeKeypoints(2) });
            const connections = getDrawableConnections(annotation);
            expect(connections.length).toBeGreaterThan(0);
            expect(connections[0]).toHaveProperty('from');
            expect(connections[0]).toHaveProperty('to');
            expect(connections[0]).toHaveProperty('confidence');
        });

        it('should return no connections when keypoints are not visible', () => {
            const annotation = makeAnnotation({ keypoints: makeKeypoints(0) });
            expect(getDrawableConnections(annotation)).toHaveLength(0);
        });
    });

    describe('getBoundingBoxWithPadding', () => {
        it('should add default padding of 10', () => {
            const annotation = makeAnnotation({ bbox: [10, 20, 100, 200] });
            const result = getBoundingBoxWithPadding(annotation);
            expect(result).toEqual({ x: 0, y: 10, width: 120, height: 220 });
        });

        it('should support custom padding', () => {
            const annotation = makeAnnotation({ bbox: [50, 50, 100, 100] });
            const result = getBoundingBoxWithPadding(annotation, 5);
            expect(result).toEqual({ x: 45, y: 45, width: 110, height: 110 });
        });
    });

    describe('cocoToLegacyPose', () => {
        it('should convert annotation to legacy format', () => {
            const annotation = makeAnnotation({ track_id: 5, score: 0.85 });
            const legacy = cocoToLegacyPose(annotation);
            expect(legacy.id).toBe(5);
            expect(legacy.confidence).toBe(0.85);
            expect(legacy.keypoints).toHaveLength(17);
            expect(legacy.keypoints[0]).toHaveProperty('x');
            expect(legacy.keypoints[0]).toHaveProperty('y');
            expect(legacy.keypoints[0]).toHaveProperty('confidence');
        });

        it('should fall back to id when track_id is missing', () => {
            const annotation = makeAnnotation({ id: 42, track_id: undefined });
            const legacy = cocoToLegacyPose(annotation);
            expect(legacy.id).toBe(42);
        });
    });

    describe('isValidCOCOPersonData', () => {
        it('should validate direct array with keypoints and bbox', async () => {
            const data = [{ keypoints: makeKeypoints(), bbox: [0, 0, 100, 100] }];
            const file = createMockFile(JSON.stringify(data), 'test.json');
            expect(await isValidCOCOPersonData(file)).toBe(true);
        });

        it('should validate COCO format with annotations', async () => {
            const data = { annotations: [{}] };
            const file = createMockFile(JSON.stringify(data), 'test.json');
            expect(await isValidCOCOPersonData(file)).toBe(true);
        });

        it('should reject invalid data', async () => {
            const file = createMockFile('not json at all', 'bad.json');
            expect(await isValidCOCOPersonData(file)).toBe(false);
        });

        it('should accept empty array', async () => {
            const file = createMockFile('[]', 'empty.json');
            expect(await isValidCOCOPersonData(file)).toBe(true);
        });
    });

    describe('getTrackStatistics', () => {
        it('should compute statistics per track', () => {
            const annotations = [
                makeAnnotation({ track_id: 1, timestamp: 0, score: 0.8, num_keypoints: 15 }),
                makeAnnotation({ track_id: 1, timestamp: 1, score: 0.9, num_keypoints: 17 }),
                makeAnnotation({ track_id: 2, timestamp: 0, score: 0.7, num_keypoints: 10 }),
            ];
            const stats = getTrackStatistics(annotations);
            expect(stats).toHaveLength(2);

            const track1 = stats.find(s => s.trackId === 1)!;
            expect(track1.frameCount).toBe(2);
            expect(track1.duration).toBe(1);
            expect(track1.avgConfidence).toBeCloseTo(0.85);
            expect(track1.avgKeypoints).toBe(16);

            const track2 = stats.find(s => s.trackId === 2)!;
            expect(track2.frameCount).toBe(1);
            expect(track2.duration).toBe(0);
        });

        it('should handle empty annotations', () => {
            expect(getTrackStatistics([])).toEqual([]);
        });
    });
});
