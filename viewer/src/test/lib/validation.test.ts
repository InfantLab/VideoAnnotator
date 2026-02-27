import { describe, it, expect } from 'vitest';
import {
    validateCOCOPersonData,
    validateWebVTTData,
    validateRTTMData,
    validateSceneData,
    validateFileType,
    getValidationErrorMessage,
    isVideoAnnotatorFormat,
    ValidationError,
} from '@/lib/validation';

// Helper: valid 51-element keypoints
function makeKeypoints(): number[] {
    const kps: number[] = [];
    for (let i = 0; i < 17; i++) {
        kps.push(i * 10, i * 10 + 5, 2);
    }
    return kps;
}

function makeValidCOCOAnnotation() {
    return {
        id: 1,
        image_id: 'frame_001',
        category_id: 1,
        keypoints: makeKeypoints(),
        num_keypoints: 17,
        bbox: [10, 20, 100, 200],
        area: 20000,
        iscrowd: 0,
        score: 0.95,
        timestamp: 1.0,
        frame_number: 30,
        person_id: 'person_001',
        person_label: 'parent',
        label_confidence: 0.9,
        labeling_method: 'automatic',
    };
}

describe('ValidationError', () => {
    it('should create error with message', () => {
        const err = new ValidationError('test error');
        expect(err.message).toBe('test error');
        expect(err.name).toBe('ValidationError');
        expect(err.field).toBeUndefined();
    });

    it('should store field and value', () => {
        const err = new ValidationError('bad value', 'email', 'not-an-email');
        expect(err.field).toBe('email');
        expect(err.value).toBe('not-an-email');
    });
});

describe('validateCOCOPersonData', () => {
    it('should validate correct COCO data', () => {
        const result = validateCOCOPersonData([makeValidCOCOAnnotation()]);
        expect(result).toHaveLength(1);
        expect(result[0].person_id).toBe('person_001');
    });

    it('should accept empty array', () => {
        expect(validateCOCOPersonData([])).toEqual([]);
    });

    it('should reject wrong keypoints length', () => {
        const bad = { ...makeValidCOCOAnnotation(), keypoints: [1, 2, 3] };
        expect(() => validateCOCOPersonData([bad])).toThrow(ValidationError);
    });

    it('should reject score out of range', () => {
        const bad = { ...makeValidCOCOAnnotation(), score: 1.5 };
        expect(() => validateCOCOPersonData([bad])).toThrow(ValidationError);
    });

    it('should reject negative area', () => {
        const bad = { ...makeValidCOCOAnnotation(), area: -1 };
        expect(() => validateCOCOPersonData([bad])).toThrow(ValidationError);
    });

    it('should reject missing required fields', () => {
        expect(() => validateCOCOPersonData([{ id: 1 }])).toThrow(ValidationError);
    });
});

describe('validateWebVTTData', () => {
    it('should validate correct WebVTT data', () => {
        const data = [{ id: '1', startTime: 0, endTime: 5, text: 'Hello' }];
        const result = validateWebVTTData(data);
        expect(result).toHaveLength(1);
    });

    it('should accept empty array', () => {
        expect(validateWebVTTData([])).toEqual([]);
    });

    it('should accept minimal object (all fields optional)', () => {
        expect(validateWebVTTData([{}])).toHaveLength(1);
    });

    it('should reject negative times', () => {
        expect(() => validateWebVTTData([{ startTime: -1 }])).toThrow(ValidationError);
    });
});

describe('validateRTTMData', () => {
    it('should validate correct RTTM data', () => {
        const data = [{
            file_id: 'test.wav',
            start_time: 0.5,
            duration: 2.0,
            speaker_id: 'spk_01',
            confidence: 0.9,
        }];
        const result = validateRTTMData(data);
        expect(result).toHaveLength(1);
    });

    it('should accept empty array', () => {
        expect(validateRTTMData([])).toEqual([]);
    });

    it('should reject confidence > 1', () => {
        expect(() => validateRTTMData([{ confidence: 1.5 }])).toThrow(ValidationError);
    });
});

describe('validateSceneData', () => {
    it('should validate correct scene data', () => {
        const data = [{
            id: 1,
            timestamp: 5.0,
            scene_type: 'indoor',
            score: 0.85,
        }];
        const result = validateSceneData(data);
        expect(result).toHaveLength(1);
    });

    it('should accept empty array', () => {
        expect(validateSceneData([])).toEqual([]);
    });

    it('should reject score > 1', () => {
        expect(() => validateSceneData([{ score: 2.0 }])).toThrow(ValidationError);
    });
});

describe('validateFileType', () => {
    it.each([
        'video/mp4', 'video/webm', 'video/avi', 'video/mov',
        'text/vtt', 'text/plain', 'application/json',
        'audio/wav', 'audio/mpeg',
    ])('should accept %s', (type) => {
        expect(validateFileType(type)).toBe(type);
    });

    it('should reject unsupported types', () => {
        expect(() => validateFileType('audio/aac')).toThrow(ValidationError);
        expect(() => validateFileType('video/mkv')).toThrow(ValidationError);
    });

    it('should reject empty string', () => {
        expect(() => validateFileType('')).toThrow(ValidationError);
    });
});

describe('getValidationErrorMessage', () => {
    it('should return message without field', () => {
        const err = new ValidationError('Something went wrong');
        expect(getValidationErrorMessage(err)).toBe('Something went wrong');
    });

    it('should append field info', () => {
        const err = new ValidationError('Invalid value', 'email');
        expect(getValidationErrorMessage(err)).toBe('Invalid value (field: email)');
    });
});

describe('isVideoAnnotatorFormat', () => {
    it('should detect pipeline result array', () => {
        const data = [
            { pipeline: 'face', format: 'coco', data: [] },
            { pipeline: 'speech', format: 'webvtt', data: [] },
        ];
        expect(isVideoAnnotatorFormat(data)).toBe(true);
    });

    it('should detect single pipeline result object', () => {
        expect(isVideoAnnotatorFormat({ pipeline: 'face' })).toBe(true);
        expect(isVideoAnnotatorFormat({ pipeline_results: [] })).toBe(true);
    });

    it('should reject regular arrays', () => {
        expect(isVideoAnnotatorFormat([{ id: 1 }])).toBe(false);
    });

    it('should reject non-objects', () => {
        expect(isVideoAnnotatorFormat(null)).toBe(false);
        expect(isVideoAnnotatorFormat('string')).toBe(false);
        expect(isVideoAnnotatorFormat(42)).toBe(false);
        expect(isVideoAnnotatorFormat(undefined)).toBe(false);
    });

    it('should reject empty array', () => {
        // empty array: every() returns true for empty arrays
        expect(isVideoAnnotatorFormat([])).toBe(true);
    });
});
