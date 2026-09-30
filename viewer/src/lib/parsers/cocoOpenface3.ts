/**
 * COCO-OpenFace3 Parser
 * Parses COCO format files that contain embedded OpenFace3 data from VideoAnnotator
 * Converts VideoAnnotator COCO+OpenFace3 format to StandardFaceAnnotation format
 */

import { StandardFaceAnnotation } from '../../types/annotations';

/**
 * COCO annotation with embedded OpenFace3 data (VideoAnnotator format)
 */
interface COCOOpenFace3Annotation {
  id: number;
  image_id: number;
  category_id: number;
  bbox: [number, number, number, number]; // [x, y, width, height]
  area: number;
  iscrowd: number;
  keypoints: number[]; // COCO keypoints format
  num_keypoints: number;
  openface3: {
    confidence: number;
    timestamp: number;
    landmarks_2d?: [number, number][];
    action_units?: {
      [key: string]: {
        intensity: number;
        presence: boolean;
      };
    };
    head_pose?: {
      pitch: number;
      yaw: number;
      roll: number;
      confidence: number;
    };
    gaze?: {
      direction_x: number;
      direction_y: number;
      direction_z: number;
      confidence: number;
    };
    emotion?: {
      dominant: string;
      probabilities: {
        [emotion: string]: number;
      };
      valence: number;
      arousal: number;
      confidence: number;
    };
  };
}

/**
 * COCO file structure with OpenFace3 data
 */
interface COCOOpenFace3Data {
  info: {
    description: string;
    version: string;
    year: number;
    contributor: string;
    date_created: string;
  };
  images: {
    id: number;
    width: number;
    height: number;
    file_name: string;
    timestamp: number;
  }[];
  annotations: COCOOpenFace3Annotation[];
}

export class COCOOpenFace3Parser {
  private static instance: COCOOpenFace3Parser;
  
  public static getInstance(): COCOOpenFace3Parser {
    if (!COCOOpenFace3Parser.instance) {
      COCOOpenFace3Parser.instance = new COCOOpenFace3Parser();
    }
    return COCOOpenFace3Parser.instance;
  }

  /**
   * Main parsing method - converts COCO+OpenFace3 JSON to StandardFaceAnnotation[]
   */
  public async parseCOCOOpenFace3Data(file: File): Promise<StandardFaceAnnotation[]> {
    try {
      const content = await file.text();
      const data = JSON.parse(content) as COCOOpenFace3Data;
      
      // Validate input structure
      if (!this.isValidCOCOOpenFace3Data(data)) {
        throw new Error('Invalid COCO+OpenFace3 data structure');
      }

      // Create image timestamp lookup
      const imageTimestamps = new Map<number, number>();
      data.images.forEach(img => {
        imageTimestamps.set(img.id, img.timestamp);
      });

      // Convert each annotation
      const standardAnnotations: StandardFaceAnnotation[] = data.annotations
        .filter(annotation => annotation.openface3) // Only process annotations with OpenFace3 data
        .map((annotation, index) => this.convertAnnotation(annotation, imageTimestamps, index));

      console.log(`✓ COCO-OpenFace3 Parser: Converted ${standardAnnotations.length} face annotations`);
      return standardAnnotations;

    } catch (error) {
      console.error('COCO-OpenFace3 parsing failed:', error);
      throw new Error(`Failed to parse COCO+OpenFace3 data: ${error}`);
    }
  }

  /**
   * Convert single COCO annotation with OpenFace3 data to StandardFaceAnnotation
   */
  private convertAnnotation(
    annotation: COCOOpenFace3Annotation, 
    imageTimestamps: Map<number, number>,
    index: number
  ): StandardFaceAnnotation {
    const timestamp = imageTimestamps.get(annotation.image_id) || 0;
    
    // Convert COCO bbox [x, y, width, height] to tuple format
    const bbox: [number, number, number, number] = [
      annotation.bbox[0],
      annotation.bbox[1], 
      annotation.bbox[2],
      annotation.bbox[3]
    ];

    // Convert landmarks from [x, y] array to OpenFace3Landmark objects
    const landmarks_2d = annotation.openface3.landmarks_2d?.map(([x, y]) => ({ x, y }));

    // Convert action units to proper format - map dynamic keys to expected AU names
    let action_units;
    if (annotation.openface3.action_units) {
      const auMap = annotation.openface3.action_units;
      action_units = {
        AU01_Inner_Brow_Raiser: auMap['AU01_Inner_Brow_Raiser'] || { intensity: 0, presence: false },
        AU02_Outer_Brow_Raiser: auMap['AU02_Outer_Brow_Raiser'] || { intensity: 0, presence: false },
        AU04_Brow_Lowerer: auMap['AU04_Brow_Lowerer'] || { intensity: 0, presence: false },
        AU05_Upper_Lid_Raiser: auMap['AU05_Upper_Lid_Raiser'] || { intensity: 0, presence: false },
        AU06_Cheek_Raiser: auMap['AU06_Cheek_Raiser'] || { intensity: 0, presence: false },
        AU07_Lid_Tightener: auMap['AU07_Lid_Tightener'] || { intensity: 0, presence: false },
        AU09_Nose_Wrinkler: auMap['AU09_Nose_Wrinkler'] || { intensity: 0, presence: false },
        AU10_Upper_Lip_Raiser: auMap['AU10_Upper_Lip_Raiser'] || { intensity: 0, presence: false }
      };
    }

    // Convert emotion probabilities to proper format
    let emotion;
    if (annotation.openface3.emotion) {
      const srcEmotion = annotation.openface3.emotion;
      emotion = {
        dominant: srcEmotion.dominant,
        probabilities: {
          neutral: srcEmotion.probabilities['neutral'] || 0,
          happiness: srcEmotion.probabilities['happiness'] || 0,
          sadness: srcEmotion.probabilities['sadness'] || 0,
          anger: srcEmotion.probabilities['anger'] || 0,
          fear: srcEmotion.probabilities['fear'] || 0,
          surprise: srcEmotion.probabilities['surprise'] || 0,
          disgust: srcEmotion.probabilities['disgust'] || 0,
          contempt: srcEmotion.probabilities['contempt'] || 0
        },
        valence: srcEmotion.valence,
        arousal: srcEmotion.arousal,
        confidence: srcEmotion.confidence
      };
    }

    return {
      annotation_id: annotation.id || index,
      bbox: bbox,
      timestamp: timestamp,
      face_confidence: annotation.openface3.confidence,
      openface3: {
        landmarks_2d: landmarks_2d,
        action_units: action_units,
        head_pose: annotation.openface3.head_pose,
        gaze: annotation.openface3.gaze,
        emotion: emotion
      }
    };
  }

  /**
   * Validate COCO+OpenFace3 data structure
   */
  private isValidCOCOOpenFace3Data(data: unknown): data is COCOOpenFace3Data {
    if (!data || typeof data !== 'object') {
      return false;
    }

    const record = data as Record<string, unknown>;

    // Check required top-level structure
    if (!record.info || !record.images || !record.annotations || 
        !Array.isArray(record.images) || !Array.isArray(record.annotations)) {
      return false;
    }

    const info = record.info as Record<string, unknown>;
    const annotations = record.annotations as unknown[];

    // Check info structure (VideoAnnotator COCO Export)
    if (typeof info.description !== 'string' || !info.description.includes('VideoAnnotator')) {
      return false;
    }

    // Check at least one annotation has OpenFace3 data
    if (annotations.length > 0) {
      const hasOpenFace3 = annotations.some((annotation) => {
        if (!annotation || typeof annotation !== 'object') return false;
        const ann = annotation as Record<string, unknown>;
        return !!ann.openface3 && typeof ann.openface3 === 'object';
      });
      if (!hasOpenFace3) {
        return false;
      }
    }

    return true;
  }
}

/**
 * Helper function for easy access
 */
export async function parseCOCOOpenFace3Data(file: File): Promise<StandardFaceAnnotation[]> {
  const parser = COCOOpenFace3Parser.getInstance();
  return parser.parseCOCOOpenFace3Data(file);
}
