/**
 * OpenFace3 Parser
 * Converts OpenFace3 JSON format to StandardFaceAnnotation format
 * Supports all OpenFace3 features: landmarks, action units, head pose, gaze, emotions
 */

import { 
  OpenFace3Data, 
  OpenFace3FaceAnnotation, 
  StandardFaceAnnotation 
} from '../../types/annotations';

export class OpenFace3Parser {
  private static instance: OpenFace3Parser;
  
  public static getInstance(): OpenFace3Parser {
    if (!OpenFace3Parser.instance) {
      OpenFace3Parser.instance = new OpenFace3Parser();
    }
    return OpenFace3Parser.instance;
  }

  /**
   * Main parsing method - converts OpenFace3 JSON to StandardFaceAnnotation[]
   */
  public parseOpenFace3Data(jsonData: any): StandardFaceAnnotation[] {
    try {
      // Validate input structure
      if (!this.isValidOpenFace3Data(jsonData)) {
        throw new Error('Invalid OpenFace3 data structure');
      }

      const openface3Data = jsonData as OpenFace3Data;
      
      // Convert each face annotation
      const standardAnnotations: StandardFaceAnnotation[] = openface3Data.faces.map(
        (face, index) => this.convertFaceAnnotation(face, index)
      );

      console.log(`✓ OpenFace3 Parser: Converted ${standardAnnotations.length} face annotations`);
      return standardAnnotations;

    } catch (error) {
      console.error('OpenFace3 parsing failed:', error);
      throw new Error(`Failed to parse OpenFace3 data: ${error}`);
    }
  }

  /**
   * Convert single OpenFace3 face to StandardFaceAnnotation
   */
  private convertFaceAnnotation(
    face: OpenFace3FaceAnnotation, 
    index: number
  ): StandardFaceAnnotation {
    return {
      annotation_id: face.annotation_id || index,
      bbox: face.bbox,
      timestamp: face.timestamp,
      face_confidence: face.features.confidence,
      openface3: {
        landmarks_2d: face.features.landmarks_2d,
        action_units: face.features.action_units,
        head_pose: face.features.head_pose,
        gaze: face.features.gaze,
        emotion: face.features.emotion
      }
    };
  }

  /**
   * Validate OpenFace3 data structure
   */
  private isValidOpenFace3Data(data: any): data is OpenFace3Data {
    if (!data || typeof data !== 'object') {
      return false;
    }

    // Check required top-level structure
    if (!data.metadata || !data.faces || !Array.isArray(data.faces)) {
      return false;
    }

    // Check metadata structure
    if (!data.metadata.pipeline || !data.metadata.model_info) {
      return false;
    }

    // Check at least one face has valid structure
    if (data.faces.length > 0) {
      const firstFace = data.faces[0];
      if (!firstFace.bbox || !firstFace.features || !firstFace.timestamp) {
        return false;
      }
    }

    return true;
  }

  /**
   * Extract available features from OpenFace3 metadata
   */
  public getAvailableFeatures(openface3Data: OpenFace3Data): string[] {
    const features = openface3Data.metadata.model_info.features;
    const available: string[] = [];

    if (features.landmarks) available.push('landmarks_2d');
    if (features.action_units) available.push('action_units');
    if (features.head_pose) available.push('head_pose');
    if (features.gaze) available.push('gaze');
    if (features.emotions) available.push('emotions');
    if (features.face_tracking) available.push('face_tracking');

    return available;
  }

  /**
   * Get processing statistics from OpenFace3 data
   */
  public getProcessingStats(openface3Data: OpenFace3Data): {
    totalFaces: number;
    avgProcessingTime: number;
    detectionConfidence: number;
    modelInfo: string;
  } {
    return {
      totalFaces: openface3Data.metadata.processing_stats.total_faces,
      avgProcessingTime: openface3Data.metadata.processing_stats.avg_processing_time,
      detectionConfidence: openface3Data.metadata.config.detection_confidence,
      modelInfo: `${openface3Data.metadata.model_info.model_name} v${openface3Data.metadata.model_info.version}`
    };
  }

  /**
   * Filter faces by confidence threshold
   */
  public filterByConfidence(
    annotations: StandardFaceAnnotation[], 
    threshold: number = 0.5
  ): StandardFaceAnnotation[] {
    return annotations.filter(annotation => 
      (annotation.face_confidence || 0) >= threshold
    );
  }

  /**
   * Get unique timestamps for timeline synchronization
   */
  public getTimestamps(annotations: StandardFaceAnnotation[]): number[] {
    const timestamps = annotations.map(annotation => annotation.timestamp);
    return [...new Set(timestamps)].sort((a, b) => a - b);
  }
}

// Export singleton instance
export const openface3Parser = OpenFace3Parser.getInstance();
