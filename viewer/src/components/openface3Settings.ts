export interface OpenFace3Settings {
  // Master toggle
  enabled: boolean;

  // Individual feature toggles
  landmarks_2d: boolean;
  action_units: boolean;
  head_pose: boolean;
  gaze: boolean;
  emotions: boolean;
  face_boxes: boolean;

  // Display options
  confidence_threshold: number;
  show_confidence_scores: boolean;
  show_feature_labels: boolean;
}

export const defaultOpenFace3Settings: OpenFace3Settings = {
  enabled: true,
  landmarks_2d: true,
  action_units: false,
  head_pose: false,
  gaze: false,
  emotions: true,
  face_boxes: true,
  confidence_threshold: 0.5,
  show_confidence_scores: false,
  show_feature_labels: true,
};
