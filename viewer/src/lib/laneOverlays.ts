/**
 * The Video page switches overlays by lane kind (spec 028); the player still
 * draws through its older switches. These map one onto the other, with each
 * switch's default where the researcher hasn't chosen.
 */
import { defaultOpenFace3Settings, type OpenFace3Settings } from '@/components/openface3Settings';
import type { OverlaySettings } from '@/types/annotations';

const on = (overlays: Record<string, boolean>, key: string, fallback = true) => overlays[key] ?? fallback;

export function overlaySettingsFromLanes(overlays: Record<string, boolean>): OverlaySettings {
  return {
    pose: on(overlays, 'people'),
    subtitles: on(overlays, 'speech'),
    speakers: on(overlays, 'speakers'),
    scenes: on(overlays, 'scenes'),
    faces: on(overlays, 'faces'),
    emotions: on(overlays, 'emotion'),
  };
}

/** The Faces lane's OpenFace 3 menu (spec 028 T023), by overlay key. */
export const OPENFACE3_OVERLAYS = [
  { key: 'faces.landmarks', label: 'Landmarks', setting: 'landmarks_2d' },
  { key: 'faces.gaze', label: 'Gaze', setting: 'gaze' },
  { key: 'faces.head_pose', label: 'Head pose', setting: 'head_pose' },
  { key: 'faces.action_units', label: 'Action units', setting: 'action_units' },
] as const;

export function openface3SettingsFromLanes(overlays: Record<string, boolean>): OpenFace3Settings {
  const settings: OpenFace3Settings = {
    ...defaultOpenFace3Settings,
    face_boxes: on(overlays, 'faces.openface3'),
    emotions: on(overlays, 'emotion.openface3'),
  };
  for (const { key, setting } of OPENFACE3_OVERLAYS) settings[setting] = on(overlays, key, defaultOpenFace3Settings[setting]);
  return settings;
}
