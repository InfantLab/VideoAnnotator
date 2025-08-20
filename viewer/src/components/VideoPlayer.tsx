import { forwardRef, useEffect, useRef, useCallback, useState } from 'react';
import { StandardAnnotationData, OverlaySettings, COCOPersonAnnotation, WebVTTCue, RTTMSegment, SceneAnnotation, LAIONFaceAnnotation, COCO_SKELETON_CONNECTIONS, YOLO_POSE_PALETTE, YOLO_LIMB_COLORS, YOLO_KEYPOINT_COLORS } from '@/types/annotations';
import { getFacesAtTime, getDominantEmotion } from '@/lib/parsers/face';
import { OpenFace3Settings } from './OpenFace3Controls';

interface VideoPlayerProps {
  videoFile: File;
  annotationData: StandardAnnotationData;
  currentTime: number;
  overlaySettings: OverlaySettings;
  openface3Settings?: OpenFace3Settings;
  onTimeUpdate: (time: number) => void;
  onDurationChange: (duration: number) => void;
  onPlayStateChange: (playing: boolean) => void;
}

export const VideoPlayer = forwardRef<HTMLVideoElement, VideoPlayerProps>(
  ({ videoFile, annotationData, currentTime, overlaySettings, openface3Settings, onTimeUpdate, onDurationChange, onPlayStateChange }, ref) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const [videoUrl, setVideoUrl] = useState<string>('');
    const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

    // Create video URL from file
    useEffect(() => {
      if (videoFile) {
        const url = URL.createObjectURL(videoFile);
        setVideoUrl(url);
        return () => URL.revokeObjectURL(url);
      }
    }, [videoFile]);

    // Get current pose data based on current time
    const getCurrentPoseData = useCallback((): COCOPersonAnnotation[] => {
      if (!annotationData?.person_tracking) return [];

      // Find poses within a small time window around current time (±0.5 seconds for debugging)
      const timeWindow = 0.5;
      return annotationData.person_tracking.filter(pose =>
        Math.abs(pose.timestamp - currentTime) <= timeWindow
      );
    }, [currentTime, annotationData]);

    // Get current speech data
    const getCurrentSpeechData = useCallback((): WebVTTCue | null => {
      if (!annotationData?.speech_recognition) return null;

      return annotationData.speech_recognition.find(cue =>
        currentTime >= cue.startTime && currentTime <= cue.endTime
      ) || null;
    }, [currentTime, annotationData]);

    // Get current speaker data
    const getCurrentSpeakerData = useCallback((): RTTMSegment[] => {
      if (!annotationData?.speaker_diarization) return [];

      return annotationData.speaker_diarization.filter(segment =>
        currentTime >= segment.start_time && currentTime <= segment.end_time
      );
    }, [currentTime, annotationData]);

    // Get current scene data
    const getCurrentSceneData = useCallback((): SceneAnnotation | null => {
      if (!annotationData?.scene_detection) return null;

      return annotationData.scene_detection.find(scene =>
        currentTime >= scene.start_time && currentTime <= scene.end_time
      ) || null;
    }, [currentTime, annotationData]);

    // Get current face data
    const getCurrentFaceData = useCallback((): LAIONFaceAnnotation[] => {
      if (!annotationData?.face_analysis) return [];

      return getFacesAtTime(annotationData.face_analysis, currentTime, 0.1);
    }, [currentTime, annotationData]);

    // Draw COCO pose overlay with YOLO/Ultralytics colors
    const drawPose = useCallback((ctx: CanvasRenderingContext2D, poses: COCOPersonAnnotation[]) => {
      if (!overlaySettings.pose || poses.length === 0) return;

      // Calculate scaling factors from original video dimensions to canvas dimensions
      const scaleX = ctx.canvas.width / dimensions.width;
      const scaleY = ctx.canvas.height / dimensions.height;

      poses.forEach((person, index) => {
        // COCO keypoints are stored as [x1, y1, visibility1, x2, y2, visibility2, ...]
        // Convert to array of keypoint objects for easier handling
        const keypoints = [];
        for (let i = 0; i < person.keypoints.length; i += 3) {
          keypoints.push({
            x: person.keypoints[i] * scaleX,      // Scale X coordinate
            y: person.keypoints[i + 1] * scaleY,  // Scale Y coordinate
            visibility: person.keypoints[i + 2] // 0=not labeled, 1=labeled but not visible, 2=labeled and visible
          });
        }

        // Draw keypoints with YOLO colors (only visible ones)
        keypoints.forEach((keypoint, keypointIndex) => {
          if (keypoint.visibility === 2 && keypointIndex < YOLO_KEYPOINT_COLORS.length) { 
            const colorIndex = YOLO_KEYPOINT_COLORS[keypointIndex];
            const [r, g, b] = YOLO_POSE_PALETTE[colorIndex] || [255, 255, 255];
            
            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            ctx.beginPath();
            ctx.arc(keypoint.x, keypoint.y, 4, 0, 2 * Math.PI);
            ctx.fill();
          }
        });

        // Draw skeleton connections using YOLO colors
        COCO_SKELETON_CONNECTIONS.forEach(([i, j], connectionIndex) => {
          const kp1 = keypoints[i];
          const kp2 = keypoints[j];
          if (kp1 && kp2 && kp1.visibility === 2 && kp2.visibility === 2 && connectionIndex < YOLO_LIMB_COLORS.length) {
            const colorIndex = YOLO_LIMB_COLORS[connectionIndex];
            const [r, g, b] = YOLO_POSE_PALETTE[colorIndex] || [255, 255, 255];
            
            ctx.strokeStyle = `rgb(${r}, ${g}, ${b})`;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(kp1.x, kp1.y);
            ctx.lineTo(kp2.x, kp2.y);
            ctx.stroke();
          }
        });

        // Draw bounding box if enabled
        if (person.bbox && person.bbox.length === 4) {
          // Scale bounding box coordinates
          const scaledX = person.bbox[0] * scaleX;
          const scaledY = person.bbox[1] * scaleY;
          const scaledWidth = person.bbox[2] * scaleX;
          const scaledHeight = person.bbox[3] * scaleY;

          // Use person-specific color based on track_id or index
          const hue = ((person.track_id || index) * 137.508) % 360;
          ctx.strokeStyle = `hsl(${hue}, 70%, 40%)`;
          ctx.lineWidth = 1;
          ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);

          // Draw track ID if available
          if (person.track_id !== undefined) {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            ctx.fillRect(scaledX, scaledY - 20, 40, 18);
            ctx.fillStyle = `hsl(${hue}, 70%, 60%)`;
            ctx.font = '12px monospace';
            ctx.fillText(`ID:${person.track_id}`, scaledX + 2, scaledY - 6);
          }
        }
      });
    }, [overlaySettings.pose, dimensions]);

    // Draw subtitle overlay (from WebVTT)
    const drawSubtitles = useCallback((ctx: CanvasRenderingContext2D) => {
      if (!overlaySettings.subtitles) return;

      const currentSpeech = getCurrentSpeechData();
      if (currentSpeech && currentSpeech.text) {
        const canvasHeight = ctx.canvas.height;
        const text = currentSpeech.text;

        ctx.font = '18px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
        ctx.textAlign = 'center';

        const textWidth = ctx.measureText(text).width;
        const x = ctx.canvas.width / 2;
        const y = canvasHeight - 50;

        // Background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(x - textWidth / 2 - 10, y - 25, textWidth + 20, 35);

        // Text
        ctx.fillStyle = 'white';
        ctx.fillText(text, x, y - 5);
        ctx.textAlign = 'left';
      }
    }, [overlaySettings.subtitles, getCurrentSpeechData]);

    // Draw speaker diarization overlay (from RTTM)
    const drawSpeakers = useCallback((ctx: CanvasRenderingContext2D) => {
      if (!overlaySettings.speakers) return;

      const currentSpeakers = getCurrentSpeakerData();
      if (currentSpeakers.length > 0) {
        currentSpeakers.forEach((speaker, index) => {
          ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
          ctx.fillRect(10, 60 + index * 35, 200, 30);

          const hue = (speaker.speaker_id.charCodeAt(0) * 137.508) % 360;
          ctx.fillStyle = `hsl(${hue}, 70%, 60%)`;
          ctx.font = '14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
          ctx.fillText(`Speaker: ${speaker.speaker_id}`, 15, 80 + index * 35);
        });
      }
    }, [overlaySettings.speakers, getCurrentSpeakerData]);

    // Draw scene overlay
    const drawScenes = useCallback((ctx: CanvasRenderingContext2D) => {
      if (!overlaySettings.scenes) return;

      const currentScene = getCurrentSceneData();
      if (currentScene) {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(10, 10, 250, 30);

        ctx.fillStyle = 'hsl(200, 70%, 60%)';
        ctx.font = '14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
        ctx.fillText(`Scene: ${currentScene.scene_type}`, 15, 30);
      }
    }, [overlaySettings.scenes, getCurrentSceneData]);

    // Draw face detection overlay
    const drawFaces = useCallback((ctx: CanvasRenderingContext2D) => {
      if (!overlaySettings.faces) return;

      const currentFaces = getCurrentFaceData();
      if (currentFaces.length === 0) return;

      // Calculate scaling factors from original video dimensions to canvas dimensions
      const scaleX = ctx.canvas.width / dimensions.width;
      const scaleY = ctx.canvas.height / dimensions.height;

      currentFaces.forEach((face, index) => {
        const [x, y, width, height] = face.bbox;
        
        // Scale bounding box coordinates
        const scaledX = x * scaleX;
        const scaledY = y * scaleY;
        const scaledWidth = width * scaleX;
        const scaledHeight = height * scaleY;
        
        // Use different colors for different faces
        const hue = (face.face_id * 137.508) % 360;
        ctx.strokeStyle = `hsl(${hue}, 70%, 60%)`;
        ctx.lineWidth = 2;

        // Draw face bounding box
        ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);

        // Draw face ID
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(scaledX, scaledY - 20, 60, 18);
        ctx.fillStyle = `hsl(${hue}, 70%, 60%)`;
        ctx.font = '12px monospace';
        ctx.fillText(`Face:${face.face_id}`, scaledX + 2, scaledY - 6);
      });
    }, [overlaySettings.faces, getCurrentFaceData, dimensions]);

    // Draw emotion recognition overlay
    const drawEmotions = useCallback((ctx: CanvasRenderingContext2D) => {
      if (!overlaySettings.emotions) return;

      const currentFaces = getCurrentFaceData();
      if (currentFaces.length === 0) return;

      // Calculate scaling factors from original video dimensions to canvas dimensions
      const scaleX = ctx.canvas.width / dimensions.width;
      const scaleY = ctx.canvas.height / dimensions.height;

      currentFaces.forEach((face, index) => {
        const dominantEmotion = getDominantEmotion(face);
        if (!dominantEmotion) return;

        const [x, y, width, height] = face.bbox;
        
        // Scale coordinates
        const scaledX = x * scaleX;
        const scaledY = y * scaleY;
        const scaledHeight = height * scaleY;
        
        // Position emotion label below face box
        const labelY = scaledY + scaledHeight + 25;
        const emotionText = `${dominantEmotion.emotion}: ${(dominantEmotion.score * 100).toFixed(1)}%`;
        
        // Measure text width for background
        ctx.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
        const textWidth = ctx.measureText(emotionText).width;

        // Draw background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(scaledX, labelY - 15, textWidth + 8, 18);

        // Draw emotion text with color based on emotion type
        const emotionColors: Record<string, string> = {
          'pleasure_ecstasy': 'hsl(60, 70%, 60%)',  // Yellow
          'astonishment_surprise': 'hsl(30, 70%, 60%)', // Orange
          'emotional_vulnerability': 'hsl(240, 70%, 60%)', // Blue
          'pain': 'hsl(0, 70%, 60%)', // Red
          'interest': 'hsl(120, 70%, 60%)', // Green
          'arousal': 'hsl(300, 70%, 60%)', // Pink
          'elation': 'hsl(45, 70%, 60%)', // Gold
          'embarrassment': 'hsl(15, 70%, 60%)' // Red-orange
        };

        ctx.fillStyle = emotionColors[dominantEmotion.emotion] || 'hsl(180, 70%, 60%)';
        ctx.fillText(emotionText, scaledX + 4, labelY - 3);
      });
    }, [overlaySettings.emotions, getCurrentFaceData, dimensions]);

    // =============================================================================
    // OPENFACE3 OVERLAY RENDERING FUNCTIONS
    // =============================================================================

    // Get current OpenFace3 face data at the current timestamp
    const getCurrentOpenFace3Data = useCallback(() => {
      if (!annotationData?.openface3_faces) return [];
      
      return annotationData.openface3_faces.filter(face => 
        Math.abs(face.timestamp - currentTime) < 0.1 // 100ms tolerance
      );
    }, [annotationData?.openface3_faces, currentTime]);

    // Draw OpenFace3 98-point facial landmarks
    const drawOpenFace3Landmarks = useCallback((ctx: CanvasRenderingContext2D) => {
      if (!openface3Settings?.landmarks_2d || !openface3Settings.enabled) return;

      const currentFaces = getCurrentOpenFace3Data();
      if (currentFaces.length === 0) return;

      const scaleX = ctx.canvas.width / dimensions.width;
      const scaleY = ctx.canvas.height / dimensions.height;

      currentFaces.forEach((face, faceIndex) => {
        if (!face.openface3?.landmarks_2d) return;
        
        const landmarks = face.openface3.landmarks_2d;
        const hue = (faceIndex * 137.508) % 360;
        
        // Draw landmarks as small circles
        ctx.fillStyle = `hsl(${hue}, 70%, 60%)`;
        landmarks.forEach((landmark, pointIndex) => {
          const scaledX = landmark.x * scaleX;
          const scaledY = landmark.y * scaleY;
          
          ctx.beginPath();
          ctx.arc(scaledX, scaledY, 1.5, 0, 2 * Math.PI);
          ctx.fill();
          
          // Draw point numbers for key landmarks if labels enabled
          if (openface3Settings.show_feature_labels && pointIndex % 10 === 0) {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
            ctx.font = '8px monospace';
            ctx.fillText(pointIndex.toString(), scaledX + 3, scaledY - 3);
            ctx.fillStyle = `hsl(${hue}, 70%, 60%)`;
          }
        });
      });
    }, [getCurrentOpenFace3Data, openface3Settings?.landmarks_2d, openface3Settings?.enabled, openface3Settings?.show_feature_labels, dimensions]);

    // Draw OpenFace3 Action Units
    const drawOpenFace3ActionUnits = useCallback((ctx: CanvasRenderingContext2D) => {
      if (!openface3Settings?.action_units || !openface3Settings.enabled) return;

      const currentFaces = getCurrentOpenFace3Data();
      if (currentFaces.length === 0) return;

      const scaleX = ctx.canvas.width / dimensions.width;
      const scaleY = ctx.canvas.height / dimensions.height;

      currentFaces.forEach((face, faceIndex) => {
        if (!face.openface3?.action_units) return;
        
        const aus = face.openface3.action_units;
        const [x, y, width, height] = face.bbox;
        const scaledX = x * scaleX;
        const scaledY = y * scaleY;
        
        // Display active Action Units
        const activeAUs = Object.entries(aus).filter(([_, au]) => {
          const actionUnit = au as any; // Type assertion for now
          return actionUnit.presence && actionUnit.intensity > 0.5;
        });
        
        if (activeAUs.length > 0) {
          const labelY = scaledY + (height * scaleY) + 25;
          const auText = activeAUs.map(([name, au]) => {
            const actionUnit = au as any; // Type assertion for now
            return `${name.replace('AU', '').replace('_', ' ')}: ${actionUnit.intensity.toFixed(1)}`;
          }).join(', ');
          
          // Background
          const textWidth = ctx.measureText(auText).width;
          ctx.fillStyle = 'rgba(128, 0, 128, 0.8)'; // Purple background
          ctx.fillRect(scaledX, labelY - 15, textWidth + 8, 18);
          
          // Text
          ctx.fillStyle = 'white';
          ctx.font = '11px monospace';
          ctx.fillText(auText, scaledX + 4, labelY - 3);
        }
      });
    }, [getCurrentOpenFace3Data, openface3Settings?.action_units, openface3Settings?.enabled, dimensions]);

    // Draw OpenFace3 Head Pose (3D orientation)
    const drawOpenFace3HeadPose = useCallback((ctx: CanvasRenderingContext2D) => {
      if (!openface3Settings?.head_pose || !openface3Settings.enabled) return;

      const currentFaces = getCurrentOpenFace3Data();
      if (currentFaces.length === 0) return;

      const scaleX = ctx.canvas.width / dimensions.width;
      const scaleY = ctx.canvas.height / dimensions.height;

      currentFaces.forEach((face, faceIndex) => {
        if (!face.openface3?.head_pose) return;
        
        const headPose = face.openface3.head_pose;
        const [x, y, width, height] = face.bbox;
        const centerX = (x + width / 2) * scaleX;
        const centerY = (y + height / 2) * scaleY;
        
        // Draw orientation vectors
        const arrowLength = 30;
        const pitch = headPose.pitch * Math.PI / 180;
        const yaw = headPose.yaw * Math.PI / 180;
        const roll = headPose.roll * Math.PI / 180;
        
        // Draw coordinate axes (simplified)
        ctx.lineWidth = 2;
        
        // Yaw (horizontal rotation) - Red
        ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)';
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(centerX + Math.cos(yaw) * arrowLength, centerY + Math.sin(yaw) * arrowLength);
        ctx.stroke();
        
        // Pitch (vertical rotation) - Green  
        ctx.strokeStyle = 'rgba(0, 255, 0, 0.8)';
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(centerX, centerY + Math.sin(pitch) * arrowLength);
        ctx.stroke();
        
        // Show numerical values if labels enabled
        if (openface3Settings.show_feature_labels) {
          const poseText = `P:${headPose.pitch.toFixed(0)}° Y:${headPose.yaw.toFixed(0)}° R:${headPose.roll.toFixed(0)}°`;
          const labelY = (y + height) * scaleY + 45;
          
          ctx.fillStyle = 'rgba(0, 0, 255, 0.8)'; // Blue background
          const textWidth = ctx.measureText(poseText).width;
          ctx.fillRect(centerX - textWidth/2 - 4, labelY - 15, textWidth + 8, 18);
          
          ctx.fillStyle = 'white';
          ctx.font = '10px monospace';
          ctx.fillText(poseText, centerX - textWidth/2, labelY - 3);
        }
      });
    }, [getCurrentOpenFace3Data, openface3Settings?.head_pose, openface3Settings?.enabled, openface3Settings?.show_feature_labels, dimensions]);

    // Draw OpenFace3 Gaze Direction
    const drawOpenFace3Gaze = useCallback((ctx: CanvasRenderingContext2D) => {
      if (!openface3Settings?.gaze || !openface3Settings.enabled) return;

      const currentFaces = getCurrentOpenFace3Data();
      if (currentFaces.length === 0) return;

      const scaleX = ctx.canvas.width / dimensions.width;
      const scaleY = ctx.canvas.height / dimensions.height;

      currentFaces.forEach((face, faceIndex) => {
        if (!face.openface3?.gaze) return;
        
        const gaze = face.openface3.gaze;
        const [x, y, width, height] = face.bbox;
        
        // Estimate eye positions (approximate)
        const leftEyeX = (x + width * 0.35) * scaleX;
        const rightEyeX = (x + width * 0.65) * scaleX;
        const eyeY = (y + height * 0.4) * scaleY;
        
        // Draw gaze vectors from both eyes
        const gazeLength = 40;
        const gazeEndX = gaze.direction_x * gazeLength;
        const gazeEndY = gaze.direction_y * gazeLength;
        
        ctx.strokeStyle = 'rgba(0, 255, 255, 0.9)'; // Cyan
        ctx.lineWidth = 2;
        
        // Left eye gaze
        ctx.beginPath();
        ctx.moveTo(leftEyeX, eyeY);
        ctx.lineTo(leftEyeX + gazeEndX, eyeY + gazeEndY);
        ctx.stroke();
        
        // Right eye gaze
        ctx.beginPath();
        ctx.moveTo(rightEyeX, eyeY);
        ctx.lineTo(rightEyeX + gazeEndX, eyeY + gazeEndY);
        ctx.stroke();
        
        // Draw eye points
        ctx.fillStyle = 'rgba(0, 255, 255, 0.9)';
        ctx.beginPath();
        ctx.arc(leftEyeX, eyeY, 3, 0, 2 * Math.PI);
        ctx.arc(rightEyeX, eyeY, 3, 0, 2 * Math.PI);
        ctx.fill();
        
        // Show confidence if labels enabled
        if (openface3Settings.show_feature_labels) {
          const gazeText = `Gaze: ${(gaze.confidence * 100).toFixed(0)}%`;
          const labelY = (y + height) * scaleY + 65;
          
          ctx.fillStyle = 'rgba(0, 128, 128, 0.8)'; // Teal background
          const textWidth = ctx.measureText(gazeText).width;
          ctx.fillRect((x + width/2) * scaleX - textWidth/2 - 4, labelY - 15, textWidth + 8, 18);
          
          ctx.fillStyle = 'white';
          ctx.font = '10px monospace';
          ctx.fillText(gazeText, (x + width/2) * scaleX - textWidth/2, labelY - 3);
        }
      });
    }, [getCurrentOpenFace3Data, openface3Settings?.gaze, openface3Settings?.enabled, openface3Settings?.show_feature_labels, dimensions]);

    // Draw OpenFace3 Enhanced Emotions (8 categories + valence/arousal)
    const drawOpenFace3Emotions = useCallback((ctx: CanvasRenderingContext2D) => {
      if (!openface3Settings?.emotions || !openface3Settings.enabled) return;

      const currentFaces = getCurrentOpenFace3Data();
      if (currentFaces.length === 0) return;

      const scaleX = ctx.canvas.width / dimensions.width;
      const scaleY = ctx.canvas.height / dimensions.height;

      currentFaces.forEach((face, faceIndex) => {
        if (!face.openface3?.emotion) return;
        
        const emotion = face.openface3.emotion;
        const [x, y, width, height] = face.bbox;
        const scaledX = x * scaleX;
        const scaledY = y * scaleY;
        
        // Main emotion display
        const emotionText = `${emotion.dominant} (${(emotion.confidence * 100).toFixed(0)}%)`;
        const labelY = scaledY + (height * scaleY) + 5;
        
        // Emotion-specific colors
        const emotionColors: Record<string, string> = {
          'happiness': 'hsl(60, 100%, 50%)',      // Yellow
          'sadness': 'hsl(240, 100%, 60%)',      // Blue
          'anger': 'hsl(0, 100%, 60%)',          // Red
          'fear': 'hsl(270, 100%, 60%)',         // Purple
          'surprise': 'hsl(30, 100%, 60%)',      // Orange
          'disgust': 'hsl(120, 100%, 30%)',      // Dark Green
          'contempt': 'hsl(300, 100%, 40%)',     // Magenta
          'neutral': 'hsl(0, 0%, 60%)'           // Gray
        };
        
        const emotionColor = emotionColors[emotion.dominant.toLowerCase()] || 'hsl(180, 70%, 60%)';
        
        // Background
        const textWidth = ctx.measureText(emotionText).width;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(scaledX, labelY - 15, textWidth + 8, 18);
        
        // Text
        ctx.fillStyle = emotionColor;
        ctx.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
        ctx.fillText(emotionText, scaledX + 4, labelY - 3);
        
        // Show valence/arousal if labels enabled
        if (openface3Settings.show_feature_labels) {
          const vaText = `V:${emotion.valence.toFixed(1)} A:${emotion.arousal.toFixed(1)}`;
          const vaLabelY = labelY + 20;
          
          ctx.fillStyle = 'rgba(255, 165, 0, 0.8)'; // Orange background
          const vaTextWidth = ctx.measureText(vaText).width;
          ctx.fillRect(scaledX, vaLabelY - 15, vaTextWidth + 8, 18);
          
          ctx.fillStyle = 'white';
          ctx.font = '10px monospace';
          ctx.fillText(vaText, scaledX + 4, vaLabelY - 3);
        }
      });
    }, [getCurrentOpenFace3Data, openface3Settings?.emotions, openface3Settings?.enabled, openface3Settings?.show_feature_labels, dimensions]);

    // Draw OpenFace3 Enhanced Face Boxes
    const drawOpenFace3FaceBoxes = useCallback((ctx: CanvasRenderingContext2D) => {
      if (!openface3Settings?.face_boxes || !openface3Settings.enabled) return;

      const currentFaces = getCurrentOpenFace3Data();
      if (currentFaces.length === 0) return;

      const scaleX = ctx.canvas.width / dimensions.width;
      const scaleY = ctx.canvas.height / dimensions.height;

      currentFaces.forEach((face, faceIndex) => {
        const [x, y, width, height] = face.bbox;
        
        // Scale bounding box coordinates
        const scaledX = x * scaleX;
        const scaledY = y * scaleY;
        const scaledWidth = width * scaleX;
        const scaledHeight = height * scaleY;
        
        // Face-specific color
        const hue = (face.annotation_id * 137.508) % 360;
        const confidence = face.face_confidence || 0;
        
        // Box color changes with confidence
        ctx.strokeStyle = `hsl(${hue}, 70%, ${40 + confidence * 20}%)`;
        ctx.lineWidth = confidence > 0.7 ? 3 : 2;

        // Draw face bounding box
        ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);

        // Draw face info
        const faceText = `OF3:${face.annotation_id}`;
        ctx.fillStyle = 'rgba(0, 100, 255, 0.8)'; // Blue background for OF3
        ctx.fillRect(scaledX, scaledY - 20, 65, 18);
        ctx.fillStyle = `hsl(${hue}, 70%, 80%)`;
        ctx.font = '11px monospace';
        ctx.fillText(faceText, scaledX + 2, scaledY - 6);
        
        // Show confidence if enabled
        if (openface3Settings.show_confidence_scores && face.face_confidence) {
          const confText = `${(face.face_confidence * 100).toFixed(0)}%`;
          ctx.fillStyle = 'rgba(0, 100, 255, 0.8)';
          ctx.fillRect(scaledX + 70, scaledY - 20, 35, 18);
          ctx.fillStyle = 'white';
          ctx.font = '10px monospace';
          ctx.fillText(confText, scaledX + 73, scaledY - 6);
        }
      });
    }, [getCurrentOpenFace3Data, openface3Settings?.face_boxes, openface3Settings?.enabled, openface3Settings?.show_confidence_scores, dimensions]);

    // Render all overlays
    const renderOverlays = useCallback(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Get current data
      const poseData = getCurrentPoseData();

      // DEBUG: Log pose data to console
      if (annotationData?.person_tracking && annotationData.person_tracking.length > 0) {
        if (poseData.length > 0) {
          console.log('🎯 Person tracking data found:', poseData.length, 'people at time', currentTime);
          console.log('  - First person keypoints:', poseData[0].keypoints.length, 'values');
          console.log('  - Person has bbox:', !!poseData[0].bbox);
          console.log('  - Overlay settings pose enabled:', overlaySettings.pose);
        } else {
          console.log('❌ No person tracking data at current time', currentTime, 'but have', annotationData.person_tracking.length, 'total entries');
          console.log('  - Available timestamps:', annotationData.person_tracking.slice(0, 5).map(p => p.timestamp));
          console.log('  - Current time window: ±0.5 seconds around', currentTime);
        }
      } else {
        console.log('❌ No person tracking data loaded at all');
        console.log('Available annotation data keys:', Object.keys(annotationData || {}));
      }

      // Draw all enabled overlays
      drawPose(ctx, poseData);
      drawSubtitles(ctx);
      drawSpeakers(ctx);
      drawScenes(ctx);
      drawFaces(ctx);
      drawEmotions(ctx);
      
      // Draw OpenFace3 overlays if enabled and available
      if (openface3Settings?.enabled) {
        drawOpenFace3FaceBoxes(ctx);
        drawOpenFace3Landmarks(ctx);
        drawOpenFace3ActionUnits(ctx);
        drawOpenFace3HeadPose(ctx);
        drawOpenFace3Gaze(ctx);
        drawOpenFace3Emotions(ctx);
      }
    }, [getCurrentPoseData, drawPose, drawSubtitles, drawSpeakers, drawScenes, drawFaces, drawEmotions, 
        getCurrentOpenFace3Data, drawOpenFace3FaceBoxes, drawOpenFace3Landmarks, drawOpenFace3ActionUnits,
        drawOpenFace3HeadPose, drawOpenFace3Gaze, drawOpenFace3Emotions, 
        currentTime, overlaySettings.pose, annotationData, openface3Settings]);

    // Update overlays when current time changes
    useEffect(() => {
      renderOverlays();
    }, [currentTime, overlaySettings, openface3Settings, renderOverlays]);

    // Handle video events
    const handleLoadedMetadata = useCallback(() => {
      const video = ref as React.MutableRefObject<HTMLVideoElement>;
      if (video.current) {
        onDurationChange(video.current.duration);
        setDimensions({
          width: video.current.videoWidth,
          height: video.current.videoHeight
        });
      }
    }, [onDurationChange, ref]);

    const handleTimeUpdate = useCallback(() => {
      const video = ref as React.MutableRefObject<HTMLVideoElement>;
      if (video.current) {
        onTimeUpdate(video.current.currentTime);
      }
    }, [onTimeUpdate, ref]);

    const handlePlay = useCallback(() => {
      onPlayStateChange(true);
    }, [onPlayStateChange]);

    const handlePause = useCallback(() => {
      onPlayStateChange(false);
    }, [onPlayStateChange]);

    // Resize canvas to match video exactly
    const resizeCanvas = useCallback(() => {
      const canvas = canvasRef.current;
      const video = ref as React.MutableRefObject<HTMLVideoElement>;
      const container = containerRef.current;
      
      if (!canvas || !video.current || !container || !dimensions.width || !dimensions.height) return;

      // Get the actual displayed video dimensions (after object-contain scaling)
      const videoRect = video.current.getBoundingClientRect();
      const containerRect = container.getBoundingClientRect();
      
      // Calculate the actual video display size within the container
      const videoAspectRatio = dimensions.width / dimensions.height;
      const containerAspectRatio = containerRect.width / containerRect.height;
      
      let displayWidth, displayHeight;
      
      if (videoAspectRatio > containerAspectRatio) {
        // Video is wider - constrained by container width
        displayWidth = containerRect.width;
        displayHeight = containerRect.width / videoAspectRatio;
      } else {
        // Video is taller - constrained by container height
        displayHeight = containerRect.height;
        displayWidth = containerRect.height * videoAspectRatio;
      }
      
      // Set canvas size to match the actual video display size
      canvas.width = displayWidth;
      canvas.height = displayHeight;
      canvas.style.width = `${displayWidth}px`;
      canvas.style.height = `${displayHeight}px`;
      
      // Center the canvas within the container
      const leftOffset = (containerRect.width - displayWidth) / 2;
      const topOffset = (containerRect.height - displayHeight) / 2;
      
      canvas.style.position = 'absolute';
      canvas.style.left = `${leftOffset}px`;
      canvas.style.top = `${topOffset}px`;
      
      // Re-render overlays after resize
      renderOverlays();
    }, [dimensions, ref, renderOverlays]);

    // Resize canvas when dimensions change
    useEffect(() => {
      const timeoutId = setTimeout(resizeCanvas, 10); // Small delay to ensure DOM updates
      return () => clearTimeout(timeoutId);
    }, [resizeCanvas]);

    // Resize canvas on window resize and container size changes
    useEffect(() => {
      const handleResize = () => {
        setTimeout(resizeCanvas, 100); // Delay to ensure all elements have resized
      };

      // Use ResizeObserver for more precise container size tracking
      const resizeObserver = new ResizeObserver(handleResize);
      if (containerRef.current) {
        resizeObserver.observe(containerRef.current);
      }

      // Fallback to window resize events
      window.addEventListener('resize', handleResize);
      
      return () => {
        resizeObserver.disconnect();
        window.removeEventListener('resize', handleResize);
      };
    }, [resizeCanvas]);

    return (
      <div ref={containerRef} className="relative w-full h-full flex items-center justify-center bg-video-bg">
        <div className="relative w-full h-full">
          <video
            ref={ref}
            src={videoUrl}
            className="w-full h-full object-contain"
            onLoadedMetadata={handleLoadedMetadata}
            onTimeUpdate={handleTimeUpdate}
            onPlay={handlePlay}
            onPause={handlePause}
            style={{ display: 'block' }}
          />
          <canvas
            ref={canvasRef}
            className="absolute inset-0 pointer-events-none"
          />
        </div>
      </div>
    );
  }
);