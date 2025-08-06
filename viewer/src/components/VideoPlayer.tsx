import { forwardRef, useEffect, useRef, useCallback, useState } from 'react';
import { StandardAnnotationData, OverlaySettings, COCOPersonAnnotation, WebVTTCue, RTTMSegment, SceneAnnotation, LAIONFaceAnnotation, COCO_SKELETON_CONNECTIONS, YOLO_POSE_PALETTE, YOLO_LIMB_COLORS, YOLO_KEYPOINT_COLORS } from '@/types/annotations';
import { getFacesAtTime, getDominantEmotion } from '@/lib/parsers/face';

interface VideoPlayerProps {
  videoFile: File;
  annotationData: StandardAnnotationData;
  currentTime: number;
  overlaySettings: OverlaySettings;
  onTimeUpdate: (time: number) => void;
  onDurationChange: (duration: number) => void;
  onPlayStateChange: (playing: boolean) => void;
}

export const VideoPlayer = forwardRef<HTMLVideoElement, VideoPlayerProps>(
  ({ videoFile, annotationData, currentTime, overlaySettings, onTimeUpdate, onDurationChange, onPlayStateChange }, ref) => {
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

      poses.forEach((person, index) => {
        // COCO keypoints are stored as [x1, y1, visibility1, x2, y2, visibility2, ...]
        // Convert to array of keypoint objects for easier handling
        const keypoints = [];
        for (let i = 0; i < person.keypoints.length; i += 3) {
          keypoints.push({
            x: person.keypoints[i],
            y: person.keypoints[i + 1],
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
          // Use person-specific color based on track_id or index
          const hue = ((person.track_id || index) * 137.508) % 360;
          ctx.strokeStyle = `hsl(${hue}, 70%, 40%)`;
          ctx.lineWidth = 1;
          ctx.strokeRect(person.bbox[0], person.bbox[1], person.bbox[2], person.bbox[3]);

          // Draw track ID if available
          if (person.track_id !== undefined) {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            ctx.fillRect(person.bbox[0], person.bbox[1] - 20, 40, 18);
            ctx.fillStyle = `hsl(${hue}, 70%, 60%)`;
            ctx.font = '12px monospace';
            ctx.fillText(`ID:${person.track_id}`, person.bbox[0] + 2, person.bbox[1] - 6);
          }
        }
      });
    }, [overlaySettings.pose]);

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

      currentFaces.forEach((face, index) => {
        const [x, y, width, height] = face.bbox;
        
        // Use different colors for different faces
        const hue = (face.face_id * 137.508) % 360;
        ctx.strokeStyle = `hsl(${hue}, 70%, 60%)`;
        ctx.lineWidth = 2;

        // Draw face bounding box
        ctx.strokeRect(x, y, width, height);

        // Draw face ID
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(x, y - 20, 60, 18);
        ctx.fillStyle = `hsl(${hue}, 70%, 60%)`;
        ctx.font = '12px monospace';
        ctx.fillText(`Face:${face.face_id}`, x + 2, y - 6);
      });
    }, [overlaySettings.faces, getCurrentFaceData]);

    // Draw emotion recognition overlay
    const drawEmotions = useCallback((ctx: CanvasRenderingContext2D) => {
      if (!overlaySettings.emotions) return;

      const currentFaces = getCurrentFaceData();
      if (currentFaces.length === 0) return;

      currentFaces.forEach((face, index) => {
        const dominantEmotion = getDominantEmotion(face);
        if (!dominantEmotion) return;

        const [x, y, width, height] = face.bbox;
        
        // Position emotion label below face box
        const labelY = y + height + 25;
        const emotionText = `${dominantEmotion.emotion}: ${(dominantEmotion.score * 100).toFixed(1)}%`;
        
        // Measure text width for background
        ctx.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
        const textWidth = ctx.measureText(emotionText).width;

        // Draw background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(x, labelY - 15, textWidth + 8, 18);

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
        ctx.fillText(emotionText, x + 4, labelY - 3);
      });
    }, [overlaySettings.emotions, getCurrentFaceData]);

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
    }, [getCurrentPoseData, drawPose, drawSubtitles, drawSpeakers, drawScenes, drawFaces, drawEmotions, currentTime, overlaySettings.pose, annotationData]);

    // Update overlays when current time changes
    useEffect(() => {
      renderOverlays();
    }, [currentTime, overlaySettings, renderOverlays]);

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

    // Resize canvas to match video
    useEffect(() => {
      const canvas = canvasRef.current;
      const container = containerRef.current;
      if (!canvas || !container || !dimensions.width || !dimensions.height) return;

      const containerRect = container.getBoundingClientRect();
      const aspectRatio = dimensions.width / dimensions.height;

      let width = containerRect.width;
      let height = width / aspectRatio;

      if (height > containerRect.height) {
        height = containerRect.height;
        width = height * aspectRatio;
      }

      canvas.width = width;
      canvas.height = height;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
    }, [dimensions]);

    return (
      <div ref={containerRef} className="relative w-full h-full flex items-center justify-center">
        <div className="relative">
          <video
            ref={ref}
            src={videoUrl}
            className="max-w-full max-h-full"
            onLoadedMetadata={handleLoadedMetadata}
            onTimeUpdate={handleTimeUpdate}
            onPlay={handlePlay}
            onPause={handlePause}
            style={{ display: 'block' }}
          />
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 pointer-events-none"
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'contain'
            }}
          />
        </div>
      </div>
    );
  }
);