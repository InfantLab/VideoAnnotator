/**
 * OpenFace3 Controls Component
 * Hierarchical control system for OpenFace3 features
 * Supports master/child toggle relationships with color coding
 */

import React from 'react';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import { StandardFaceAnnotation } from '@/types/annotations';

// OpenFace3 Control Settings Interface
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

// Default settings
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
  show_feature_labels: true
};

interface OpenFace3ControlsProps {
  settings: OpenFace3Settings;
  onChange: (settings: OpenFace3Settings) => void;
  faceData?: StandardFaceAnnotation[] | null;
  isCollapsed?: boolean;
}

export const OpenFace3Controls = ({ 
  settings, 
  onChange, 
  faceData,
  isCollapsed = false 
}: OpenFace3ControlsProps) => {
  
  // Feature availability check
  const getFeatureAvailability = () => {
    if (!faceData || faceData.length === 0) {
      return {
        landmarks_2d: false,
        action_units: false,
        head_pose: false,
        gaze: false,
        emotions: false,
        face_boxes: false
      };
    }

    // Check if any face has each feature
    const sampleFace = faceData[0];
    const openface3Data = sampleFace.openface3;
    
    return {
      landmarks_2d: !!(openface3Data?.landmarks_2d && openface3Data.landmarks_2d.length > 0),
      action_units: !!(openface3Data?.action_units),
      head_pose: !!(openface3Data?.head_pose),
      gaze: !!(openface3Data?.gaze),
      emotions: !!(openface3Data?.emotion),
      face_boxes: true // Always available if we have face data
    };
  };

  const availability = getFeatureAvailability();
  const hasData = faceData && faceData.length > 0;

  // Master toggle handler
  const handleMasterToggle = () => {
    const newEnabled = !settings.enabled;
    onChange({
      ...settings,
      enabled: newEnabled,
      // Disable all children if master is disabled
      landmarks_2d: newEnabled ? settings.landmarks_2d : false,
      action_units: newEnabled ? settings.action_units : false,
      head_pose: newEnabled ? settings.head_pose : false,
      gaze: newEnabled ? settings.gaze : false,
      emotions: newEnabled ? settings.emotions : false,
      face_boxes: newEnabled ? settings.face_boxes : false
    });
  };

  // Individual feature toggle handler
  const handleFeatureToggle = (feature: keyof Omit<OpenFace3Settings, 'enabled' | 'confidence_threshold' | 'show_confidence_scores' | 'show_feature_labels'>) => {
    const newSettings = {
      ...settings,
      [feature]: !settings[feature]
    };

    // Auto-enable master if any child is enabled
    if (!settings[feature] && !settings.enabled) {
      newSettings.enabled = true;
    }

    // Auto-disable master if all children are disabled
    const childFeatures: (keyof OpenFace3Settings)[] = ['landmarks_2d', 'action_units', 'head_pose', 'gaze', 'emotions', 'face_boxes'];
    const anyChildEnabled = childFeatures.some(key => key === feature ? newSettings[key] : settings[key]);
    if (!anyChildEnabled) {
      newSettings.enabled = false;
    }

    onChange(newSettings);
  };

  // Toggle all features
  const handleToggleAll = () => {
    const allEnabled = settings.landmarks_2d && settings.action_units && settings.head_pose && 
                      settings.gaze && settings.emotions && settings.face_boxes;
    
    onChange({
      ...settings,
      enabled: !allEnabled,
      landmarks_2d: !allEnabled && availability.landmarks_2d,
      action_units: !allEnabled && availability.action_units,
      head_pose: !allEnabled && availability.head_pose,
      gaze: !allEnabled && availability.gaze,
      emotions: !allEnabled && availability.emotions,
      face_boxes: !allEnabled && availability.face_boxes
    });
  };

  // Confidence threshold handler
  const handleConfidenceChange = (value: number) => {
    onChange({
      ...settings,
      confidence_threshold: value
    });
  };

  // Get statistics
  const getStats = () => {
    if (!faceData) return null;
    
    const totalFaces = faceData.length;
    const avgConfidence = faceData.reduce((sum, face) => 
      sum + (face.face_confidence || 0), 0) / totalFaces;
    
    return {
      totalFaces,
      avgConfidence: Math.round(avgConfidence * 100) / 100
    };
  };

  const stats = getStats();

  // Hide component entirely if no face data available
  if (!hasData) {
    return null;
  }

  if (isCollapsed) {
    return (
      <div className="flex items-center gap-2">
        <Label htmlFor="openface3-master" className="text-sm font-medium text-blue-600">
          OpenFace3
        </Label>
        <Switch
          id="openface3-master"
          checked={settings.enabled}
          onCheckedChange={handleMasterToggle}
          disabled={!hasData}
        />
        {stats && (
          <Badge variant="secondary" className="text-xs">
            {stats.totalFaces} faces
          </Badge>
        )}
      </div>
    );
  }

  return (
    <Card className="w-full">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium text-blue-600 flex items-center gap-2">
            OpenFace3 Features
            {stats && (
              <Badge variant="secondary" className="text-xs">
                {stats.totalFaces} faces • ~{stats.avgConfidence} conf
              </Badge>
            )}
          </CardTitle>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleToggleAll}
              disabled={!hasData}
              className="text-xs"
            >
              Toggle All
            </Button>
            <Switch
              id="openface3-master"
              checked={settings.enabled}
              onCheckedChange={handleMasterToggle}
              disabled={!hasData}
            />
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-3">
        {/* Core Features */}
        <div className="space-y-2">
          <div className="grid grid-cols-2 gap-2">
            {/* Landmarks */}
            <div className="flex items-center space-x-2">
              <Switch
                id="landmarks-2d"
                checked={settings.landmarks_2d && settings.enabled}
                onCheckedChange={() => handleFeatureToggle('landmarks_2d')}
                disabled={!availability.landmarks_2d || !hasData}
              />
              <Label 
                htmlFor="landmarks-2d" 
                className={`text-xs ${availability.landmarks_2d ? 'text-green-600' : 'text-gray-400'}`}
              >
                Landmarks (98pts)
              </Label>
            </div>

            {/* Face Boxes */}
            <div className="flex items-center space-x-2">
              <Switch
                id="face-boxes"
                checked={settings.face_boxes && settings.enabled}
                onCheckedChange={() => handleFeatureToggle('face_boxes')}
                disabled={!availability.face_boxes || !hasData}
              />
              <Label 
                htmlFor="face-boxes" 
                className={`text-xs ${availability.face_boxes ? 'text-green-600' : 'text-gray-400'}`}
              >
                Face Boxes
              </Label>
            </div>

            {/* Emotions */}
            <div className="flex items-center space-x-2">
              <Switch
                id="emotions"
                checked={settings.emotions && settings.enabled}
                onCheckedChange={() => handleFeatureToggle('emotions')}
                disabled={!availability.emotions || !hasData}
              />
              <Label 
                htmlFor="emotions" 
                className={`text-xs ${availability.emotions ? 'text-orange-600' : 'text-gray-400'}`}
              >
                Emotions (8 types)
              </Label>
            </div>

            {/* Action Units */}
            <div className="flex items-center space-x-2">
              <Switch
                id="action-units"
                checked={settings.action_units && settings.enabled}
                onCheckedChange={() => handleFeatureToggle('action_units')}
                disabled={!availability.action_units || !hasData}
              />
              <Label 
                htmlFor="action-units" 
                className={`text-xs ${availability.action_units ? 'text-purple-600' : 'text-gray-400'}`}
              >
                Action Units (8 AUs)
              </Label>
            </div>
          </div>
        </div>

        <Separator />

        {/* Advanced Features */}
        <div className="space-y-2">
          <div className="grid grid-cols-2 gap-2">
            {/* Head Pose */}
            <div className="flex items-center space-x-2">
              <Switch
                id="head-pose"
                checked={settings.head_pose && settings.enabled}
                onCheckedChange={() => handleFeatureToggle('head_pose')}
                disabled={!availability.head_pose || !hasData}
              />
              <Label 
                htmlFor="head-pose" 
                className={`text-xs ${availability.head_pose ? 'text-blue-600' : 'text-gray-400'}`}
              >
                Head Pose (3D)
              </Label>
            </div>

            {/* Gaze */}
            <div className="flex items-center space-x-2">
              <Switch
                id="gaze"
                checked={settings.gaze && settings.enabled}
                onCheckedChange={() => handleFeatureToggle('gaze')}
                disabled={!availability.gaze || !hasData}
              />
              <Label 
                htmlFor="gaze" 
                className={`text-xs ${availability.gaze ? 'text-teal-600' : 'text-gray-400'}`}
              >
                Gaze Direction
              </Label>
            </div>
          </div>
        </div>

        <Separator />

        {/* Display Options */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label htmlFor="confidence-threshold" className="text-xs text-gray-600">
              Confidence: {settings.confidence_threshold}
            </Label>
            <input
              id="confidence-threshold"
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={settings.confidence_threshold}
              onChange={(e) => handleConfidenceChange(parseFloat(e.target.value))}
              className="w-20"
              disabled={!hasData}
            />
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Switch
                id="show-confidence"
                checked={settings.show_confidence_scores}
                onCheckedChange={(checked) => onChange({ ...settings, show_confidence_scores: checked })}
                disabled={!hasData}
              />
              <Label htmlFor="show-confidence" className="text-xs text-gray-600">
                Show Confidence
              </Label>
            </div>

            <div className="flex items-center space-x-2">
              <Switch
                id="show-labels"
                checked={settings.show_feature_labels}
                onCheckedChange={(checked) => onChange({ ...settings, show_feature_labels: checked })}
                disabled={!hasData}
              />
              <Label htmlFor="show-labels" className="text-xs text-gray-600">
                Show Labels
              </Label>
            </div>
          </div>
        </div>

        {!hasData && (
          <div className="text-xs text-gray-500 text-center py-2">
            Load OpenFace3 data to enable controls
          </div>
        )}
      </CardContent>
    </Card>
  );
};
