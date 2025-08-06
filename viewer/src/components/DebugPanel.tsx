import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { DEMO_DATA_SETS } from '@/utils/debugUtils';
import { detectFileType } from '@/lib/parsers/merger';

interface DebugPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

export const DebugPanel = ({ isOpen, onClose }: DebugPanelProps) => {
  const [logs, setLogs] = useState<string[]>([]);
  const [isRunning, setIsRunning] = useState(false);

  const addLog = (message: string) => {
    setLogs(prev => [...prev, `${new Date().toLocaleTimeString()}: ${message}`]);
  };

  const clearLogs = () => setLogs([]);

  const testFileDetection = async (filename: string) => {
    try {
      addLog(`Testing ${filename} (simulating drag & drop)...`);
      const response = await fetch(`demo/videos_out/3/${filename}`);
      
      if (!response.ok) {
        addLog(`❌ HTTP ${response.status}: ${response.statusText}`);
        return;
      }
      
      const blob = await response.blob();
      const file = new File([blob], filename, { type: 'application/json' });
      
      addLog(`File size: ${file.size} bytes`);
      
      // Show first 200 characters for preview
      const sample = await file.slice(0, 200).text();
      addLog(`Content sample: ${sample}${sample.length === 200 ? '...' : ''}`);
      
      // === SIMULATE EXACT DRAG & DROP DETECTION PIPELINE ===
      // This is the same code as FileUploader.tsx detectFiles function
      
      // Step 1: Initial fileUtils detection
      const { detectFileType: fileUtilsDetect, detectJSONType } = await import('@/lib/fileUtils');
      let detected = fileUtilsDetect(file);
      addLog(`📝 Step 1 - fileUtils detection: ${detected.type} (${detected.confidence})`);
      
      // Step 2: JSON content analysis (if unknown JSON)
      if (detected.type === 'unknown' && detected.extension === 'json') {
        addLog(`🔍 Step 2 - JSON file detected as unknown, trying detectJSONType...`);
        try {
          detected = await detectJSONType(file);
          addLog(`📝 Step 2 - detectJSONType result: ${detected.type} (${detected.confidence})`);
          
          // Step 3: Merger fallback (if still unknown)
          if (detected.type === 'unknown') {
            addLog(`🔍 Step 3 - Still unknown, trying merger fallback...`);
            const { detectFileType: mergerDetect } = await import('@/lib/parsers/merger');
            const mergerResult = await mergerDetect(file);
            
            // Convert merger result to fileUtils format (same as FileUploader)
            detected = {
              type: mergerResult.type === 'complete_results' ? 'unknown' : mergerResult.type as any,
              extension: 'json',
              mimeType: 'application/json',
              confidence: mergerResult.confidence > 0.7 ? 'high' : 
                         mergerResult.confidence > 0.4 ? 'medium' : 'low',
              reason: `Detected via content analysis (${mergerResult.confidence.toFixed(2)} confidence)`
            };
            addLog(`📝 Step 3 - merger result: ${detected.type} (${detected.confidence}) - confidence: ${mergerResult.confidence.toFixed(3)}`);
          }
        } catch (error) {
          addLog(`❌ JSON detection failed: ${error.message}`);
        }
      }
      
      addLog(`🎯 FINAL RESULT: ${detected.type} (${detected.confidence})`);
      addLog(`   Reason: ${detected.reason}`);
      
      // Show if this would be accepted by FileUploader
      const wouldBeAccepted = detected.type !== 'unknown';
      addLog(`📋 Would be accepted by drag & drop: ${wouldBeAccepted ? '✅ YES' : '❌ NO - shows as "Unknown file type"'}`);
      
    } catch (error) {
      addLog(`❌ Failed to load file: ${error.message}`);
    }
  };

  const runAllTests = async () => {
    setIsRunning(true);
    clearLogs();
    
    // Test files that we know exist based on the complete_results.json structure
    const testFiles = [
      '3_scene_detection.json', 
      'scene_results.json',
      '3_person_tracking.json'
    ];
    
    // Also test additional VEATIC files
    const additionalFiles = [
      '3_laion_face_annotations.json',
      'face_results.json',
      'person_results.json',
      '3_person_tracks.json',
      'complete_results.json'
    ];
    
    for (const filename of testFiles) {
      await testFileDetection(filename);
      addLog('---');
    }
    
    addLog('Testing additional VEATIC files...');
    for (const filename of additionalFiles) {
      try {
        const response = await fetch(`demo/videos_out/3/${filename}`);
        if (response.ok) {
          addLog(`✅ Found: ${filename}`);
          await testFileDetection(filename);
        } else {
          addLog(`❌ Not found: ${filename} (${response.status})`);
        }
      } catch (error) {
        addLog(`❌ Error checking ${filename}: ${error.message}`);
      }
      addLog('---');
    }
    
    setIsRunning(false);
  };

  const testDatasetIntegrity = async () => {
    setIsRunning(true);
    addLog('Testing all datasets...');
    
    try {
      // Access the function from the global window object
      const debugUtils = (window as any).debugUtils;
      if (!debugUtils || !debugUtils.checkDataIntegrity) {
        addLog('❌ debugUtils.checkDataIntegrity not available');
        addLog('Make sure debug utilities are loaded');
        setIsRunning(false);
        return;
      }
      
      for (const [key, _] of Object.entries(DEMO_DATA_SETS)) {
        addLog(`Checking ${key}...`);
        const result = await debugUtils.checkDataIntegrity(key);
        addLog(`${key}: ${result.valid ? '✅ Valid' : '❌ Issues found'}`);
        if (!result.valid) {
          result.issues.forEach(issue => addLog(`  - ${issue}`));
        }
      }
    } catch (error) {
      addLog(`Error: ${error.message}`);
    }
    
    setIsRunning(false);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
      <Card className="w-full max-w-4xl max-h-[80vh] overflow-hidden flex flex-col">
        <div className="p-4 border-b flex items-center justify-between">
          <h2 className="text-lg font-semibold">🐛 Debug Panel</h2>
          <Button variant="ghost" onClick={onClose}>✕</Button>
        </div>
        
        <div className="p-4 border-b">
          <div className="flex gap-2">
            <Button 
              onClick={runAllTests}
              disabled={isRunning}
              size="sm"
            >
              🧪 Test VEATIC Files
            </Button>
            <Button 
              onClick={testDatasetIntegrity}
              disabled={isRunning}
              size="sm"
            >
              🔍 Check All Datasets
            </Button>
            <Button 
              onClick={clearLogs}
              variant="outline"
              size="sm"
            >
              🗑️ Clear Logs
            </Button>
          </div>
        </div>
        
        <div className="flex-1 overflow-auto p-4">
          <div className="bg-black text-green-400 p-4 rounded font-mono text-sm max-h-full overflow-auto">
            {logs.length === 0 ? (
              <div className="text-gray-500">No logs yet. Click a test button to start.</div>
            ) : (
              logs.map((log, i) => (
                <div key={i} className="mb-1">
                  {log}
                </div>
              ))
            )}
            {isRunning && (
              <div className="text-yellow-400">⏳ Running tests...</div>
            )}
          </div>
        </div>
      </Card>
    </div>
  );
};