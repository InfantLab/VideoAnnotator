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
      addLog(`Testing ${filename}...`);
      const response = await fetch(`demo/videos_out/3/${filename}`);
      const blob = await response.blob();
      const file = new File([blob], filename, { type: 'application/json' });
      
      addLog(`File size: ${file.size} bytes`);
      
      // Show first 500 characters
      const sample = await file.slice(0, 500).text();
      addLog(`Content sample: ${sample.substring(0, 100)}...`);
      
      // Test detection
      const detected = await detectFileType(file);
      addLog(`Detection result: ${detected.type} (confidence: ${detected.confidence})`);
      
      try {
        const data = JSON.parse(sample);
        addLog(`JSON keys: ${Object.keys(data).slice(0, 10).join(', ')}`);
        addLog(`Is array: ${Array.isArray(data)}`);
        if (Array.isArray(data) && data.length > 0) {
          addLog(`First item keys: ${Object.keys(data[0] || {}).slice(0, 5).join(', ')}`);
        }
      } catch (e) {
        addLog(`JSON parse failed: ${e.message}`);
      }
    } catch (error) {
      addLog(`Failed to load file: ${error.message}`);
    }
  };

  const runAllTests = async () => {
    setIsRunning(true);
    clearLogs();
    
    const testFiles = [
      '3_face_analysis.json',
      '3_scene_detection.json', 
      'scene_results.json',
      '3_person_tracking.json'
    ];
    
    for (const filename of testFiles) {
      await testFileDetection(filename);
      addLog('---');
    }
    
    setIsRunning(false);
  };

  const testDatasetIntegrity = async () => {
    setIsRunning(true);
    addLog('Testing all datasets...');
    
    try {
      const { checkDataIntegrity } = await import('../utils/debugUtils');
      
      for (const [key, _] of Object.entries(DEMO_DATA_SETS)) {
        addLog(`Checking ${key}...`);
        const result = await checkDataIntegrity(key as keyof typeof DEMO_DATA_SETS);
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