import { useState } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Input } from '@/components/ui/input';
import { Copy, Download, Search, FileText } from 'lucide-react';
import { StandardAnnotationData } from '@/types/annotations';

interface FileViewerProps {
  annotationData: StandardAnnotationData;
  trigger?: React.ReactNode;
  defaultTab?: string;
}

export const FileViewer = ({ annotationData, trigger, defaultTab = 'summary' }: FileViewerProps) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [activeTab, setActiveTab] = useState(defaultTab);

  // Reset tab to default when dialog opens
  const handleDialogOpen = () => {
    setActiveTab(defaultTab);
    setSearchTerm('');
  };

  const formatJSON = (obj: any, indent = 2) => {
    return JSON.stringify(obj, null, indent);
  };

  const highlightJSON = (jsonString: string, searchTerm: string) => {
    if (!searchTerm) return jsonString;
    
    const regex = new RegExp(`(${searchTerm.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&')})`, 'gi');
    return jsonString.replace(regex, '<mark className="bg-yellow-200">$1</mark>');
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    // TODO: Add toast notification
  };

  const downloadJSON = (data: any, filename: string) => {
    const blob = new Blob([formatJSON(data)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const getDataSummary = () => {
    const summary = {
      video_info: annotationData.video_info,
      data_counts: {
        person_tracking: annotationData.person_tracking?.length || 0,
        face_analysis: annotationData.face_analysis?.length || 0,
        speech_recognition: annotationData.speech_recognition?.length || 0,
        speaker_diarization: annotationData.speaker_diarization?.length || 0,
        scene_detection: annotationData.scene_detection?.length || 0,
      },
      metadata: annotationData.metadata,
      pipelines_detected: annotationData.metadata?.pipelines || [],
      processing_time: annotationData.metadata?.processing_time || 0,
    };
    return summary;
  };

  const getFilteredData = (data: any) => {
    if (!searchTerm) return data;
    
    const jsonString = formatJSON(data);
    return jsonString.includes(searchTerm.toLowerCase()) ? data : null;
  };

  const defaultTrigger = (
    <Button variant="ghost" size="sm" className="h-6 w-6 p-0 text-xs">
      📄
    </Button>
  );

  return (
    <Dialog onOpenChange={(open) => open && handleDialogOpen()}>
      <DialogTrigger asChild>
        {trigger || defaultTrigger}
      </DialogTrigger>
      <DialogContent className="max-w-4xl h-[80vh]">
        <DialogHeader className="pb-4">
          <DialogTitle className="flex items-center gap-2">
            <FileText className="w-5 h-5" />
            JSON Data Viewer
          </DialogTitle>
        </DialogHeader>

        <div className="flex flex-col h-full">
          {/* Search and Controls */}
          <div className="flex items-center gap-2 mb-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                placeholder="Search JSON content..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => copyToClipboard(formatJSON(annotationData))}
              className="flex items-center gap-2"
            >
              <Copy className="w-4 h-4" />
              Copy All
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => downloadJSON(annotationData, 'annotation-data.json')}
              className="flex items-center gap-2"
            >
              <Download className="w-4 h-4" />
              Download
            </Button>
          </div>

          {/* Tabbed Content */}
          <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col">
            <TabsList className="grid w-full grid-cols-6">
              <TabsTrigger value="summary">📊 Summary</TabsTrigger>
              <TabsTrigger value="person">👤 Person</TabsTrigger>
              <TabsTrigger value="face">😊 Face</TabsTrigger>
              <TabsTrigger value="speech">🔵 Speech</TabsTrigger>
              <TabsTrigger value="speakers">🟣 Speakers</TabsTrigger>
              <TabsTrigger value="scenes">🎬 Scenes</TabsTrigger>
            </TabsList>

            <div className="flex-1 mt-4">
              <TabsContent value="summary" className="h-full">
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 bg-muted rounded-lg">
                      <h4 className="font-medium mb-2">📹 Video Information</h4>
                      <pre className="text-xs text-muted-foreground whitespace-pre-wrap">
                        {formatJSON(annotationData.video_info)}
                      </pre>
                    </div>
                    <div className="p-4 bg-muted rounded-lg">
                      <h4 className="font-medium mb-2">📈 Data Counts</h4>
                      <pre className="text-xs text-muted-foreground whitespace-pre-wrap">
                        {formatJSON(getDataSummary().data_counts)}
                      </pre>
                    </div>
                  </div>
                  <ScrollArea className="h-64">
                    <pre className="text-xs text-muted-foreground whitespace-pre-wrap p-4 bg-muted rounded-lg">
                      {highlightJSON(formatJSON(getDataSummary()), searchTerm)}
                    </pre>
                  </ScrollArea>
                </div>
              </TabsContent>

              <TabsContent value="person" className="h-full">
                <div className="flex justify-between items-center mb-4">
                  <h4 className="font-medium">👤 Person Tracking Data ({annotationData.person_tracking?.length || 0} entries)</h4>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => downloadJSON(annotationData.person_tracking, 'person-tracking.json')}
                    disabled={!annotationData.person_tracking?.length}
                  >
                    Download
                  </Button>
                </div>
                <ScrollArea className="h-96">
                  <pre className="text-xs whitespace-pre-wrap p-4 bg-muted rounded-lg">
                    {annotationData.person_tracking?.length 
                      ? highlightJSON(formatJSON(getFilteredData(annotationData.person_tracking)), searchTerm)
                      : 'No person tracking data available'
                    }
                  </pre>
                </ScrollArea>
              </TabsContent>

              <TabsContent value="face" className="h-full">
                <div className="flex justify-between items-center mb-4">
                  <h4 className="font-medium">😊 Face Analysis Data ({annotationData.face_analysis?.length || 0} entries)</h4>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => downloadJSON(annotationData.face_analysis, 'face-analysis.json')}
                    disabled={!annotationData.face_analysis?.length}
                  >
                    Download
                  </Button>
                </div>
                <ScrollArea className="h-96">
                  <pre className="text-xs whitespace-pre-wrap p-4 bg-muted rounded-lg">
                    {annotationData.face_analysis?.length 
                      ? highlightJSON(formatJSON(getFilteredData(annotationData.face_analysis)), searchTerm)
                      : 'No face analysis data available'
                    }
                  </pre>
                </ScrollArea>
              </TabsContent>

              <TabsContent value="speech" className="h-full">
                <div className="flex justify-between items-center mb-4">
                  <h4 className="font-medium">🔵 Speech Recognition Data ({annotationData.speech_recognition?.length || 0} entries)</h4>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => downloadJSON(annotationData.speech_recognition, 'speech-recognition.json')}
                    disabled={!annotationData.speech_recognition?.length}
                  >
                    Download
                  </Button>
                </div>
                <ScrollArea className="h-96">
                  <pre className="text-xs whitespace-pre-wrap p-4 bg-muted rounded-lg">
                    {annotationData.speech_recognition?.length 
                      ? highlightJSON(formatJSON(getFilteredData(annotationData.speech_recognition)), searchTerm)
                      : 'No speech recognition data available'
                    }
                  </pre>
                </ScrollArea>
              </TabsContent>

              <TabsContent value="speakers" className="h-full">
                <div className="flex justify-between items-center mb-4">
                  <h4 className="font-medium">🟣 Speaker Diarization Data ({annotationData.speaker_diarization?.length || 0} entries)</h4>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => downloadJSON(annotationData.speaker_diarization, 'speaker-diarization.json')}
                    disabled={!annotationData.speaker_diarization?.length}
                  >
                    Download
                  </Button>
                </div>
                <ScrollArea className="h-96">
                  <pre className="text-xs whitespace-pre-wrap p-4 bg-muted rounded-lg">
                    {annotationData.speaker_diarization?.length 
                      ? highlightJSON(formatJSON(getFilteredData(annotationData.speaker_diarization)), searchTerm)
                      : 'No speaker diarization data available'
                    }
                  </pre>
                </ScrollArea>
              </TabsContent>

              <TabsContent value="scenes" className="h-full">
                <div className="flex justify-between items-center mb-4">
                  <h4 className="font-medium">🎬 Scene Detection Data ({annotationData.scene_detection?.length || 0} entries)</h4>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => downloadJSON(annotationData.scene_detection, 'scene-detection.json')}
                    disabled={!annotationData.scene_detection?.length}
                  >
                    Download
                  </Button>
                </div>
                <ScrollArea className="h-96">
                  <pre className="text-xs whitespace-pre-wrap p-4 bg-muted-foreground rounded-lg">
                    {annotationData.scene_detection?.length 
                      ? highlightJSON(formatJSON(getFilteredData(annotationData.scene_detection)), searchTerm)
                      : 'No scene detection data available'
                    }
                  </pre>
                </ScrollArea>
              </TabsContent>
            </div>
          </Tabs>
        </div>
      </DialogContent>
    </Dialog>
  );
};