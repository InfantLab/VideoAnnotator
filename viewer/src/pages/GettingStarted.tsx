import { useState } from 'react';
import { Link } from 'react-router-dom';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import {
  Upload, Server, Play, BookOpen, Settings,
  Users, Smile, Mic, MessageSquare, Film,
  ArrowRight, Download, Github, ExternalLink,
} from 'lucide-react';
import { installAllBundledDemos, DEMO_LABELS } from '@/lib/localLibrary/installDemoDataset';
import { useToast } from '@/hooks/use-toast';
import { createCopyAction } from '@/lib/toastHelpers';
import { GITHUB_URL } from '@/utils/version';
import viewerPreview from '@/assets/VideoAnnotationViewer.png';

const annotationTypes = [
  {
    icon: <Users className="w-5 h-5" />,
    title: 'Pose Detection',
    description: 'COCO-format skeletal pose overlays with 17-point body landmarks',
  },
  {
    icon: <Smile className="w-5 h-5" />,
    title: 'Facial Emotions',
    description: 'OpenFace3 facial action units, gaze direction, and emotion recognition',
  },
  {
    icon: <Mic className="w-5 h-5" />,
    title: 'Speech Transcription',
    description: 'Whisper-based transcription with word-level timestamps (WebVTT)',
  },
  {
    icon: <MessageSquare className="w-5 h-5" />,
    title: 'Speaker Diarization',
    description: 'RTTM speaker labels identifying who is speaking when',
  },
  {
    icon: <Film className="w-5 h-5" />,
    title: 'Scene Detection',
    description: 'Automatic scene boundary detection and visual change analysis',
  },
];

export default function GettingStarted() {
  const { toast } = useToast();
  const [isInstallingDemo, setIsInstallingDemo] = useState(false);
  const [demoProgress, setDemoProgress] = useState<string | null>(null);

  const installDemos = async () => {
    setIsInstallingDemo(true);
    setDemoProgress(null);
    try {
      const result = await installAllBundledDemos((msg) => setDemoProgress(msg));

      if (result.installed === 0 && result.failed.length === 0) {
        toast({ title: 'Demo datasets already installed', description: `All ${result.skipped} demos are up to date.`, duration: 4000 });
      } else if (result.failed.length > 0) {
        const detail = result.failed.map(f => `${DEMO_LABELS[f.key] ?? f.key}: ${f.error}`).join('\n');
        const fullText = `Demo install errors\n\nInstalled ${result.installed}, failed ${result.failed.length}.\n${detail}`;
        toast({ title: 'Some demos failed to install', description: `Installed ${result.installed}, failed ${result.failed.length}.`, variant: 'destructive', duration: 12000, action: createCopyAction(fullText) });
      } else {
        toast({ title: 'Demo datasets installed', description: `Installed ${result.installed} demo dataset${result.installed === 1 ? '' : 's'}.`, duration: 5000 });
      }
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : String(error);
      toast({ title: 'Demo install failed', description: message, variant: 'destructive', duration: 10000, action: createCopyAction(`Demo install failed\n\n${message}`) });
    } finally {
      setIsInstallingDemo(false);
      setDemoProgress(null);
    }
  };

  return (
    <div className="container mx-auto px-6 py-10 max-w-5xl">
      {/* Header */}
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold mb-4">Getting Started</h1>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
          Video Annotation Viewer (VAV) helps researchers visualize multimodal video annotations
          with interactive overlays and a synchronized timeline.
        </p>
      </div>

      {/* Interface Preview */}
      <Card className="overflow-hidden border-0 shadow-xl mb-12">
        <img
          src={viewerPreview}
          alt="Video Annotation Viewer showing pose detection, timeline, and controls"
          className="w-full h-auto"
        />
      </Card>

      {/* Two Ways to Use */}
      <h2 className="text-2xl font-bold text-center mb-6">Two Ways to Use VAV</h2>
      <div className="grid md:grid-cols-2 gap-6 mb-12">
        {/* Local Files */}
        <Card className="p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-lg bg-blue-100 text-blue-600 flex items-center justify-center">
              <Upload className="w-5 h-5" />
            </div>
            <h3 className="text-xl font-semibold">View Local Files</h3>
          </div>
          <p className="text-muted-foreground mb-4">
            Upload a video and its annotation JSON directly in your browser. No server needed.
          </p>
          <ol className="space-y-2 mb-6 text-sm">
            <li className="flex items-start gap-2">
              <span className="font-semibold text-primary min-w-[1.5rem]">1.</span>
              Open the Viewer page
            </li>
            <li className="flex items-start gap-2">
              <span className="font-semibold text-primary min-w-[1.5rem]">2.</span>
              Upload a video file and its JSON annotation data
            </li>
            <li className="flex items-start gap-2">
              <span className="font-semibold text-primary min-w-[1.5rem]">3.</span>
              Explore with interactive overlays and timeline
            </li>
          </ol>
          <Link to="/viewer">
            <Button className="w-full gap-2">
              <Upload className="w-4 h-4" />
              Open Viewer
              <ArrowRight className="w-4 h-4" />
            </Button>
          </Link>
        </Card>

        {/* Server */}
        <Card className="p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-lg bg-green-100 text-green-600 flex items-center justify-center">
              <Server className="w-5 h-5" />
            </div>
            <h3 className="text-xl font-semibold">Use with VideoAnnotator</h3>
          </div>
          <p className="text-muted-foreground mb-4">
            Connect to a VideoAnnotator server to create annotation jobs and view results.
          </p>
          <ol className="space-y-2 mb-6 text-sm">
            <li className="flex items-start gap-2">
              <span className="font-semibold text-primary min-w-[1.5rem]">1.</span>
              Configure your server connection in Settings
            </li>
            <li className="flex items-start gap-2">
              <span className="font-semibold text-primary min-w-[1.5rem]">2.</span>
              Create annotation jobs by uploading videos
            </li>
            <li className="flex items-start gap-2">
              <span className="font-semibold text-primary min-w-[1.5rem]">3.</span>
              View results in the Viewer or save to your Library
            </li>
          </ol>
          <Link to="/settings">
            <Button variant="outline" className="w-full gap-2">
              <Settings className="w-4 h-4" />
              Configure Server
              <ArrowRight className="w-4 h-4" />
            </Button>
          </Link>
        </Card>
      </div>

      {/* Try the Demo */}
      <Card className="p-6 mb-12 border-primary/20 bg-primary/5">
        <div className="text-center">
          <div className="flex items-center justify-center gap-3 mb-3">
            <Play className="w-6 h-6 text-primary" />
            <h2 className="text-2xl font-bold">Try the Demo</h2>
          </div>
          <p className="text-muted-foreground mb-6 max-w-xl mx-auto">
            Install bundled sample datasets to explore the viewer without any server.
            Includes example videos with pose detection, speech, and emotion annotations.
          </p>
          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <Button
              size="lg"
              onClick={() => void installDemos()}
              disabled={isInstallingDemo}
              className="gap-2"
            >
              <Download className="w-5 h-5" />
              {isInstallingDemo ? (demoProgress ?? 'Installing demos…') : 'Install Demo Datasets'}
            </Button>
            <Link to="/library">
              <Button size="lg" variant="outline" className="gap-2">
                <BookOpen className="w-5 h-5" />
                Open Library
              </Button>
            </Link>
          </div>
        </div>
      </Card>

      {/* Supported Annotation Types */}
      <h2 className="text-2xl font-bold text-center mb-6">Supported Annotation Types</h2>
      <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-12">
        {annotationTypes.map((type, index) => (
          <Card key={index} className="p-4">
            <div className="flex items-start gap-3">
              <div className="w-9 h-9 rounded-lg bg-primary/10 text-primary flex items-center justify-center flex-shrink-0 mt-0.5">
                {type.icon}
              </div>
              <div>
                <h3 className="font-semibold mb-1">{type.title}</h3>
                <p className="text-sm text-muted-foreground">{type.description}</p>
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* Help Section */}
      <Card className="p-6">
        <h2 className="text-xl font-bold mb-4">Need Help?</h2>
        <div className="grid sm:grid-cols-3 gap-4">
          <a
            href="https://deepwiki.com/InfantLab/video-annotation-viewer"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            <ExternalLink className="w-4 h-4" />
            Documentation
          </a>
          <a
            href={`${GITHUB_URL}/issues`}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            <Github className="w-4 h-4" />
            Report an Issue
          </a>
          <a
            href="https://github.com/InfantLab/VideoAnnotator#server-setup"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            <Server className="w-4 h-4" />
            Server Setup Guide
          </a>
        </div>
      </Card>
    </div>
  );
}
