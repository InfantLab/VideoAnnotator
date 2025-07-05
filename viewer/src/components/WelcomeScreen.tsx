import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Play, Upload, FileText, Eye } from 'lucide-react';
import viewerPreview from '@/assets/viewer-interface-preview.jpg';

interface WelcomeScreenProps {
  onGetStarted: () => void;
}

export const WelcomeScreen = ({ onGetStarted }: WelcomeScreenProps) => {
  const features = [
    {
      icon: <Eye className="w-6 h-6" />,
      title: "Multimodal Overlays",
      description: "View pose detection, facial emotions, audio sentiment, and events overlaid on video"
    },
    {
      icon: <Play className="w-6 h-6" />,
      title: "Synchronized Timeline", 
      description: "Navigate through rich timeline with waveforms, motion graphs, and event markers"
    },
    {
      icon: <FileText className="w-6 h-6" />,
      title: "JSON Annotations",
      description: "Load your analysis results from any computer vision or audio processing pipeline"
    }
  ];

  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-transparent to-accent/10" />
        
        <div className="container mx-auto px-6 py-16 relative">
          <div className="text-center max-w-4xl mx-auto mb-12">
            <h1 className="text-5xl font-bold mb-6 bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              Video Annotation Viewer
            </h1>
            <p className="text-xl text-muted-foreground mb-8 leading-relaxed">
              Advanced multimodal analysis tool for reviewing video annotations with pose detection, 
              emotion recognition, audio analysis, and interactive timeline visualization.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button 
                size="lg" 
                onClick={onGetStarted}
                className="text-lg px-8 py-3 animate-pulse-glow"
              >
                <Upload className="w-5 h-5 mr-2" />
                Get Started
              </Button>
              <Button 
                variant="outline" 
                size="lg"
                className="text-lg px-8 py-3"
              >
                <Play className="w-5 h-5 mr-2" />
                View Demo
              </Button>  
            </div>
          </div>

          {/* Interface Preview */}
          <div className="max-w-6xl mx-auto mb-16">
            <Card className="overflow-hidden border-0 shadow-2xl">
              <img 
                src={viewerPreview} 
                alt="Video Annotation Viewer Interface"
                className="w-full h-auto"
              />
            </Card>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="py-16 bg-card/50">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold text-center mb-12">Powerful Analysis Features</h2>
          
          <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
            {features.map((feature, index) => (
              <Card key={index} className="p-6 text-center hover:shadow-lg transition-shadow">
                <div className="w-12 h-12 mx-auto mb-4 bg-primary/10 rounded-lg flex items-center justify-center text-primary">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-semibold mb-3">{feature.title}</h3>
                <p className="text-muted-foreground">{feature.description}</p>
              </Card>
            ))}
          </div>
        </div>
      </div>

      {/* How it Works */}
      <div className="py-16">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold text-center mb-12">How It Works</h2>
          
          <div className="max-w-4xl mx-auto">
            <div className="grid md:grid-cols-3 gap-8">
              <div className="text-center">
                <div className="w-16 h-16 mx-auto mb-4 bg-primary rounded-full flex items-center justify-center text-primary-foreground text-2xl font-bold">
                  1
                </div>
                <h3 className="text-lg font-semibold mb-2">Load Your Files</h3>
                <p className="text-muted-foreground">Upload a video file and its corresponding JSON annotation data</p>
              </div>
              
              <div className="text-center">
                <div className="w-16 h-16 mx-auto mb-4 bg-accent rounded-full flex items-center justify-center text-accent-foreground text-2xl font-bold">
                  2
                </div>
                <h3 className="text-lg font-semibold mb-2">Configure Views</h3>
                <p className="text-muted-foreground">Toggle different annotation overlays and timeline tracks to focus on what matters</p>
              </div>
              
              <div className="text-center">
                <div className="w-16 h-16 mx-auto mb-4 bg-secondary rounded-full flex items-center justify-center text-secondary-foreground text-2xl font-bold">
                  3
                </div>
                <h3 className="text-lg font-semibold mb-2">Analyze & Navigate</h3>
                <p className="text-muted-foreground">Use the interactive timeline to jump to events and analyze multimodal data</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="py-16 bg-gradient-to-r from-primary/10 to-accent/10">
        <div className="container mx-auto px-6 text-center">
          <h2 className="text-3xl font-bold mb-4">Ready to Analyze Your Video Data?</h2>
          <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
            Start exploring your multimodal annotations with our powerful visualization tools.
          </p>
          <Button 
            size="lg" 
            onClick={onGetStarted}
            className="text-lg px-8 py-3"
          >
            <Upload className="w-5 h-5 mr-2" />
            Upload Your Files
          </Button>
        </div>
      </div>
    </div>
  );
};