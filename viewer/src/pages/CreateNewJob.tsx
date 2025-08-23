import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { ArrowLeft, ArrowRight, Upload, Play } from "lucide-react";
import { Link } from "react-router-dom";

// Wizard steps
const STEPS = [
  { id: 1, title: "Upload Video", description: "Select video file to process" },
  { id: 2, title: "Select Pipelines", description: "Choose annotation pipelines" },
  { id: 3, title: "Configure", description: "Set pipeline parameters" },
  { id: 4, title: "Review & Submit", description: "Review and start job" },
];

const CreateNewJob = () => {
  const [currentStep, setCurrentStep] = useState(1);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const progress = (currentStep / STEPS.length) * 100;

  const nextStep = () => {
    if (currentStep < STEPS.length) {
      setCurrentStep(currentStep + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  const renderStepContent = () => {
    switch (currentStep) {
      case 1:
        return <VideoUploadStep selectedFile={selectedFile} setSelectedFile={setSelectedFile} />;
      case 2:
        return <PipelineSelectionStep />;
      case 3:
        return <ConfigurationStep />;
      case 4:
        return <ReviewStep selectedFile={selectedFile} />;
      default:
        return null;
    }
  };

  const canProceed = () => {
    switch (currentStep) {
      case 1:
        return !!selectedFile;
      case 2:
      case 3:
      case 4:
        return true;
      default:
        return false;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Link to="/create/jobs">
          <Button variant="outline" size="sm">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back
          </Button>
        </Link>
        <div>
          <h2 className="text-2xl font-bold">Create New Annotation Job</h2>
          <p className="text-gray-600">Process a video through the VideoAnnotator pipeline</p>
        </div>
      </div>

      {/* Progress */}
      <Card>
        <CardContent className="pt-6">
          <div className="space-y-4">
            <div className="flex justify-between text-sm">
              <span>Step {currentStep} of {STEPS.length}</span>
              <span>{Math.round(progress)}% Complete</span>
            </div>
            <Progress value={progress} className="h-2" />
            
            <div className="flex justify-between">
              {STEPS.map((step) => (
                <div
                  key={step.id}
                  className={`text-center ${
                    step.id === currentStep
                      ? "text-blue-600"
                      : step.id < currentStep
                      ? "text-green-600"
                      : "text-gray-400"
                  }`}
                >
                  <div className="text-sm font-medium">{step.title}</div>
                  <div className="text-xs">{step.description}</div>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Step Content */}
      <Card className="min-h-[400px]">
        <CardHeader>
          <CardTitle>{STEPS[currentStep - 1].title}</CardTitle>
        </CardHeader>
        <CardContent>
          {renderStepContent()}
        </CardContent>
      </Card>

      {/* Navigation */}
      <div className="flex justify-between">
        <Button
          variant="outline"
          onClick={prevStep}
          disabled={currentStep === 1}
        >
          <ArrowLeft className="h-4 w-4 mr-2" />
          Previous
        </Button>

        {currentStep === STEPS.length ? (
          <Button disabled>
            <Play className="h-4 w-4 mr-2" />
            Submit Job (Coming Soon)
          </Button>
        ) : (
          <Button
            onClick={nextStep}
            disabled={!canProceed()}
          >
            Next
            <ArrowRight className="h-4 w-4 ml-2" />
          </Button>
        )}
      </div>
    </div>
  );
};

// Step Components
const VideoUploadStep = ({ 
  selectedFile, 
  setSelectedFile 
}: { 
  selectedFile: File | null;
  setSelectedFile: (file: File | null) => void;
}) => {
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] || null;
    setSelectedFile(file);
  };

  return (
    <div className="space-y-6">
      <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
        <Upload className="h-12 w-12 mx-auto text-gray-400 mb-4" />
        <h3 className="text-lg font-medium mb-2">Upload Video File</h3>
        <p className="text-gray-600 mb-4">
          Select a video file to process. Supported formats: MP4, WebM, AVI, MOV
        </p>
        
        <input
          type="file"
          accept="video/*"
          onChange={handleFileChange}
          className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-medium file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
        />
      </div>

      {selectedFile && (
        <div className="p-4 bg-green-50 rounded-lg">
          <h4 className="font-medium text-green-800 mb-2">Selected File:</h4>
          <div className="text-sm text-green-700">
            <p>Name: {selectedFile.name}</p>
            <p>Size: {(selectedFile.size / (1024 * 1024)).toFixed(1)} MB</p>
            <p>Type: {selectedFile.type}</p>
          </div>
        </div>
      )}
    </div>
  );
};

const PipelineSelectionStep = () => (
  <div className="space-y-6">
    <p className="text-gray-600">
      Select the annotation pipelines you want to run on your video.
    </p>
    
    <div className="space-y-4">
      {["Scene Detection", "Person Tracking", "Face Analysis", "Audio Processing"].map((pipeline) => (
        <div key={pipeline} className="flex items-center space-x-3 p-3 border rounded-lg">
          <input type="checkbox" className="rounded" defaultChecked />
          <div>
            <div className="font-medium">{pipeline}</div>
            <div className="text-sm text-gray-600">
              {pipeline === "Scene Detection" && "Detect scene boundaries using PySceneDetect + CLIP"}
              {pipeline === "Person Tracking" && "Track people using YOLO11 + ByteTrack"}
              {pipeline === "Face Analysis" && "Analyze faces using OpenFace 3.0"}
              {pipeline === "Audio Processing" && "Speech recognition and speaker diarization"}
            </div>
          </div>
        </div>
      ))}
    </div>
  </div>
);

const ConfigurationStep = () => (
  <div className="space-y-6">
    <p className="text-gray-600">
      Configure pipeline parameters (using default settings for now).
    </p>
    
    <div className="p-4 bg-blue-50 rounded-lg">
      <h4 className="font-medium text-blue-800 mb-2">Default Configuration</h4>
      <pre className="text-sm text-blue-700">
{`{
  "scene_detection": {
    "min_scene_len_sec": 8,
    "use_clip": true
  },
  "person_tracking": {
    "yolo_conf_thresh": 0.5,
    "bytetrack_enabled": true
  },
  "face_analysis": {
    "openface3_enabled": true
  },
  "audio_processing": {
    "whisper_model": "small",
    "diarization": true
  }
}`}
      </pre>
    </div>
  </div>
);

const ReviewStep = ({ selectedFile }: { selectedFile: File | null }) => (
  <div className="space-y-6">
    <p className="text-gray-600">
      Review your job configuration before submission.
    </p>
    
    <div className="space-y-4">
      <div className="p-4 border rounded-lg">
        <h4 className="font-medium mb-2">Video File</h4>
        <p>{selectedFile?.name || "No file selected"}</p>
      </div>
      
      <div className="p-4 border rounded-lg">
        <h4 className="font-medium mb-2">Selected Pipelines</h4>
        <p>Scene Detection, Person Tracking, Face Analysis, Audio Processing</p>
      </div>
      
      <div className="p-4 border rounded-lg">
        <h4 className="font-medium mb-2">Estimated Processing Time</h4>
        <p>~5-10 minutes (depending on video length)</p>
      </div>
    </div>
  </div>
);

export default CreateNewJob;