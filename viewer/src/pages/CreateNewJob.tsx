import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { ArrowLeft, ArrowRight, Upload, Play, X, AlertCircle } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import { apiClient, handleAPIError } from "@/api/client";
import { Alert, AlertDescription } from "@/components/ui/alert";
import vavIcon from "@/assets/v-a-v.icon.png";

// Wizard steps
const STEPS = [
  { id: 1, title: "Upload Videos", description: "Select video files to process" },
  { id: 2, title: "Select Pipelines", description: "Choose annotation pipelines" },
  { id: 3, title: "Configure", description: "Set pipeline parameters" },
  { id: 4, title: "Review & Submit", description: "Review and start jobs" },
];

// Available pipelines
const AVAILABLE_PIPELINES = [
  {
    name: "scene_detection",
    displayName: "Scene Detection",
    description: "Identifies scene changes and transitions in video using PySceneDetect + CLIP. Detects cuts, fades, and content-based scene boundaries.",
    enabled: true
  },
  {
    name: "person_tracking", 
    displayName: "Person Tracking",
    description: "Tracks people movement throughout the video using YOLO11 + ByteTrack. Provides bounding boxes, pose keypoints, and persistent person IDs.",
    enabled: true
  },
  {
    name: "face_analysis",
    displayName: "Face Analysis", 
    description: "Facial expression analysis, demographic estimation, and facial landmark detection using OpenFace 3.0. Includes age, gender, and emotion recognition.",
    enabled: true
  },
  {
    name: "audio_processing",
    displayName: "Audio Processing",
    description: "Speech recognition with Whisper and speaker diarization with pyannote. Transcribes speech and identifies different speakers with timestamps.",
    enabled: true
  }
];

// Default configuration
const DEFAULT_CONFIG = {
  scene_detection: {
    min_scene_len_sec: 8,
    use_clip: true
  },
  person_tracking: {
    yolo_conf_thresh: 0.5,
    bytetrack_enabled: true
  },
  face_analysis: {
    openface3_enabled: true
  },
  audio_processing: {
    whisper_model: "small",
    diarization: true
  }
};

const CreateNewJob = () => {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState(1);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [selectedPipelines, setSelectedPipelines] = useState<string[]>(
    AVAILABLE_PIPELINES.filter(p => p.enabled).map(p => p.name)
  );
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitSuccess, setSubmitSuccess] = useState<string[]>([]);

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

  const handleSubmitJobs = async () => {
    console.log('🚀 Submit button clicked - starting job submission');
    console.log('Selected files:', selectedFiles.map(f => f.name));
    console.log('Selected pipelines:', selectedPipelines);
    console.log('Config:', config);
    
    if (selectedFiles.length === 0) {
      setSubmitError("No videos selected");
      return;
    }

    if (selectedPipelines.length === 0) {
      setSubmitError("No pipelines selected");
      return;
    }

    setIsSubmitting(true);
    setSubmitError(null);
    setSubmitSuccess([]);

    const jobIds: string[] = [];
    const errors: string[] = [];

    try {
      console.log(`📤 Submitting ${selectedFiles.length} job(s) to VideoAnnotator API...`);
      
      // Submit each video as a separate job
      for (const file of selectedFiles) {
        try {
          console.log(`📹 Submitting job for: ${file.name}`);
          const response = await apiClient.submitJob(
            file,
            selectedPipelines,
            config
          );
          console.log(`✅ Job created successfully: ${response.id}`);
          jobIds.push(response.id);
        } catch (error) {
          console.error(`❌ Job submission failed for ${file.name}:`, error);
          const errorMsg = handleAPIError(error);
          errors.push(`${file.name}: ${errorMsg}`);
        }
      }

      if (jobIds.length > 0) {
        setSubmitSuccess(jobIds);
        
        // If all jobs succeeded, navigate to jobs list after a delay
        if (errors.length === 0) {
          setTimeout(() => {
            navigate('/create/jobs');
          }, 2000);
        }
      }

      if (errors.length > 0) {
        console.warn(`⚠️ Some jobs failed: ${errors.length} out of ${selectedFiles.length}`);
        setSubmitError(`Failed to submit ${errors.length} job(s): \n${errors.join('\n')}`);
      }
      
      if (jobIds.length === 0 && errors.length > 0) {
        console.error('❌ All job submissions failed');
        setSubmitError(
          `All job submissions failed. Common issues:\n\n` +
          `• VideoAnnotator API server not running (check http://localhost:8000)\n` +
          `• Invalid API token or authentication\n` +
          `• Network connectivity issues\n\n` +
          `Errors:\n${errors.join('\n')}`
        );
      }
    } catch (error) {
      console.error('💥 Unexpected error during job submission:', error);
      setSubmitError(handleAPIError(error));
    } finally {
      setIsSubmitting(false);
      console.log('🏁 Job submission process completed');
    }
  };

  const renderStepContent = () => {
    switch (currentStep) {
      case 1:
        return (
          <VideoUploadStep 
            selectedFiles={selectedFiles} 
            setSelectedFiles={setSelectedFiles} 
          />
        );
      case 2:
        return (
          <PipelineSelectionStep 
            selectedPipelines={selectedPipelines}
            setSelectedPipelines={setSelectedPipelines}
          />
        );
      case 3:
        return (
          <ConfigurationStep 
            config={config}
            setConfig={setConfig}
          />
        );
      case 4:
        return (
          <ReviewStep 
            selectedFiles={selectedFiles}
            selectedPipelines={selectedPipelines}
            config={config}
            onSubmit={handleSubmitJobs}
            isSubmitting={isSubmitting}
            submitError={submitError}
            submitSuccess={submitSuccess}
          />
        );
      default:
        return null;
    }
  };

  const canProceed = () => {
    switch (currentStep) {
      case 1:
        return selectedFiles.length > 0;
      case 2:
        return selectedPipelines.length > 0;
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
        <div className="flex items-center gap-3">
          <img src={vavIcon} alt="VideoAnnotator" className="h-8 w-8" />
          <div>
            <h2 className="text-2xl font-bold text-foreground">Create New Annotation Jobs</h2>
            <p className="text-muted-foreground">Process videos through the VideoAnnotator pipeline (supports batch processing)</p>
          </div>
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
                      : "text-muted-foreground"
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
          <Button 
            onClick={handleSubmitJobs}
            disabled={isSubmitting || !canProceed()}
          >
            <Play className="h-4 w-4 mr-2" />
            {isSubmitting ? 'Submitting...' : `Submit ${selectedFiles.length} Job${selectedFiles.length > 1 ? 's' : ''}`}
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
  selectedFiles, 
  setSelectedFiles 
}: { 
  selectedFiles: File[];
  setSelectedFiles: (files: File[]) => void;
}) => {
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    setSelectedFiles([...selectedFiles, ...files]);
  };

  const removeFile = (index: number) => {
    setSelectedFiles(selectedFiles.filter((_, i) => i !== index));
  };

  const totalSize = selectedFiles.reduce((sum, file) => sum + file.size, 0);

  return (
    <div className="space-y-6">
      <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
        <Upload className="h-12 w-12 mx-auto text-gray-400 mb-4" />
        <h3 className="text-lg font-medium mb-2">Upload Video Files</h3>
        <p className="text-muted-foreground mb-4">
          Select video files to process. Supports batch processing. Formats: MP4, WebM, AVI, MOV
        </p>
        
        <input
          type="file"
          accept="video/*"
          multiple
          onChange={handleFileChange}
          className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-medium file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
        />
      </div>

      {selectedFiles.length > 0 && (
        <div className="p-4 bg-green-50 rounded-lg">
          <div className="flex justify-between items-center mb-2">
            <h4 className="font-medium text-green-800">
              Selected Files ({selectedFiles.length}):
            </h4>
            <span className="text-sm text-green-700">
              Total: {(totalSize / (1024 * 1024)).toFixed(1)} MB
            </span>
          </div>
          <div className="space-y-2 max-h-40 overflow-y-auto">
            {selectedFiles.map((file, index) => (
              <div key={index} className="flex justify-between items-center bg-white p-2 rounded">
                <div className="text-sm text-green-700">
                  <span className="font-medium">{file.name}</span>
                  <span className="text-gray-500 ml-2">
                    ({(file.size / (1024 * 1024)).toFixed(1)} MB)
                  </span>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => removeFile(index)}
                  className="h-6 w-6 p-0 text-red-600 hover:text-red-800"
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

const PipelineSelectionStep = ({
  selectedPipelines,
  setSelectedPipelines
}: {
  selectedPipelines: string[];
  setSelectedPipelines: (pipelines: string[]) => void;
}) => {
  const togglePipeline = (pipelineName: string) => {
    if (selectedPipelines.includes(pipelineName)) {
      setSelectedPipelines(selectedPipelines.filter(p => p !== pipelineName));
    } else {
      setSelectedPipelines([...selectedPipelines, pipelineName]);
    }
  };

  return (
    <div className="space-y-6">
      <p className="text-foreground">
        Select the annotation pipelines you want to run on your videos.
      </p>
      
      <div className="space-y-4">
        {AVAILABLE_PIPELINES.map((pipeline) => (
          <div key={pipeline.name} className="flex items-center space-x-3 p-3 border rounded-lg bg-card">
            <input 
              type="checkbox" 
              className="rounded" 
              checked={selectedPipelines.includes(pipeline.name)}
              onChange={() => togglePipeline(pipeline.name)}
              disabled={!pipeline.enabled}
            />
            <div className="flex-1">
              <div className="font-medium text-foreground">{pipeline.displayName}</div>
              <div className="text-sm text-muted-foreground">
                {pipeline.description}
              </div>
            </div>
          </div>
        ))}
      </div>
      
      {selectedPipelines.length === 0 && (
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Please select at least one pipeline to proceed.
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
};

const ConfigurationStep = ({
  config,
  setConfig
}: {
  config: typeof DEFAULT_CONFIG;
  setConfig: (config: typeof DEFAULT_CONFIG) => void;
}) => {
  return (
    <div className="space-y-6">
      <p className="text-muted-foreground">
        Configure pipeline parameters. You can modify the configuration or use the default settings.
      </p>
      
      <div className="p-4 bg-blue-50 rounded-lg">
        <h4 className="font-medium text-blue-800 mb-2">Pipeline Configuration</h4>
        <div className="bg-white p-3 rounded border">
          <textarea
            value={JSON.stringify(config, null, 2)}
            onChange={(e) => {
              try {
                const newConfig = JSON.parse(e.target.value);
                setConfig(newConfig);
              } catch {
                // Invalid JSON, don't update
              }
            }}
            className="w-full h-64 text-sm font-mono resize-none border focus:ring-2 focus:ring-blue-500 focus:border-blue-500 rounded px-3 py-2 text-foreground bg-background"
            placeholder="Pipeline configuration JSON..."
          />
        </div>
        <p className="text-xs text-blue-600 mt-2">
          Edit the JSON configuration above. Invalid JSON will be ignored.
        </p>
      </div>
    </div>
  );
};

const ReviewStep = ({ 
  selectedFiles, 
  selectedPipelines, 
  config,
  onSubmit,
  isSubmitting,
  submitError,
  submitSuccess
}: { 
  selectedFiles: File[];
  selectedPipelines: string[];
  config: typeof DEFAULT_CONFIG;
  onSubmit: () => void;
  isSubmitting: boolean;
  submitError: string | null;
  submitSuccess: string[];
}) => {
  const totalSize = selectedFiles.reduce((sum, file) => sum + file.size, 0);
  const pipelineNames = selectedPipelines.map(name => 
    AVAILABLE_PIPELINES.find(p => p.name === name)?.displayName || name
  ).join(", ");

  return (
    <div className="space-y-6">
      <p className="text-muted-foreground">
        Review your job configuration before submission.
      </p>
      
      {submitError && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription className="whitespace-pre-line">
            {submitError}
          </AlertDescription>
        </Alert>
      )}
      
      {submitSuccess.length > 0 && (
        <Alert>
          <AlertDescription>
            Successfully submitted {submitSuccess.length} job(s). Job IDs: {submitSuccess.join(", ")}
            {submitError ? "" : " Redirecting to jobs list..."}
          </AlertDescription>
        </Alert>
      )}
      
      <div className="space-y-4">
        <div className="p-4 border rounded-lg">
          <h4 className="font-medium mb-2">Video Files ({selectedFiles.length})</h4>
          <div className="space-y-1">
            {selectedFiles.slice(0, 3).map((file, index) => (
              <p key={index} className="text-sm text-foreground">
                {file.name} ({(file.size / (1024 * 1024)).toFixed(1)} MB)
              </p>
            ))}
            {selectedFiles.length > 3 && (
              <p className="text-sm text-muted-foreground">...and {selectedFiles.length - 3} more files</p>
            )}
            <p className="text-sm font-medium text-muted-foreground">
              Total size: {(totalSize / (1024 * 1024)).toFixed(1)} MB
            </p>
          </div>
        </div>
        
        <div className="p-4 border rounded-lg">
          <h4 className="font-medium mb-2">Selected Pipelines</h4>
          <p>{pipelineNames || "None selected"}</p>
        </div>
        
        <div className="p-4 border rounded-lg">
          <h4 className="font-medium mb-2">Estimated Processing Time</h4>
          <p>~{Math.ceil(selectedFiles.length * 7)} minutes (depending on video lengths and selected pipelines)</p>
        </div>
        
        <div className="p-4 border rounded-lg">
          <h4 className="font-medium mb-2">Configuration Preview</h4>
          <pre className="text-xs text-muted-foreground bg-muted p-2 rounded overflow-x-auto">
            {JSON.stringify(config, null, 2)}
          </pre>
        </div>
      </div>
    </div>
  );
};

export default CreateNewJob;