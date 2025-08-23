import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Database, Plus, Folder, Info } from "lucide-react";

const CreateDatasets = () => {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold">Dataset Management</h2>
          <p className="text-gray-600">Manage video datasets for batch processing</p>
        </div>
        <Button disabled>
          <Plus className="h-4 w-4 mr-2" />
          Register Dataset (Coming Soon)
        </Button>
      </div>

      {/* Info Alert */}
      <Alert>
        <Info className="h-4 w-4" />
        <AlertDescription>
          Dataset management will allow you to register video collections for batch processing.
          This feature is planned for a future release.
        </AlertDescription>
      </Alert>

      {/* Placeholder Content */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <Card className="border-dashed border-2">
          <CardContent className="flex flex-col items-center justify-center p-8 text-center">
            <Database className="h-12 w-12 text-gray-400 mb-4" />
            <h3 className="font-medium text-gray-600 mb-2">No Datasets Registered</h3>
            <p className="text-sm text-gray-500 mb-4">
              Register your first video dataset to enable batch processing
            </p>
            <Button variant="outline" disabled>
              <Plus className="h-4 w-4 mr-2" />
              Add Dataset
            </Button>
          </CardContent>
        </Card>

        {/* Example dataset cards for future reference */}
        <Card className="opacity-50">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <Folder className="h-5 w-5" />
              Example Dataset
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 text-sm">
              <div>
                <span className="font-medium">Path:</span>
                <span className="ml-2 text-gray-600">/videos/training</span>
              </div>
              <div>
                <span className="font-medium">Videos:</span>
                <span className="ml-2 text-gray-600">125 files</span>
              </div>
              <div>
                <span className="font-medium">Total Size:</span>
                <span className="ml-2 text-gray-600">2.3 GB</span>
              </div>
            </div>
            <div className="mt-4 flex gap-2">
              <Button size="sm" variant="outline" disabled>Scan</Button>
              <Button size="sm" variant="outline" disabled>Process</Button>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Future Features */}
      <Card>
        <CardHeader>
          <CardTitle>Planned Features</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex items-start gap-3">
              <div className="w-2 h-2 rounded-full bg-blue-500 mt-2"></div>
              <div>
                <h4 className="font-medium">Dataset Registration</h4>
                <p className="text-sm text-gray-600">
                  Register video directories with metadata for organized batch processing
                </p>
              </div>
            </div>
            
            <div className="flex items-start gap-3">
              <div className="w-2 h-2 rounded-full bg-blue-500 mt-2"></div>
              <div>
                <h4 className="font-medium">Video Scanning</h4>
                <p className="text-sm text-gray-600">
                  Automatically discover and catalog video files in registered datasets
                </p>
              </div>
            </div>
            
            <div className="flex items-start gap-3">
              <div className="w-2 h-2 rounded-full bg-blue-500 mt-2"></div>
              <div>
                <h4 className="font-medium">Batch Processing</h4>
                <p className="text-sm text-gray-600">
                  Process entire datasets with consistent pipeline configurations
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <div className="w-2 h-2 rounded-full bg-blue-500 mt-2"></div>
              <div>
                <h4 className="font-medium">Progress Tracking</h4>
                <p className="text-sm text-gray-600">
                  Monitor batch job progress across multiple videos simultaneously
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default CreateDatasets;