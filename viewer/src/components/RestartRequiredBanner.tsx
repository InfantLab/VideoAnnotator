import { RefreshCw } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

interface RestartRequiredBannerProps {
  restartRequired: boolean;
}

/**
 * Informs the user that a pipeline-extras install completed but the
 * VideoAnnotator server hasn't restarted to activate it yet - an expected
 * next step in the current workflow, not an error state.
 * See specs/002-pipeline-extras-install (User Story 3, FR-008).
 */
export const RestartRequiredBanner = ({ restartRequired }: RestartRequiredBannerProps) => {
  if (!restartRequired) return null;

  return (
    <Alert>
      <RefreshCw className="h-4 w-4" />
      <AlertTitle>Server restart needed</AlertTitle>
      <AlertDescription>
        A pipeline extras install has finished downloading, but the VideoAnnotator server needs to
        restart to activate it. Newly installed pipelines will keep showing as locked until then -
        restart the server, then refresh this page.
      </AlertDescription>
    </Alert>
  );
};

export default RestartRequiredBanner;
