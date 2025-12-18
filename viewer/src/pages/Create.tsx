import { Outlet } from "react-router-dom";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Link, useLocation } from "react-router-dom";
import { Plus, Database, List, Settings, ArrowLeft } from "lucide-react";
import { Footer } from "@/components/Footer";
import { TokenStatusIndicator } from "@/components/TokenStatusIndicator";
import { ConnectionErrorBanner } from "@/components/ConnectionErrorBanner";
import { useServerCapabilitiesContext } from "@/contexts/ServerCapabilitiesContext";
import vavIcon from "@/assets/v-a-v.icon.png";

/**
 * Detects if error is likely a CORS or network connectivity issue
 * Auth errors (401, 403, 404) should not be treated as connection issues
 */
function isCorsOrNetworkError(error: Error): boolean {
  const message = error.message.toLowerCase();

  // Check if it's an auth/permission error - these should NOT show the connection banner
  const isAuthError =
    message.includes('401') ||
    message.includes('403') ||
    message.includes('404') ||
    message.includes('unauthorized') ||
    message.includes('forbidden') ||
    message.includes('not found') ||
    message.includes('auth') ||
    message.includes('permission');

  if (isAuthError) {
    console.log('[Create] Auth error detected, not showing connection banner:', message);
    return false;
  }

  // Now check for actual connection issues
  const isConnectionError =
    message.includes('cors') ||
    message.includes('fetch') ||
    message.includes('network') ||
    message.includes('failed to fetch') ||
    message.includes('networkerror') ||
    message.includes('timeout') ||
    message.includes('timed out') ||
    message.includes('access-control-allow-origin');

  console.log('[Create] Connection error check:', { message, isConnectionError });
  return isConnectionError;
}

const CreateLayout = () => {
  const location = useLocation();
  const { error, refresh, capabilities, isLoading } = useServerCapabilitiesContext();

  // Capabilities are auto-fetched by the provider after a short delay
  // This avoids React StrictMode double-mount issues

  const navigationItems = [
    { path: "/create/jobs", label: "Jobs", icon: List },
    { path: "/create/new", label: "New Job", icon: Plus },
    { path: "/create/datasets", label: "Datasets", icon: Database },
    { path: "/create/settings", label: "Settings", icon: Settings },
  ];

  const apiUrl = localStorage.getItem('videoannotator_api_url') ||
    import.meta.env.VITE_API_BASE_URL ||
    '';

  // Only show connection error if we have an actual connection problem AND no successful capabilities
  // If we have capabilities, the server is reachable even if health check had issues
  const showConnectionError = error && !isLoading && !capabilities && isCorsOrNetworkError(error);

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <div className="flex-1">
        <div className="container mx-auto px-4 py-6">
          {/* Header */}
          <div className="mb-6">
            <div className="flex items-center gap-4 mb-2">
              <Link to="/viewer">
                <Button variant="outline" size="sm">
                  <ArrowLeft className="h-4 w-4 mr-2" />
                  Back to Viewer
                </Button>
              </Link>
              <div className="flex items-center gap-3">
                <img src={vavIcon} alt="VideoAnnotator" className="w-8 h-8" />
                <h1 className="text-3xl font-bold text-gray-900">
                  VideoAnnotator Control Panel
                </h1>
              </div>
            </div>
            <p className="text-gray-700 ml-16">
              Create and manage annotation jobs using the VideoAnnotator pipeline
            </p>
          </div>

          {/* Navigation */}
          <div className="mb-6">
            <div className="flex items-center justify-between">
              <nav className="flex space-x-2">
                {navigationItems.map(({ path, label, icon: Icon }) => (
                  <Link key={path} to={path}>
                    <Button
                      variant={location.pathname === path ? "default" : "outline"}
                      className="flex items-center gap-2"
                    >
                      <Icon className="h-4 w-4" />
                      {label}
                    </Button>
                  </Link>
                ))}
              </nav>
              <TokenStatusIndicator showDetails={true} />
            </div>
          </div>

          {/* Connection Error Banner - only show if truly unreachable */}
          {showConnectionError && (
            <div className="mb-6">
              <ConnectionErrorBanner
                error={error}
                apiUrl={apiUrl}
                onRetry={refresh}
              />
            </div>
          )}

          {/* Content */}
          <Card className="min-h-[600px] p-6">
            <Outlet />
          </Card>
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default CreateLayout;