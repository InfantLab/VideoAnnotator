import { useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  Popover,
  PopoverContent,
  PopoverTrigger
} from '@/components/ui/popover';
import {
  CheckCircle,
  AlertTriangle,
  Settings,
  RefreshCw,
  ExternalLink,
  Shield,
  ShieldAlert,
  ShieldOff,
  Loader2
} from 'lucide-react';
import { Link } from 'react-router-dom';
import { useTokenStatus } from '@/hooks/useTokenStatus';
import { useServerCapabilitiesContext } from '@/contexts/ServerCapabilitiesContext';

interface TokenStatusIndicatorProps {
  showDetails?: boolean;
  className?: string;
  compact?: boolean;
}

export function TokenStatusIndicator({
  showDetails = false,
  className = '',
  compact = false
}: TokenStatusIndicatorProps) {
  const { isValid, user, error, isLoading, refresh: refreshToken } = useTokenStatus();
  const { capabilities, error: capsError, isLoading: capsLoading, refresh: refreshCaps } = useServerCapabilitiesContext();
  const [isOpen, setIsOpen] = useState(false);

  const loading = isLoading || capsLoading;
  const serverVersion = capabilities?.version || 'unknown';

  // We're connected if we successfully fetched capabilities
  const isConnected = !!capabilities;

  // Determine authentication requirement status
  // Note: v1.3.0 doesn't expose auth_required in health endpoint
  // We infer from connection attempts: if we get 401 without token, auth is required
  const authStatus = error?.includes('401') || error?.includes('Unauthorized')
    ? 'required'
    : isValid
      ? 'optional'
      : undefined;

  if (loading && !capabilities) {
    return (
      <Badge variant="secondary" className={className}>
        <Loader2 className="h-3 w-3 mr-1 animate-spin" />
        {compact ? 'Checking...' : 'Checking connection...'}
      </Badge>
    );
  }

  const statusBadge = (
    <Badge
      variant={isConnected ? "default" : "destructive"}
      className={`cursor-pointer ${className}`}
    >
      {isConnected ? (
        <CheckCircle className="h-3 w-3 mr-1" />
      ) : (
        <AlertTriangle className="h-3 w-3 mr-1" />
      )}
      {isConnected ? 'Connected' : 'Error'}
      {isConnected && !compact && serverVersion !== 'unknown' && (
        <span className="ml-1 text-xs opacity-80">v{serverVersion}</span>
      )}
    </Badge>
  );

  if (!showDetails) {
    return statusBadge;
  }

  return (
    <Popover open={isOpen} onOpenChange={setIsOpen}>
      <PopoverTrigger asChild>
        {statusBadge}
      </PopoverTrigger>
      <PopoverContent className="w-80" align="end">
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="font-medium">API Status</h4>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                refreshToken();
                refreshCaps();
              }}
              disabled={loading}
            >
              <RefreshCw className={`h-3 w-3 ${loading ? 'animate-spin' : ''}`} />
            </Button>
          </div>

          {isConnected ? (
            <Alert>
              <CheckCircle className="h-4 w-4 text-green-600" />
              <AlertDescription>
                <div className="space-y-1">
                  <p className="font-medium text-green-700">✅ Connected to VideoAnnotator</p>
                  {user && <p className="text-sm">Authenticated as: {user}</p>}
                  {serverVersion !== 'unknown' && (
                    <p className="text-sm text-muted-foreground">Server version: {serverVersion}</p>
                  )}
                </div>
              </AlertDescription>
            </Alert>
          ) : (
            <Alert variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>
                <div className="space-y-2">
                  <p className="font-medium">❌ Connection Issue</p>
                  <p className="text-sm">{capsError?.message || error || 'Unable to connect to server'}</p>
                  {error?.includes('401') || error?.includes('Unauthorized') || capsError?.message?.includes('401') || capsError?.message?.includes('Unauthorized') ? (
                    <p className="text-xs text-muted-foreground mt-2">
                      💡 Hint: Check your API token in Settings
                    </p>
                  ) : error?.includes('fetch') || error?.includes('network') || capsError?.message?.includes('fetch') || capsError?.message?.includes('network') ? (
                    <p className="text-xs text-muted-foreground mt-2">
                      💡 Hint: Verify server is running at the configured URL
                    </p>
                  ) : null}
                </div>
              </AlertDescription>
            </Alert>
          )}

          {/* Authentication Mode Indicator (T039) */}
          {authStatus && (
            <div className="flex items-center gap-2 text-sm px-3 py-2 rounded-md bg-muted">
              {authStatus === 'required' ? (
                <>
                  <Shield className="h-4 w-4 text-blue-600" />
                  <span>Authentication Required</span>
                </>
              ) : authStatus === 'optional' ? (
                <>
                  <ShieldAlert className="h-4 w-4 text-yellow-600" />
                  <span>Authentication Optional</span>
                </>
              ) : null}
            </div>
          )}

          {/* Unsecured Connection Warning (T041) */}
          {isValid && authStatus === 'optional' && (
            <Alert variant="default" className="border-yellow-200 bg-yellow-50">
              <ShieldOff className="h-4 w-4 text-yellow-600" />
              <AlertDescription>
                <p className="text-sm text-yellow-800">
                  ⚠️ Server is running without authentication
                </p>
                <p className="text-xs text-yellow-700 mt-1">
                  Consider enabling VIDEOANNOTATOR_REQUIRE_AUTH in production
                </p>
              </AlertDescription>
            </Alert>
          )}

          <div className="flex items-center gap-2 pt-2 border-t">
            <Link to="/create/settings">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setIsOpen(false)}
                className="flex items-center gap-1"
              >
                <Settings className="h-3 w-3" />
                Settings
              </Button>
            </Link>

            <Button
              variant="link"
              size="sm"
              onClick={() => setIsOpen(false)}
              asChild
              className="p-0 h-auto"
            >
              <a
                href="https://github.com/InfantLab/VideoAnnotator/blob/master/docs/CLIENT_TOKEN_SETUP_GUIDE.md"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1 text-xs"
              >
                Help <ExternalLink className="h-3 w-3" />
              </a>
            </Button>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}