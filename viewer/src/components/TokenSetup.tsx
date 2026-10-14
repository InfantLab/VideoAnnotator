import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { CheckCircle, AlertCircle, Settings, ExternalLink, Eye, EyeOff, Rocket, Stethoscope } from 'lucide-react';
import { apiClient } from '@/api/client';
import { handleAPIError } from '@/api/handleError';
import { runConnectionDiagnostics, formatDiagnosticReport, type DiagnosticReport } from '@/lib/connectionDiagnostics';
import {
  API_TOKEN_STORAGE_KEY,
  API_URL_STORAGE_KEY,
  defaultApiUrl,
  describeApiUrl,
  normalizeApiUrl,
  servedWithApi,
  tokenFormatProblem
} from '@/lib/apiConnection';
import { TokenHelp } from '@/components/TokenHelp';

interface TokenSetupProps {
  onTokenConfigured?: () => void;
}

interface TokenStatus {
  isValid: boolean;
  user?: string;
  permissions?: string[];
  expiresAt?: string;
  error?: string;
  /** From GET /api/v1/auth/me, fetched alongside validation. `undefined` when the
   * server predates that endpoint (404) - not the same as a known non-admin (`false`). */
  isAdmin?: boolean;
  username?: string;
}

export function TokenSetup({ onTokenConfigured }: TokenSetupProps) {
  const [apiUrl, setApiUrl] = useState(() => localStorage.getItem(API_URL_STORAGE_KEY) ?? defaultApiUrl());
  const [token, setToken] = useState(() => {
    const saved = localStorage.getItem(API_TOKEN_STORAGE_KEY);
    if (saved !== null) return saved;
    return import.meta.env.VITE_API_TOKEN || '';
  });
  const tokenProblem = tokenFormatProblem(token);
  const [showToken, setShowToken] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [tokenStatus, setTokenStatus] = useState<TokenStatus | null>(null);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [serverAuthRequired, setServerAuthRequired] = useState<boolean | null>(null);
  const [diagnosticReport, setDiagnosticReport] = useState<DiagnosticReport | null>(null);
  const [isRunningDiagnostics, setIsRunningDiagnostics] = useState(false);

  // Check token validity - Uses the robust client validation
  const validateToken = async (testUrl?: string, testToken?: string) => {
    const urlToTest = testUrl ?? apiUrl;
    const tokenToTest = testToken !== undefined ? testToken : token;

    setIsValidating(true);

    // Update API client temporarily for this test
    const originalUrl = apiClient.baseURL;
    const originalToken = apiClient.token;
    apiClient.updateConfig(urlToTest, tokenToTest);

    try {
      // Use the robust validation from API client
      // This checks both health AND permissions (by hitting /jobs)
      const result = await apiClient.validateToken();

      // Fetch admin status alongside validation (GET /api/v1/auth/me). Best-effort:
      // a 404 just means an older server that predates this endpoint, a 401 means
      // the token isn't actually authenticated - neither should block showing the
      // rest of the validation result.
      let currentUser: { isAdmin: boolean; username: string } | null = null;
      if (result.isValid && tokenToTest) {
        try {
          const me = await apiClient.getCurrentUser();
          currentUser = { isAdmin: me.isAdmin, username: me.username };
        } catch {
          // Leave isAdmin/username undefined - see TokenStatus.isAdmin doc comment.
        }
      }

      setTokenStatus({
        isValid: result.isValid,
        user: result.user,
        permissions: result.permissions,
        expiresAt: result.expiresAt,
        error: result.error,
        isAdmin: currentUser?.isAdmin,
        username: currentUser?.username
      });

      // Also update auth required status
      try {
        const health = await apiClient.getSystemHealth();
        setServerAuthRequired(health.security?.auth_required ?? null);
      } catch {
        // Ignore health check failure here, validateToken result is what matters
      }

    } catch (error: unknown) {
      setTokenStatus({
        isValid: false,
        error: error instanceof Error ? error.message : String(error)
      });
    } finally {
      // Restore original config
      apiClient.updateConfig(originalUrl, originalToken);
      setIsValidating(false);
    }
  };

  // Save configuration
  const saveConfiguration = () => {
    const finalUrl = normalizeApiUrl(apiUrl);
    setApiUrl(finalUrl);

    localStorage.setItem(API_URL_STORAGE_KEY, finalUrl);
    localStorage.setItem(API_TOKEN_STORAGE_KEY, token);
    setHasUnsavedChanges(false);

    apiClient.updateConfig(finalUrl, token);

    // Show success message
    console.log(`✅ Configuration saved: URL=${finalUrl}, Token=${token ? '***' : '(empty)'}`);

    // Note: The validation error (if any) is separate from saving the config
    // The config is saved successfully even if the server requires auth

    onTokenConfigured?.();

    // Force a page refresh to ensure all components pick up the new token
    // This is necessary because some queries might be cached
    console.log('🔄 Reloading page to apply new token configuration...');
    setTimeout(() => {
      window.location.reload();
    }, 500);
  };

  // Load saved configuration on mount
  // Don't auto-validate - let user click "Test Connection" button
  useEffect(() => {
    const savedUrl = localStorage.getItem(API_URL_STORAGE_KEY);
    const savedToken = localStorage.getItem(API_TOKEN_STORAGE_KEY);

    if (savedUrl !== null) setApiUrl(savedUrl);
    
    if (savedToken !== null) setToken(savedToken);

    // Don't auto-validate - it makes the page slow and isn't essential
    // User can click "Test Connection" if they want to validate
  }, []);

  // Track changes and clear stale validation status
  useEffect(() => {
    const savedUrl = localStorage.getItem(API_URL_STORAGE_KEY);
    const savedToken = localStorage.getItem(API_TOKEN_STORAGE_KEY);

    // Compare with saved values (handling nulls)
    const currentSavedUrl = savedUrl ?? defaultApiUrl();
    const currentSavedToken = savedToken !== null ? savedToken : (import.meta.env.VITE_API_TOKEN || '');

    const hasChanges = apiUrl !== currentSavedUrl || token !== currentSavedToken;
    setHasUnsavedChanges(hasChanges);

    // Clear validation status when token/url changes to avoid showing stale results
    if (hasChanges) {
      setTokenStatus(null);
    }
  }, [apiUrl, token]);

  const resetToDefaults = () => {
    const defaultUrl = defaultApiUrl();
    const defaultToken = import.meta.env.VITE_API_TOKEN || ''; // Empty = anonymous

    localStorage.removeItem(API_URL_STORAGE_KEY);
    if (defaultToken) {
        localStorage.setItem(API_TOKEN_STORAGE_KEY, defaultToken);
    } else {
        localStorage.removeItem(API_TOKEN_STORAGE_KEY);
    }

    setApiUrl(defaultUrl);
    setToken(defaultToken);
    setHasUnsavedChanges(false); // Already saved

    // Update the global API client
    apiClient.updateConfig(defaultUrl, defaultToken);

    // Clear any previous token status
    setTokenStatus(null);

    // Auto-test the defaults
    setTimeout(() => {
      validateToken(defaultUrl, defaultToken);
    }, 100);
  };

  const handleTestConnection = () => {
    validateToken();
  };

  const handleRunDiagnostics = async () => {
    setIsRunningDiagnostics(true);
    setDiagnosticReport(null);

    try {
      const report = await runConnectionDiagnostics(apiUrl, token || undefined);
      setDiagnosticReport(report);

      // Auto-copy to clipboard for easy sharing
      const formatted = formatDiagnosticReport(report);
      navigator.clipboard.writeText(formatted).catch(() => {
        // Clipboard write failed, ignore
      });
    } catch (error) {
      console.error('Diagnostics failed:', error);
    } finally {
      setIsRunningDiagnostics(false);
    }
  };

  // Check if this is a first-time user (no saved token)
  const isFirstTimeUser = !localStorage.getItem(API_TOKEN_STORAGE_KEY) && !tokenStatus;

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      {/* First-Time User Guide (T040) */}
      {isFirstTimeUser && (
        <Alert className="border-blue-200 bg-blue-50">
          <Rocket className="h-5 w-5 text-blue-600" />
          <AlertTitle className="text-blue-900 font-semibold">Welcome! Let's get you connected</AlertTitle>
          <AlertDescription className="text-blue-800 space-y-3 mt-2">
            <p>
              To create annotation jobs, connect this viewer to your VideoAnnotator server with an API key.
              The quickest way is the one-click link the server prints the first time it starts; see{' '}
              <a href="#how-to-get-a-key" className="font-medium underline">How to get a key</a> below.
            </p>
          </AlertDescription>
        </Alert>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            VideoAnnotator API Configuration
          </CardTitle>
          <CardDescription>
            Configure your VideoAnnotator server connection and authentication token.
            This is required to create and manage annotation jobs.
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-6">
          {/* Server URL Configuration */}
          <div className="space-y-2">
            <Label htmlFor="api-url">Server URL</Label>
            <div className="flex gap-2">
              <div className="relative flex-1">
                <Input
                  id="api-url"
                  value={apiUrl}
                  onChange={(e) => setApiUrl(e.target.value)}
                  placeholder={describeApiUrl('')}
                  className={!apiUrl ? "pl-24" : ""}
                />
                {!apiUrl && (
                  <div className="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none">
                    <Badge variant="secondary" className="h-6 bg-green-100 text-green-800 hover:bg-green-100">
                      {import.meta.env.DEV ? 'PROXY MODE' : 'THIS SERVER'}
                    </Badge>
                  </div>
                )}
              </div>
              {import.meta.env.DEV && (
                <Button
                  variant={!apiUrl ? "default" : "outline"}
                  onClick={() => setApiUrl('')}
                  title="Use local proxy (avoids CORS issues)"
                  type="button"
                  className={!apiUrl ? "bg-green-600 hover:bg-green-700" : ""}
                >
                  {!apiUrl ? "Using Proxy" : "Use Proxy"}
                </Button>
              )}
            </div>
            <p className="text-sm text-muted-foreground">
              {!apiUrl
                ? (import.meta.env.DEV
                  ? "Using the dev server's proxy to the API (recommended for local development)"
                  : `Using the server that served this page (${describeApiUrl('')})`)
                : "The base URL of your VideoAnnotator API server, e.g. http://127.0.0.1:18011"}
            </p>
          </div>

          {/* Token Configuration */}
          <div className="space-y-2">
            <Label htmlFor="api-token">
              API Token
              {serverAuthRequired === false && <span className="text-muted-foreground"> (Optional)</span>}
              {serverAuthRequired === true && <span className="text-destructive"> (Required)</span>}
              {serverAuthRequired === null && <span className="text-muted-foreground"> (Click Test Connection to check)</span>}
            </Label>
            <div className="relative">
              <Input
                id="api-token"
                type={showToken ? "text" : "password"}
                value={token}
                onChange={(e) => setToken(e.target.value)}
                placeholder={
                  serverAuthRequired === false
                    ? "Leave empty for anonymous access"
                    : serverAuthRequired === true
                      ? "va_ followed by 43 characters (required by this server)"
                      : "va_ followed by 43 characters, or empty if authentication is off"
                }
                className="pr-10"
              />
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="absolute right-0 top-0 h-full px-3"
                onClick={() => setShowToken(!showToken)}
              >
                {showToken ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </Button>
            </div>
            {tokenProblem && (
              <p className="text-sm text-destructive" role="alert">{tokenProblem}</p>
            )}
            {!token && serverAuthRequired === false && (
              <div className="bg-green-50 border border-green-200 rounded-md p-3">
                <p className="text-sm text-green-800">
                  ✅ <strong>No token set</strong> - You'll connect anonymously (your server doesn't require authentication)
                </p>
              </div>
            )}
            {!token && serverAuthRequired === true && (
              <div className="bg-yellow-50 border border-yellow-200 rounded-md p-3">
                <div className="space-y-3">
                  <p className="text-sm text-yellow-800 font-medium">
                    ⚠️ <strong>No token set</strong> - Your server requires authentication.
                  </p>
                  <p className="text-xs text-yellow-800">
                    See <a href="#how-to-get-a-key" className="font-medium underline">How to get a key</a> below.
                  </p>
                </div>
              </div>
            )}
            {!token && serverAuthRequired === null && (
              <div className="bg-blue-50 border border-blue-200 rounded-md p-3">
                <p className="text-sm text-blue-800">
                  💡 <strong>No token set</strong> - Click "Test Connection" to check if your server requires authentication.
                </p>
              </div>
            )}
            {token.trim() && tokenStatus && !tokenStatus.isValid && serverAuthRequired === true && (
              <div className="bg-yellow-50 border border-yellow-200 rounded-md p-3 mt-2">
                <div className="space-y-3">
                  <p className="text-sm text-yellow-800 font-medium">
                    💡 <strong>Token invalid or authentication failed</strong>
                  </p>
                  <p className="text-xs text-yellow-800">
                    See <a href="#how-to-get-a-key" className="font-medium underline">How to get a key</a> below.
                  </p>
                </div>
              </div>
            )}
            <p className="text-xs text-muted-foreground">
              💡 Click "Test Connection" to verify your configuration
            </p>
          </div>

          <Separator />

          {/* Token Status */}
          {tokenStatus && (
            <Alert>
              <div className="flex items-start gap-2">
                {tokenStatus.isValid ? (
                  <CheckCircle className="h-4 w-4 text-green-600 mt-0.5" />
                ) : (
                  <AlertCircle className="h-4 w-4 text-red-600 mt-0.5" />
                )}
                <div className="flex-1">
                  <AlertDescription>
                    {tokenStatus.isValid ? (
                      <div className="space-y-2">
                        <p className="font-medium text-green-700">✅ Token is valid and working!</p>
                        {tokenStatus.user && (
                          <div>Authenticated as: <Badge variant="secondary">{tokenStatus.user}</Badge></div>
                        )}
                        {tokenStatus.username && (
                          <div>Server identity: <Badge variant="secondary">{tokenStatus.username}</Badge></div>
                        )}
                        {tokenStatus.isAdmin !== undefined && (
                          <div>
                            Admin access:{' '}
                            <Badge variant={tokenStatus.isAdmin ? 'secondary' : 'outline'}>
                              {tokenStatus.isAdmin ? 'Yes' : 'No'}
                            </Badge>
                            {!tokenStatus.isAdmin && (
                              <span className="ml-1 text-xs text-muted-foreground">
                                (needed for pipeline installs — see below)
                              </span>
                            )}
                          </div>
                        )}
                        {tokenStatus.expiresAt && (
                          <p className="text-sm text-muted-foreground">
                            Expires: {new Date(tokenStatus.expiresAt).toLocaleString()}
                          </p>
                        )}
                        {hasUnsavedChanges && (
                          <div className="bg-green-50 border border-green-200 rounded p-2 mt-2">
                            <p className="text-sm text-green-800 font-medium">
                              ⚠️ <strong>Remember to click "Save Configuration"</strong> below to apply this token!
                            </p>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="space-y-3">
                        <p className="font-medium text-red-700">❌ {tokenStatus.error}</p>
                      </div>
                    )}
                  </AlertDescription>
                </div>
              </div>
            </Alert>
          )}

          {/* Diagnostic Report */}
          {diagnosticReport && (
            <Alert className={diagnosticReport.summary === 'all_passed' ? 'border-green-500' : 'border-yellow-500'}>
              <Stethoscope className="h-4 w-4" />
              <AlertTitle>Connection Diagnostics</AlertTitle>
              <AlertDescription>
                <div className="space-y-3 mt-2">
                  <div className="text-sm">
                    Status: <Badge variant={diagnosticReport.summary === 'all_passed' ? 'default' : 'destructive'}>
                      {diagnosticReport.summary.replace('_', ' ').toUpperCase()}
                    </Badge>
                  </div>

                  <div className="space-y-1">
                    {diagnosticReport.results.map((result, i) => (
                      <div key={i} className="text-sm flex items-start gap-2">
                        <span>{result.passed ? '✅' : '❌'}</span>
                        <span className="flex-1">
                          <strong>{result.test}</strong>
                          {result.duration && <span className="text-muted-foreground"> ({Math.round(result.duration)}ms)</span>}
                          {result.error && <div className="text-red-600 text-xs mt-1">{result.error}</div>}
                        </span>
                      </div>
                    ))}
                  </div>

                  {diagnosticReport.recommendations.length > 0 && (
                    <div className="bg-yellow-50 border border-yellow-200 rounded p-3 mt-2">
                      <p className="text-sm font-semibold text-yellow-900 mb-2">Recommendations:</p>
                      <ul className="text-xs text-yellow-800 space-y-1">
                        {diagnosticReport.recommendations.map((rec, i) => (
                          <li key={i}>• {rec}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  <p className="text-xs text-muted-foreground">
                    Report copied to clipboard. Share with support if needed.
                  </p>
                </div>
              </AlertDescription>
            </Alert>
          )}

          {/* Actions */}
          <div className="flex items-center gap-3 flex-wrap">
            <Button
              onClick={handleTestConnection}
              disabled={isValidating}
              variant="outline"
            >
              {isValidating ? 'Testing...' : 'Test Connection'}
            </Button>

            <Button
              onClick={handleRunDiagnostics}
              disabled={isRunningDiagnostics}
              variant="outline"
              className="text-blue-600 hover:text-blue-700"
            >
              <Stethoscope className="h-4 w-4 mr-2" />
              {isRunningDiagnostics ? 'Running...' : 'Run Diagnostics'}
            </Button>

            <Button
              onClick={resetToDefaults}
              variant="outline"
              className="text-orange-600 hover:text-orange-700"
              title={`Reset to: URL=${describeApiUrl(defaultApiUrl())}, Token=${import.meta.env.VITE_API_TOKEN ? '(from build settings)' : '(empty)'}`}
            >
              Reset to Defaults
            </Button>

            <Button
              onClick={saveConfiguration}
              disabled={Boolean(tokenProblem) || (!apiUrl.trim() && !servedWithApi())}
            >
              Save Configuration
            </Button>

            {hasUnsavedChanges && (
              <Badge variant="secondary">Unsaved changes</Badge>
            )}
          </div>

          <Separator />

          {/* Help Section */}
          <div className="space-y-3">
            <h4 id="how-to-get-a-key" className="font-medium">How to get a key</h4>
            <TokenHelp />

            <Button variant="link" size="sm" className="pl-0" asChild>
              <a
                href="https://github.com/InfantLab/VideoAnnotator/blob/master/docs/usage/CLIENT_TOKEN_SETUP_GUIDE.md"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1"
              >
                View Token Setup Guide <ExternalLink className="h-3 w-3" />
              </a>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}