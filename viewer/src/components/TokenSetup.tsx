import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { CheckCircle, AlertCircle, Settings, ExternalLink, Eye, EyeOff, Rocket, HelpCircle, Stethoscope } from 'lucide-react';
import { apiClient } from '@/api/client';
import { handleAPIError } from '@/api/handleError';
import { runConnectionDiagnostics, formatDiagnosticReport, type DiagnosticReport } from '@/lib/connectionDiagnostics';

interface TokenSetupProps {
  onTokenConfigured?: () => void;
}

interface TokenStatus {
  isValid: boolean;
  user?: string;
  permissions?: string[];
  expiresAt?: string;
  error?: string;
}

export function TokenSetup({ onTokenConfigured }: TokenSetupProps) {
  const [apiUrl, setApiUrl] = useState(() => {
    const saved = localStorage.getItem('videoannotator_api_url');
    if (saved !== null) {
      // Auto-fix existing localhost config to bypass broken DNS
      if (saved.includes('//localhost:')) {
        return saved.replace('//localhost:', '//127.0.0.1:');
      }
      return saved;
    }
    // Default to 127.0.0.1 instead of localhost
    const envUrl = import.meta.env.VITE_API_BASE_URL;
    if (envUrl && envUrl.includes('//localhost:')) {
      return envUrl.replace('//localhost:', '//127.0.0.1:');
    }
    return envUrl || 'http://127.0.0.1:18011';
  });
  const [token, setToken] = useState(() => {
    const saved = localStorage.getItem('videoannotator_api_token');
    if (saved !== null) return saved;
    return import.meta.env.VITE_API_TOKEN || '';
  });
  const [showToken, setShowToken] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [tokenStatus, setTokenStatus] = useState<TokenStatus | null>(null);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [serverAuthRequired, setServerAuthRequired] = useState<boolean | null>(null);
  const [diagnosticReport, setDiagnosticReport] = useState<DiagnosticReport | null>(null);
  const [isRunningDiagnostics, setIsRunningDiagnostics] = useState(false);

  // Check token validity - Uses the robust client validation
  const validateToken = async (testUrl?: string, testToken?: string) => {
    const urlToTest = testUrl || apiUrl;
    const tokenToTest = testToken !== undefined ? testToken : token;

    setIsValidating(true);

    // Update API client temporarily for this test
    const originalUrl = apiClient.baseURL;
    const originalToken = apiClient.token;
    apiClient.updateConfig(urlToTest, tokenToTest || undefined);

    try {
      // Use the robust validation from API client
      // This checks both health AND permissions (by hitting /jobs)
      const result = await apiClient.validateToken();

      setTokenStatus({
        isValid: result.isValid,
        user: result.user,
        permissions: result.permissions,
        expiresAt: result.expiresAt,
        error: result.error
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
    // Auto-fix localhost to 127.0.0.1 before saving
    let finalUrl = apiUrl;
    if (finalUrl.includes('//localhost:')) {
      finalUrl = finalUrl.replace('//localhost:', '//127.0.0.1:');
      setApiUrl(finalUrl); // Update UI state too
    }

    localStorage.setItem('videoannotator_api_url', finalUrl);
    localStorage.setItem('videoannotator_api_token', token);
    setHasUnsavedChanges(false);

    // Update the global API client
    apiClient.updateConfig(finalUrl, token.trim() ? token : undefined);

    // Show success message
    const tokenMode = token.trim() ? 'with API token' : 'in anonymous mode';
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
    const savedUrl = localStorage.getItem('videoannotator_api_url');
    const savedToken = localStorage.getItem('videoannotator_api_token');

    if (savedUrl !== null) {
      // Auto-fix existing localhost config to bypass broken DNS
      if (savedUrl.includes('//localhost:')) {
        setApiUrl(savedUrl.replace('//localhost:', '//127.0.0.1:'));
      } else {
        setApiUrl(savedUrl);
      }
    }
    
    if (savedToken !== null) setToken(savedToken);

    // Don't auto-validate - it makes the page slow and isn't essential
    // User can click "Test Connection" if they want to validate
  }, []);

  // Track changes and clear stale validation status
  useEffect(() => {
    const savedUrl = localStorage.getItem('videoannotator_api_url');
    const savedToken = localStorage.getItem('videoannotator_api_token');

    // Compare with saved values (handling nulls)
    const currentSavedUrl = savedUrl !== null ? savedUrl : (import.meta.env.VITE_API_BASE_URL || 'http://localhost:18011');
    const currentSavedToken = savedToken !== null ? savedToken : (import.meta.env.VITE_API_TOKEN || '');

    const hasChanges = apiUrl !== currentSavedUrl || token !== currentSavedToken;
    setHasUnsavedChanges(hasChanges);

    // Clear validation status when token/url changes to avoid showing stale results
    if (hasChanges) {
      setTokenStatus(null);
    }
  }, [apiUrl, token]);

  const resetToDefaults = () => {
    // Reset to environment variable defaults
    const defaultUrl = import.meta.env.VITE_API_BASE_URL || '';
    const defaultToken = import.meta.env.VITE_API_TOKEN || ''; // Empty = anonymous

    // Save defaults to localStorage immediately
    if (defaultUrl) {
        localStorage.setItem('videoannotator_api_url', defaultUrl);
    } else {
        localStorage.removeItem('videoannotator_api_url');
    }
    
    if (defaultToken) {
        localStorage.setItem('videoannotator_api_token', defaultToken);
    } else {
        localStorage.removeItem('videoannotator_api_token');
    }

    setApiUrl(defaultUrl);
    setToken(defaultToken);
    setHasUnsavedChanges(false); // Already saved

    // Update the global API client
    apiClient.updateConfig(defaultUrl, defaultToken || undefined);

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
  const isFirstTimeUser = !localStorage.getItem('videoannotator_api_token') && !tokenStatus;

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      {/* First-Time User Guide (T040) */}
      {isFirstTimeUser && (
        <Alert className="border-blue-200 bg-blue-50">
          <Rocket className="h-5 w-5 text-blue-600" />
          <AlertTitle className="text-blue-900 font-semibold">Welcome! Let's get you connected</AlertTitle>
          <AlertDescription className="text-blue-800 space-y-3 mt-2">
            <p>
              To start creating annotation jobs, you need to configure your VideoAnnotator server connection.
            </p>
            <div className="space-y-2 text-sm">
              <p className="font-medium">Quick Start:</p>
              <ol className="list-decimal list-inside space-y-1 ml-2">
                <li>Make sure your VideoAnnotator server is running (default: http://localhost:18011)</li>
                <li>Get your API token from the server console or administrator</li>
                <li>Enter the token below and click "Test Connection"</li>
                <li>Once validated, click "Save Configuration"</li>
              </ol>
            </div>
            <div className="flex items-start gap-2 mt-3 p-3 bg-white/50 rounded-md border border-blue-200">
              <HelpCircle className="h-4 w-4 text-blue-600 mt-0.5 flex-shrink-0" />
              <div className="text-sm">
                <p className="font-medium mb-1">For local development:</p>
                <p>
                  Use the default values below. Click <strong>"Reset to Defaults"</strong> to auto-fill,
                  then <strong>"Test Connection"</strong> to verify.
                </p>
              </div>
            </div>
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
                  placeholder="http://localhost:18011"
                  className={!apiUrl ? "pl-24" : ""}
                />
                {!apiUrl && (
                  <div className="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none">
                    <Badge variant="secondary" className="h-6 bg-green-100 text-green-800 hover:bg-green-100">
                      PROXY MODE
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
                ? "Using built-in proxy to bypass CORS (Recommended for local dev)" 
                : "The base URL of your VideoAnnotator API server (e.g., http://localhost:18011)"}
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
                      ? "va_xxxxxxxxxxxx (required by server)"
                      : "va_xxxxxxxxxxxx or leave empty"
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
            {token === 'dev-token' && (
              <Alert className="mt-2 bg-yellow-50 border-yellow-200">
                <AlertCircle className="h-4 w-4 text-yellow-600" />
                <AlertTitle className="text-yellow-900">Invalid Token Configuration</AlertTitle>
                <AlertDescription className="text-yellow-800 space-y-3">
                  <p>"dev-token" is a placeholder. You need to choose how to authenticate:</p>

                  <div className="space-y-3">
                    {/* Option 1: Anonymous */}
                    <div className="bg-white rounded-md p-3 border border-yellow-300">
                      <div className="flex items-start gap-2">
                        <div className="flex-1">
                          <p className="font-semibold text-yellow-900 mb-1">Option 1: No Authentication (Local Testing)</p>
                          <p className="text-xs mb-2">
                            {serverAuthRequired === null && "Click 'Test Connection' to check if your server requires authentication."}
                            {serverAuthRequired === false && "✅ Your server does not require authentication. Clear this field to connect."}
                            {serverAuthRequired === true && "❌ Your server requires authentication. Use Option 2 instead."}
                          </p>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => {
                              setToken('');
                              setHasUnsavedChanges(true);
                            }}
                            className="text-xs"
                            disabled={serverAuthRequired === true}
                          >
                            Clear Token Field
                          </Button>
                        </div>
                      </div>
                    </div>

                    {/* Option 2: Real Token */}
                    <div className="bg-white rounded-md p-3 border border-yellow-300">
                      <div className="flex items-start gap-2">
                        <div className="flex-1">
                          <p className="font-semibold text-yellow-900 mb-1">Option 2: Use API Token (Production/Secure)</p>
                          <p className="text-xs mb-2">Get a token from your VideoAnnotator server. Tokens start with "va_" or are JWT format (starts with "eyJ").</p>
                          <details className="text-xs">
                            <summary className="cursor-pointer text-yellow-700 hover:text-yellow-900 font-medium">
                              How to get a token
                            </summary>
                            <div className="mt-2 space-y-1 text-yellow-800 bg-yellow-50 p-2 rounded">
                              <p>• <strong>From server admin:</strong> Ask your VideoAnnotator admin for an API key</p>
                              <p>• <strong>Generate yourself:</strong> In the <em>backend</em> folder, run: <code className="bg-yellow-100 px-1 rounded">uv run videoannotator generate-token</code></p>
                              <p>• <strong>Token format:</strong> Should look like <code className="bg-yellow-100 px-1 rounded">va_abc123xyz...</code></p>
                            </div>
                          </details>
                        </div>
                      </div>
                    </div>
                  </div>
                </AlertDescription>
              </Alert>
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
                  <div className="bg-white/60 border border-yellow-300 rounded p-2">
                    <p className="text-xs font-semibold text-yellow-900 mb-2">How to get an API token:</p>
                    <div className="space-y-1 text-xs text-yellow-800">
                      <p>• <strong>From server admin:</strong> Ask your VideoAnnotator admin for an API key</p>
                      <p>• <strong>Generate yourself:</strong> In the <em>backend</em> folder, run: <code className="bg-yellow-100 px-1 rounded">uv run videoannotator generate-token</code></p>
                      <p>• <strong>Token format:</strong> Should look like <code className="bg-yellow-100 px-1 rounded">va_abc123xyz...</code></p>
                    </div>
                  </div>
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
                  <div className="bg-white/60 border border-yellow-300 rounded p-2">
                    <p className="text-xs font-semibold text-yellow-900 mb-2">How to get a valid API token:</p>
                    <div className="space-y-1 text-xs text-yellow-800">
                      <p>• <strong>From server admin:</strong> Ask your VideoAnnotator admin for an API key</p>
                      <p>• <strong>Generate yourself:</strong> In the <em>backend</em> folder, run: <code className="bg-yellow-100 px-1 rounded">uv run videoannotator generate-token</code></p>
                      <p>• <strong>Token format:</strong> Should look like <code className="bg-yellow-100 px-1 rounded">va_abc123xyz...</code></p>
                    </div>
                  </div>
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
                        {tokenStatus.permissions && tokenStatus.permissions.length > 0 && (
                          <div>Permissions: {tokenStatus.permissions.map(p =>
                            <Badge key={p} variant="outline" className="mr-1">{p}</Badge>
                          )}</div>
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
                        {!token && serverAuthRequired === true && (
                          <div className="bg-red-50 border border-red-200 rounded-md p-3 mt-2">
                            <p className="text-sm font-semibold text-red-900 mb-2">How to get an API token:</p>
                            <div className="space-y-1 text-xs text-red-800">
                              <p>• <strong>From server admin:</strong> Ask your VideoAnnotator admin for an API key</p>
                              <p>• <strong>Generate yourself:</strong> In the <em>backend</em> folder, run: <code className="bg-red-100 px-1 rounded">uv run videoannotator generate-token</code></p>
                              <p>• <strong>Token format:</strong> Should look like <code className="bg-red-100 px-1 rounded">va_abc123xyz...</code></p>
                            </div>
                          </div>
                        )}
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
              title={`Reset to: URL=${import.meta.env.VITE_API_BASE_URL || 'http://localhost:18011'}, Token=${import.meta.env.VITE_API_TOKEN || '(empty - anonymous)'}`}
            >
              Reset to Defaults
            </Button>

            <Button
              onClick={saveConfiguration}
              disabled={!apiUrl.trim()}
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
            <h4 className="font-medium">Need help getting your token?</h4>
            <div className="text-sm text-muted-foreground space-y-2">
              <p>1. Contact your VideoAnnotator server administrator</p>
              <p>2. Or check the server documentation for token generation</p>
              <p>3. For development, use the default token: <code className="bg-muted px-1 rounded">dev-token</code></p>
            </div>

            <Button variant="link" size="sm" className="pl-0" asChild>
              <a
                href="https://github.com/InfantLab/VideoAnnotator/blob/master/docs/CLIENT_TOKEN_SETUP_GUIDE.md"
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