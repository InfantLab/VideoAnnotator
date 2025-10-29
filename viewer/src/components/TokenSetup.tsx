import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { CheckCircle, AlertCircle, Settings, ExternalLink, Eye, EyeOff, Rocket, HelpCircle } from 'lucide-react';
import { apiClient } from '@/api/client';
import { handleAPIError } from '@/api/handleError';

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
  const [apiUrl, setApiUrl] = useState(
    localStorage.getItem('videoannotator_api_url') ||
    import.meta.env.VITE_API_BASE_URL ||
    'http://localhost:18011'
  );
  const [token, setToken] = useState(
    localStorage.getItem('videoannotator_api_token') ||
    import.meta.env.VITE_API_TOKEN ||
    '' // Empty = anonymous access
  );
  const [showToken, setShowToken] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [tokenStatus, setTokenStatus] = useState<TokenStatus | null>(null);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [serverAuthRequired, setServerAuthRequired] = useState<boolean | null>(null);

  // Check token validity
  const validateToken = async (testUrl?: string, testToken?: string) => {
    const urlToTest = testUrl || apiUrl;
    const tokenToTest = testToken !== undefined ? testToken : token;

    setIsValidating(true);

    // Empty token = anonymous access
    if (!tokenToTest.trim()) {
      try {
        // Test anonymous access
        const response = await fetch(`${urlToTest}/api/v1/jobs?per_page=1`);
        if (response.ok) {
          setTokenStatus({ isValid: true, user: 'Anonymous' });
        } else if (response.status === 401) {
          // Server requires auth - this is expected, not an error in the config
          setTokenStatus({ 
            isValid: false, 
            error: 'Server requires authentication. You need to provide an API token to connect.' 
          });
        } else {
          setTokenStatus({ isValid: false, error: `Server error: ${response.status}` });
        }
      } catch (error) {
        setTokenStatus({ isValid: false, error: 'Cannot connect to server' });
      } finally {
        setIsValidating(false);
      }
      return;
    }

    try {
      // First, check server's auth requirements
      const healthResponse = await fetch(`${urlToTest}/api/v1/system/health`);
      const healthData = await healthResponse.json();
      const authRequired = healthData.auth_required ?? true; // Default to true if not present
      setServerAuthRequired(authRequired); // Store for use in UI

      // Check if this is a known placeholder token
      const isPlaceholderToken = ['dev-token', 'test-token', 'your-api-token'].includes(tokenToTest.toLowerCase());

      // If server doesn't require auth and user has a placeholder token, tell them to clear it
      if (!authRequired && isPlaceholderToken) {
        setTokenStatus({
          isValid: false,
          error: 'This is a placeholder, not a real token. Clear this field - the server does not require authentication.'
        });
        setIsValidating(false);
        return;
      }

      // Test with actual jobs endpoint 
      const jobsResponse = await fetch(`${urlToTest}/api/v1/jobs?per_page=1`, {
        headers: { 'Authorization': `Bearer ${tokenToTest}` }
      });

      // Server rejected this token
      if (jobsResponse.status === 401) {
        if (isPlaceholderToken) {
          setTokenStatus({ 
            isValid: false, 
            error: authRequired 
              ? 'This is a placeholder, not a real token. The server requires authentication - you need a real token.'
              : 'This is a placeholder, not a real token. Clear this field to connect without authentication.'
          });
        } else {
          setTokenStatus({ 
            isValid: false, 
            error: 'Server rejected this token. It may be expired or invalid.' 
          });
        }
        setIsValidating(false);
        return;
      }

      if (!jobsResponse.ok && jobsResponse.status !== 404) {
        setTokenStatus({ isValid: false, error: `Server error: ${jobsResponse.status}` });
        setIsValidating(false);
        return;
      }

      // Server accepted a placeholder token when auth is required - shouldn't happen but handle it
      if (isPlaceholderToken && authRequired) {
        setTokenStatus({
          isValid: false,
          error: 'This is a placeholder, not a real token. Clear this field or use a real token.'
        });
        setIsValidating(false);
        return;
      }

      // Jobs endpoint accepted the token, now try to get token details
      try {
        const debugResponse = await fetch(`${urlToTest}/api/v1/debug/token-info`, {
          headers: { 'Authorization': `Bearer ${tokenToTest}` }
        });

        if (debugResponse.ok) {
          const data = await debugResponse.json();
          setTokenStatus({
            isValid: true,
            user: data.token?.user_id || 'Unknown',
            permissions: data.token?.permissions || [],
            expiresAt: data.token?.expires_at
          });
        } else {
          // Debug endpoint not available, but jobs endpoint worked
          setTokenStatus({ isValid: true, user: 'Authenticated' });
        }
      } catch {
        // Debug endpoint failed, but jobs endpoint worked
        setTokenStatus({ isValid: true, user: 'Authenticated' });
      }
    } catch (error) {
      setTokenStatus({
        isValid: false,
        error: handleAPIError(error)
      });
    } finally {
      setIsValidating(false);
    }
  };

  // Save configuration
  const saveConfiguration = () => {
    localStorage.setItem('videoannotator_api_url', apiUrl);
    localStorage.setItem('videoannotator_api_token', token);
    setHasUnsavedChanges(false);

    // Update the global API client
    (apiClient as any).baseURL = apiUrl.replace(/\/$/, '');
    (apiClient as any).token = token;

    // Show success message
    const tokenMode = token.trim() ? 'with API token' : 'in anonymous mode';
    console.log(`✅ Configuration saved: URL=${apiUrl}, Token=${token ? '***' : '(empty)'}`);
    
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

  // Load saved configuration on mount and auto-validate to get server status
  useEffect(() => {
    const savedUrl = localStorage.getItem('videoannotator_api_url');
    const savedToken = localStorage.getItem('videoannotator_api_token') || '';

    if (savedUrl) setApiUrl(savedUrl);
    if (savedToken) setToken(savedToken);

    // ALWAYS auto-validate on mount to get server's auth_required status
    // This provides immediate feedback without requiring user to click "Test Connection"
    const urlToValidate = savedUrl || apiUrl;
    const tokenToValidate = savedToken;
    
    // Small delay to let the UI render first
    setTimeout(() => {
      validateToken(urlToValidate, tokenToValidate);
    }, 100);
  }, []);

  // Track changes and clear stale validation status
  useEffect(() => {
    const savedUrl = localStorage.getItem('videoannotator_api_url') || '';
    const savedToken = localStorage.getItem('videoannotator_api_token') || '';

    const hasChanges = apiUrl !== savedUrl || token !== savedToken;
    setHasUnsavedChanges(hasChanges);

    // Clear validation status when token/url changes to avoid showing stale results
    if (hasChanges) {
      setTokenStatus(null);
    }
  }, [apiUrl, token]);

  const resetToDefaults = () => {
    // Reset to environment variable defaults
    const defaultUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:18011';
    const defaultToken = import.meta.env.VITE_API_TOKEN || ''; // Empty = anonymous

    // Save defaults to localStorage immediately
    localStorage.setItem('videoannotator_api_url', defaultUrl);
    localStorage.setItem('videoannotator_api_token', defaultToken);

    setApiUrl(defaultUrl);
    setToken(defaultToken);
    setHasUnsavedChanges(false); // Already saved

    // Update the global API client
    (apiClient as any).baseURL = defaultUrl.replace(/\/$/, '');
    (apiClient as any).token = defaultToken;

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
            <Input
              id="api-url"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              placeholder="http://localhost:18011"
            />
            <p className="text-sm text-muted-foreground">
              The base URL of your VideoAnnotator API server
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
                              <p>• <strong>Generate yourself:</strong> If you run the server, use: <code className="bg-yellow-100 px-1 rounded">uv run videoannotator generate-token</code></p>
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
                      <p>• <strong>Generate yourself:</strong> If you run the server, use: <code className="bg-yellow-100 px-1 rounded">uv run videoannotator generate-token</code></p>
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
                      <p>• <strong>Generate yourself:</strong> If you run the server, use: <code className="bg-yellow-100 px-1 rounded">uv run videoannotator generate-token</code></p>
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
                              <p>• <strong>Generate yourself:</strong> If you run the server, use: <code className="bg-red-100 px-1 rounded">uv run videoannotator generate-token</code></p>
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

          {/* Actions */}
          <div className="flex items-center gap-3">
            <Button
              onClick={handleTestConnection}
              disabled={isValidating}
              variant="outline"
            >
              {isValidating ? 'Testing...' : 'Test Connection'}
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