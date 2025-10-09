import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { CheckCircle, AlertCircle, Settings, ExternalLink, Eye, EyeOff } from 'lucide-react';
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
    'dev-token'
  );
  const [showToken, setShowToken] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [tokenStatus, setTokenStatus] = useState<TokenStatus | null>(null);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);

  // Check token validity
  const validateToken = async (testUrl?: string, testToken?: string) => {
    const urlToTest = testUrl || apiUrl;
    const tokenToTest = testToken || token;

    if (!tokenToTest.trim()) {
      setTokenStatus({ isValid: false, error: 'Token is required' });
      return;
    }

    setIsValidating(true);
    try {
      // Create a temporary client for testing
      const { APIClient } = await import('@/api/client');
      const testClient = new APIClient(urlToTest, tokenToTest);

      // Test basic connectivity
      await testClient.healthCheck();

      // Try to get token info if debug endpoint is available (optional, may not be enabled)
      try {
        const response = await fetch(`${urlToTest}/api/v1/debug/token-info`, {
          headers: { 'Authorization': `Bearer ${tokenToTest}` }
        });

        if (response.ok) {
          const data = await response.json();
          setTokenStatus({
            isValid: true,
            user: data.token?.user_id || 'Unknown',
            permissions: data.token?.permissions || [],
            expiresAt: data.token?.expires_at
          });
        } else if (response.status === 401 || response.status === 404) {
          // Debug endpoint not available or requires special permissions - this is normal
          // Fall through to basic validation
          const jobsResponse = await fetch(`${urlToTest}/api/v1/jobs?per_page=1`, {
            headers: { 'Authorization': `Bearer ${tokenToTest}` }
          });

          if (jobsResponse.ok || jobsResponse.status === 404) {
            setTokenStatus({ isValid: true });
          } else if (jobsResponse.status === 401) {
            setTokenStatus({ isValid: false, error: 'Invalid or expired token' });
          } else {
            setTokenStatus({ isValid: false, error: `Unexpected response: ${jobsResponse.status}` });
          }
        } else {
          // Token works but no debug info available
          setTokenStatus({ isValid: true });
        }
      } catch {
        // Fallback: try a simple authenticated request
        const response = await fetch(`${urlToTest}/api/v1/jobs?per_page=1`, {
          headers: { 'Authorization': `Bearer ${tokenToTest}` }
        });

        if (response.ok || response.status === 404) {
          setTokenStatus({ isValid: true });
        } else if (response.status === 401) {
          setTokenStatus({ isValid: false, error: 'Invalid or expired token' });
        } else {
          setTokenStatus({ isValid: false, error: `Unexpected response: ${response.status}` });
        }
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

    onTokenConfigured?.();
  };

  // Load saved configuration on mount
  useEffect(() => {
    const savedUrl = localStorage.getItem('videoannotator_api_url');
    const savedToken = localStorage.getItem('videoannotator_api_token');

    if (savedUrl || savedToken) {
      if (savedUrl) setApiUrl(savedUrl);
      if (savedToken) setToken(savedToken);

      // Auto-validate if we have both URL and token
      if (savedUrl && savedToken) {
        validateToken(savedUrl, savedToken);
      }
    }
  }, []);

  // Track changes
  useEffect(() => {
    const savedUrl = localStorage.getItem('videoannotator_api_url') || '';
    const savedToken = localStorage.getItem('videoannotator_api_token') || '';

    setHasUnsavedChanges(
      apiUrl !== savedUrl || token !== savedToken
    );
  }, [apiUrl, token]);

  const resetToDefaults = () => {
    // Reset to environment variable defaults
    const defaultUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:18011';
    const defaultToken = import.meta.env.VITE_API_TOKEN || 'dev-token';

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

  return (
    <div className="max-w-2xl mx-auto space-y-6">
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
            <Label htmlFor="api-token">API Token</Label>
            <div className="relative">
              <Input
                id="api-token"
                type={showToken ? "text" : "password"}
                value={token}
                onChange={(e) => setToken(e.target.value)}
                placeholder="your-api-token"
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
            <p className="text-sm text-muted-foreground">
              Your VideoAnnotator API authentication token. Keep this secure and never share it.
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
                      </div>
                    ) : (
                      <p className="font-medium text-red-700">❌ {tokenStatus.error}</p>
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
              disabled={isValidating || !token.trim()}
              variant="outline"
            >
              {isValidating ? 'Testing...' : 'Test Connection'}
            </Button>

            <Button
              onClick={resetToDefaults}
              variant="outline"
              className="text-orange-600 hover:text-orange-700"
              title={`Reset to: URL=${import.meta.env.VITE_API_BASE_URL || 'http://localhost:18011'}, Token=${import.meta.env.VITE_API_TOKEN || 'dev-token'}`}
            >
              Reset to Defaults
            </Button>

            <Button
              onClick={saveConfiguration}
              disabled={!hasUnsavedChanges || !token.trim()}
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