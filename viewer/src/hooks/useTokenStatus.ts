import { useState, useEffect } from 'react';
import { apiClient } from '@/api/client';

/**
 * Token validation status and metadata
 */
interface TokenStatus {
  /** Whether the token is valid and authenticated */
  isValid: boolean;
  /** Username or user identifier if authenticated */
  user?: string;
  /** List of user permissions */
  permissions?: string[];
  /** Token expiration timestamp (ISO 8601) */
  expiresAt?: string;
  /** Error message if validation failed */
  error?: string;
  /** Whether validation is in progress */
  isLoading: boolean;
}

/**
 * Hook for validating and monitoring API token status
 * 
 * Automatically validates token on mount and provides methods for
 * manual validation and token refresh.
 * 
 * @returns Token status object and validation functions
 * 
 * @example
 * ```tsx
 * function AuthStatus() {
 *   const { status, refreshToken } = useTokenStatus();
 *   
 *   if (status.isLoading) return <Spinner />;
 *   if (!status.isValid) return <TokenSetup />;
 *   
 *   return <div>Welcome, {status.user}!</div>;
 * }
 * ```
 */
export function useTokenStatus() {
  const [status, setStatus] = useState<TokenStatus>({
    isValid: false,
    isLoading: false // Don't auto-validate - too slow, hangs page load
  });

  const validateToken = async () => {
    setStatus(prev => ({ ...prev, isLoading: true }));

    try {
      const result = await apiClient.validateToken();
      setStatus({
        ...result,
        isLoading: false
      });
    } catch (error) {
      setStatus({
        isValid: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        isLoading: false
      });
    }
  };

  const refreshToken = () => {
    // Update API client with latest localStorage values
    const newUrl = localStorage.getItem('videoannotator_api_url');
    const newToken = localStorage.getItem('videoannotator_api_token');

    if (newUrl || newToken) {
      apiClient.updateConfig(newUrl || undefined, newToken || undefined);
    }

    validateToken();
  };

  // DISABLED: Don't auto-validate on mount - too slow, causes page hang
  // Components can call refresh() manually when needed
  /*
  useEffect(() => {
    validateToken();
  }, []);
  */

  return {
    ...status,
    refresh: refreshToken,
    validate: validateToken
  };
}