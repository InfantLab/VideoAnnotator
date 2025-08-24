import { useState, useEffect } from 'react';
import { apiClient } from '@/api/client';

interface TokenStatus {
  isValid: boolean;
  user?: string;
  permissions?: string[];
  expiresAt?: string;
  error?: string;
  isLoading: boolean;
}

export function useTokenStatus() {
  const [status, setStatus] = useState<TokenStatus>({ 
    isValid: false, 
    isLoading: true 
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

  useEffect(() => {
    validateToken();
  }, []);

  return {
    ...status,
    refresh: refreshToken,
    validate: validateToken
  };
}