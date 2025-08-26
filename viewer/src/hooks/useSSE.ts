import { useEffect, useRef, useState, useCallback } from 'react';
import { apiClient } from '@/api/client';

export interface SSEEvent {
  type: string;
  data: any;
  id?: string;
  timestamp: string;
}

interface UseSSEOptions {
  enabled?: boolean;
  jobId?: string;
  onError?: (error: Error) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
}

export const useSSE = (options: UseSSEOptions = {}) => {
  const {
    enabled = true,
    jobId,
    onError,
    onConnect,
    onDisconnect,
  } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [events, setEvents] = useState<SSEEvent[]>([]);
  const [lastEvent, setLastEvent] = useState<SSEEvent | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;

  const addEvent = useCallback((event: SSEEvent) => {
    setLastEvent(event);
    setEvents(prev => [...prev.slice(-99), event]); // Keep last 100 events
  }, []);

  const connect = useCallback(() => {
    if (!enabled || eventSourceRef.current) {
      return;
    }

    try {
      const eventSource = apiClient.createEventSource(jobId);
      eventSourceRef.current = eventSource;

      eventSource.onopen = () => {
        setIsConnected(true);
        reconnectAttempts.current = 0;
        onConnect?.();
      };

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          addEvent({
            type: 'message',
            data,
            id: event.lastEventId,
            timestamp: new Date().toISOString(),
          });
        } catch (error) {
          console.warn('Failed to parse SSE message:', event.data);
        }
      };

      // Handle specific event types
      eventSource.addEventListener('job.update', (event) => {
        try {
          const data = JSON.parse(event.data);
          addEvent({
            type: 'job.update',
            data,
            id: event.lastEventId,
            timestamp: new Date().toISOString(),
          });
        } catch (error) {
          console.warn('Failed to parse job.update event:', event.data);
        }
      });

      eventSource.addEventListener('job.log', (event) => {
        try {
          const data = JSON.parse(event.data);
          addEvent({
            type: 'job.log',
            data,
            id: event.lastEventId,
            timestamp: new Date().toISOString(),
          });
        } catch (error) {
          console.warn('Failed to parse job.log event:', event.data);
        }
      });

      eventSource.addEventListener('job.complete', (event) => {
        try {
          const data = JSON.parse(event.data);
          addEvent({
            type: 'job.complete',
            data,
            id: event.lastEventId,
            timestamp: new Date().toISOString(),
          });
        } catch (error) {
          console.warn('Failed to parse job.complete event:', event.data);
        }
      });

      eventSource.addEventListener('job.error', (event) => {
        try {
          const data = JSON.parse(event.data);
          addEvent({
            type: 'job.error',
            data,
            id: event.lastEventId,
            timestamp: new Date().toISOString(),
          });
        } catch (error) {
          console.warn('Failed to parse job.error event:', event.data);
        }
      });

      eventSource.onerror = (event) => {
        setIsConnected(false);
        
        // Check if this is a 404 error (endpoint doesn't exist)
        // EventSource doesn't give us HTTP status, but we can infer from readyState
        if (eventSource.readyState === EventSource.CLOSED) {
          console.log('🔌 SSE endpoint unavailable, stopping reconnection attempts');
          const error = new Error('SSE endpoint not available (likely 404). Server may not support real-time events.');
          onError?.(error);
          return;
        }
        
        // Attempt to reconnect with exponential backoff for other errors
        if (reconnectAttempts.current < maxReconnectAttempts) {
          const delay = Math.pow(2, reconnectAttempts.current) * 1000; // 1s, 2s, 4s, 8s, 16s
          reconnectAttempts.current++;
          
          console.log(`🔄 SSE reconnect attempt ${reconnectAttempts.current}/${maxReconnectAttempts} in ${delay/1000}s`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            disconnect();
            connect();
          }, delay);
        } else {
          console.error('❌ SSE connection failed after maximum retry attempts');
          const error = new Error('SSE connection failed after maximum retry attempts');
          onError?.(error);
        }
      };

    } catch (error) {
      const err = error instanceof Error ? error : new Error('Failed to create SSE connection');
      onError?.(err);
    }
  }, [enabled, jobId, onError, onConnect, addEvent]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
      setIsConnected(false);
      onDisconnect?.();
    }
  }, [onDisconnect]);

  const clearEvents = useCallback(() => {
    setEvents([]);
    setLastEvent(null);
  }, []);

  // Connect on mount and when dependencies change
  useEffect(() => {
    if (enabled) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [enabled, jobId]); // Don't include connect/disconnect in deps to avoid loops

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, []);

  return {
    isConnected,
    events,
    lastEvent,
    connect,
    disconnect,
    clearEvents,
  };
};

// Hook for monitoring specific job updates
export const useJobSSE = (jobId: string | undefined, options: Omit<UseSSEOptions, 'jobId'> = {}) => {
  return useSSE({
    ...options,
    jobId,
    enabled: !!jobId && (options.enabled !== false),
  });
};

// Hook for global SSE events (all jobs)
export const useGlobalSSE = (options: Omit<UseSSEOptions, 'jobId'> = {}) => {
  return useSSE({
    ...options,
    jobId: undefined,
  });
};