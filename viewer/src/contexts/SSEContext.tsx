import React, { createContext, useContext, ReactNode } from 'react';
import { useGlobalSSE, SSEEvent } from '@/hooks/useSSE';

interface SSEContextType {
  isConnected: boolean;
  events: SSEEvent[];
  lastEvent: SSEEvent | null;
  connect: () => void;
  disconnect: () => void;
  clearEvents: () => void;
}

const SSEContext = createContext<SSEContextType | undefined>(undefined);

interface SSEProviderProps {
  children: ReactNode;
  enabled?: boolean;
}

export const SSEProvider: React.FC<SSEProviderProps> = ({ 
  children, 
  enabled = true 
}) => {
  const sseHook = useGlobalSSE({
    enabled,
    onError: (error) => {
      console.error('SSE Error:', error);
    },
    onConnect: () => {
      console.log('SSE Connected');
    },
    onDisconnect: () => {
      console.log('SSE Disconnected');
    },
  });

  return (
    <SSEContext.Provider value={sseHook}>
      {children}
    </SSEContext.Provider>
  );
};

export const useSSEContext = () => {
  const context = useContext(SSEContext);
  if (context === undefined) {
    throw new Error('useSSEContext must be used within an SSEProvider');
  }
  return context;
};

// Utility hook to get job-specific events from the global SSE stream
export const useJobEvents = (jobId: string | undefined) => {
  const { events } = useSSEContext();
  
  const jobEvents = React.useMemo(() => {
    if (!jobId) return [];
    
    return events.filter(event => {
      // Filter events related to this specific job
      if (event.data?.job_id === jobId) return true;
      if (event.data?.id === jobId) return true;
      
      // For job-specific event types, assume they're for the current job context
      if (['job.update', 'job.log', 'job.complete', 'job.error'].includes(event.type)) {
        return true;
      }
      
      return false;
    });
  }, [events, jobId]);

  const lastJobEvent = React.useMemo(() => {
    return jobEvents[jobEvents.length - 1] || null;
  }, [jobEvents]);

  return {
    jobEvents,
    lastJobEvent,
  };
};