/**
 * Format utilities for ServerDiagnostics
 * T063: Format uptime as human-readable string
 */

/**
 * Format uptime in seconds to human-readable string
 * @param seconds - Uptime in seconds
 * @returns Human-readable string (e.g., "3 days, 4 hours")
 */
export function formatUptime(seconds: number): string {
  if (seconds < 60) {
    return `${seconds} second${seconds !== 1 ? 's' : ''}`;
  }

  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    return `${minutes} minute${minutes !== 1 ? 's' : ''}`;
  }

  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  
  if (hours < 24) {
    if (remainingMinutes === 0) {
      return `${hours} hour${hours !== 1 ? 's' : ''}`;
    }
    return `${hours} hour${hours !== 1 ? 's' : ''}, ${remainingMinutes} minute${remainingMinutes !== 1 ? 's' : ''}`;
  }

  const days = Math.floor(hours / 24);
  const remainingHours = hours % 24;

  if (remainingHours === 0) {
    return `${days} day${days !== 1 ? 's' : ''}`;
  }
  return `${days} day${days !== 1 ? 's' : ''}, ${remainingHours} hour${remainingHours !== 1 ? 's' : ''}`;
}

/**
 * Format bytes to human-readable string with percentage
 * @param used - Bytes used
 * @param total - Total bytes
 * @returns Formatted string (e.g., "5 GB / 10 GB (50%)")
 */
export function formatMemory(used: number, total: number): { used: string; total: string; percentage: number } {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  
  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    const value = bytes / Math.pow(1024, i);
    
    return `${value.toFixed(i > 0 ? 1 : 0)} ${units[i]}`;
  };

  const percentage = total > 0 ? Math.round((used / total) * 100) : 0;

  return {
    used: formatBytes(used),
    total: formatBytes(total),
    percentage,
  };
}

/**
 * Get color class for worker status
 * @param status - Worker status
 * @returns Tailwind color class
 */
export function getWorkerStatusColor(status: 'idle' | 'busy' | 'overloaded'): string {
  switch (status) {
    case 'idle':
      return 'text-green-600 dark:text-green-400';
    case 'busy':
      return 'text-yellow-600 dark:text-yellow-400';
    case 'overloaded':
      return 'text-red-600 dark:text-red-400';
    default:
      return 'text-gray-600 dark:text-gray-400';
  }
}

/**
 * Get color class for health status
 * @param status - Health status
 * @returns Tailwind color class
 */
export function getHealthStatusColor(status: 'healthy' | 'degraded' | 'unhealthy' | string): string {
  switch (status) {
    case 'healthy':
      return 'text-green-600 dark:text-green-400';
    case 'degraded':
      return 'text-yellow-600 dark:text-yellow-400';
    case 'unhealthy':
      return 'text-red-600 dark:text-red-400';
    default:
      return 'text-gray-600 dark:text-gray-400';
  }
}
