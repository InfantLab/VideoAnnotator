// Server capability detection for VideoAnnotator v1.3.0
// Feature detection based on health endpoint response

import type {
  ServerCapabilities,
  HealthResponse,
  EnhancedHealthResponse,
} from '@/types/api';
import { isEnhancedHealthResponse } from '@/types/api';

/**
 * Detect server capabilities from health endpoint response
 * 
 * v1.3.0 servers return enhanced health with:
 * - uptime_seconds (required)
 * - gpu_status (optional)
 * - worker_info (optional)
 * - diagnostics (optional)
 * 
 * v1.2.x servers return basic health:
 * - status
 * - api_version/videoannotator_version
 * - message
 * 
 * @param healthResponse - Raw health endpoint response
 * @returns ServerCapabilities object
 */
export function detectServerCapabilities(
  healthResponse: HealthResponse
): ServerCapabilities {
  const isV13 = isEnhancedHealthResponse(healthResponse);

  // Extract version (prefer api_version, fallback to version)
  let version = 'unknown';
  if ('api_version' in healthResponse && healthResponse.api_version) {
    version = healthResponse.api_version;
  } else if ('version' in healthResponse && healthResponse.version) {
    version = healthResponse.version;
  } else if ('videoannotator_version' in healthResponse && healthResponse.videoannotator_version) {
    version = healthResponse.videoannotator_version;
  }

  return {
    version,
    supportsJobCancellation: isV13,
    supportsConfigValidation: isV13,
    supportsEnhancedHealth: isV13,
    supportsEnhancedErrors: isV13,
    detectedAt: new Date(),
  };
}

/**
 * Check if server version is >= target version
 * 
 * @param serverVersion - Server version string (e.g., "1.3.0", "1.2.1")
 * @param targetVersion - Target version string (e.g., "1.3.0")
 * @returns true if server version >= target version
 */
export function isVersionAtLeast(
  serverVersion: string,
  targetVersion: string
): boolean {
  try {
    const serverParts = parseVersion(serverVersion);
    const targetParts = parseVersion(targetVersion);

    // Compare major.minor.patch
    for (let i = 0; i < 3; i++) {
      if (serverParts[i] > targetParts[i]) return true;
      if (serverParts[i] < targetParts[i]) return false;
    }

    return true; // Equal
  } catch {
    // If parsing fails, assume compatible
    return false;
  }
}

/**
 * Parse semantic version string into [major, minor, patch]
 * Handles versions like "1.3.0", "1.2.1-dev", "v1.3.0"
 */
function parseVersion(version: string): [number, number, number] {
  // Remove 'v' prefix if present
  const cleaned = version.replace(/^v/, '');

  // Remove suffixes like -dev, -alpha, etc.
  const base = cleaned.split('-')[0];

  // Split into parts
  const parts = base.split('.').map(Number);

  // Ensure we have 3 parts
  while (parts.length < 3) {
    parts.push(0);
  }

  return [parts[0] || 0, parts[1] || 0, parts[2] || 0];
}

/**
 * Format uptime seconds into human-readable string
 * 
 * @param seconds - Uptime in seconds
 * @returns Formatted string (e.g., "3 days, 4 hours")
 */
export function formatUptime(seconds: number): string {
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  const parts: string[] = [];

  if (days > 0) {
    parts.push(`${days} day${days !== 1 ? 's' : ''}`);
  }

  if (hours > 0) {
    parts.push(`${hours} hour${hours !== 1 ? 's' : ''}`);
  }

  if (minutes > 0 && days === 0) {
    // Only show minutes if < 1 day
    parts.push(`${minutes} minute${minutes !== 1 ? 's' : ''}`);
  }

  if (secs > 0 && days === 0 && hours === 0) {
    // Only show seconds if < 1 hour
    parts.push(`${secs} second${secs !== 1 ? 's' : ''}`);
  }

  return parts.length > 0 ? parts.join(', ') : 'just started';
}

/**
 * Format GPU memory as percentage string
 * 
 * @param used - Used memory in bytes
 * @param total - Total memory in bytes
 * @returns Formatted percentage (e.g., "45%")
 */
export function formatGpuMemory(used: number, total: number): string {
  if (total === 0) return '0%';
  const percent = Math.round((used / total) * 100);
  return `${percent}%`;
}

/**
 * Get worker status color for UI display
 * 
 * @param status - Worker status
 * @returns Tailwind color class
 */
export function getWorkerStatusColor(
  status: 'idle' | 'busy' | 'overloaded'
): string {
  switch (status) {
    case 'idle':
      return 'text-green-600';
    case 'busy':
      return 'text-yellow-600';
    case 'overloaded':
      return 'text-red-600';
    default:
      return 'text-gray-600';
  }
}

/**
 * Get diagnostic status color for UI display
 * 
 * @param status - Diagnostic status
 * @returns Tailwind color class
 */
export function getDiagnosticStatusColor(
  status: 'healthy' | 'degraded' | 'unhealthy'
): string {
  switch (status) {
    case 'healthy':
      return 'text-green-600';
    case 'degraded':
      return 'text-yellow-600';
    case 'unhealthy':
      return 'text-red-600';
    default:
      return 'text-gray-600';
  }
}

/**
 * Check if capabilities are stale (> 2 minutes old)
 * 
 * @param capabilities - Server capabilities object
 * @returns true if stale
 */
export function areCapabilitiesStale(capabilities: ServerCapabilities): boolean {
  const ageMs = Date.now() - capabilities.detectedAt.getTime();
  const twoMinutesMs = 2 * 60 * 1000;
  return ageMs > twoMinutesMs;
}
