/**
 * Version information utilities
 */

// For production builds, this would read from package.json
// For now, we'll export the version as a constant
export const VERSION = '0.6.0';
// NOTE: Keep in sync with package.json until we inject this at build time.
export const GITHUB_URL = 'https://github.com/InfantLab/video-annotation-viewer';
export const APP_NAME = 'Video Annotation Viewer';

/**
 * Get formatted version string
 */
export function getVersionString(): string {
  return `v${VERSION}`;
}

/**
 * Get full app title with version
 */
export function getAppTitle(): string {
  return `${APP_NAME} ${getVersionString()}`;
}

/**
 * Log version info to console for debugging
 */
export function logVersionInfo(): void {
  console.log(`🎬 ${getAppTitle()}`);
  console.log(`🔗 GitHub: ${GITHUB_URL}`);
  console.log(`📅 Build: ${new Date().toISOString()}`);
}

// Make version info available globally for browser console
if (typeof window !== 'undefined') {
  window.version = {
    VERSION,
    GITHUB_URL,
    APP_NAME,
    getVersionString,
    getAppTitle,
    logVersionInfo
  };

  // Auto-log version on startup
  logVersionInfo();
}
