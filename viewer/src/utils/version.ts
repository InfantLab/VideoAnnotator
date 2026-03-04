/**
 * Version information utilities
 *
 * VERSION is read from package.json at build time via Vite's JSON import,
 * so there is only one source of truth.
 */

// Vite resolves this JSON import at build time — no runtime fs access needed.
import pkg from '../../package.json';

export const VERSION: string = pkg.version;
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

  // Auto-log version on startup and set document title
  logVersionInfo();
  document.title = `${APP_NAME} v${VERSION}`;
}
