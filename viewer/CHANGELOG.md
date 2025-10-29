# Changelog

All notable changes to Video Annotation Viewer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2025-10-29

### 🎯 **Major Changes**
- **VideoAnnotator v1.3.0 Support**: Full integration with enhanced server features including job cancellation, configuration validation, and advanced diagnostics
- **Job Cancellation**: Cancel running jobs with confirmation dialog and real-time status updates
- **Configuration Validation**: Pre-submission validation with detailed error messages and helpful hints
- **Enhanced Authentication**: Improved token setup wizard with status indicator and automatic validation
- **Comprehensive Error Handling**: Consistent error display across all operations with actionable guidance
- **Server Diagnostics**: Real-time monitoring of GPU status, worker queue, and system health

### ✨ **Added**
- **Job Cancellation (US1)**:
  - Cancel button in job detail view with confirmation dialog
  - Real-time status updates via Server-Sent Events (SSE)
  - Graceful handling of already-cancelled or completed jobs
  - Optional cancellation reason support
  - Backward compatible with v1.2.x servers (feature hidden if unavailable)
  
- **Configuration Validation (US2)**:
  - Pre-submission validation with `/api/v1/validate/config` endpoint
  - Per-pipeline validation with `/api/v1/validate/pipeline/{pipeline_id}`
  - Real-time validation feedback in job creation wizard
  - Field-level error messages with specific guidance
  - Warning detection for suboptimal configurations
  - Debounced validation to reduce server load
  
- **Enhanced Authentication (US3)**:
  - Improved token setup wizard with multi-step flow
  - TokenStatusIndicator component showing real-time status
  - Automatic token validation on page load
  - Visual feedback for valid/invalid/expired tokens
  - User information display when authenticated
  - Server capabilities detection and display
  
- **Improved Error Handling (US4)**:
  - ErrorDisplay component for consistent error presentation
  - Structured error parsing with hints and field-level details
  - Collapsible technical details (error codes, request IDs)
  - Toast notifications with helpful hints
  - ErrorBoundary for React rendering errors
  - Copy-to-clipboard for technical details
  - **User-friendly connection error guidance**: No technical jargon, step-by-step troubleshooting
  - **Connection timeout**: 10-second timeout prevents indefinite hanging when server is offline
  - **Automatic CORS detection**: Clear instructions for starting VideoAnnotator server
  - **Simplified setup (v1.3.0+)**: Port 19011 auto-whitelisted, just run `uv run videoannotator`
  
- **Server Diagnostics (US5)**:
  - ServerDiagnostics component with collapsible UI
  - Real-time GPU status (device, CUDA version, memory usage)
  - Worker queue monitoring (active/queued jobs, max concurrent)
  - System diagnostics (database, storage, FFmpeg status)
  - Auto-refresh every 30 seconds when expanded
  - Manual refresh button with loading state
  - Stale data indicator (>2 minutes without update)
  - Human-readable uptime formatting

### 🔧 **Changed**
- **API Client Enhancement**:
  - Added `cancelJob()` method with reason support
  - Added `validateConfig()` and `validatePipeline()` methods
  - Added `getEnhancedHealth()` with fallback to v1.2.x `/health`
  - Improved error parsing with `parseApiError()` utility
  - Better TypeScript types for v1.3.0 responses
  
- **React Architecture**:
  - New `useJobCancellation` hook for job cancellation logic
  - New `useConfigValidation` hook with debouncing
  - New `useTokenStatus` hook for authentication state
  - ServerCapabilitiesContext for feature detection
  - Enhanced error state management across all pages
  
- **UI/UX Improvements**:
  - Consistent error display with helpful hints throughout app
  - Color-coded status indicators (green/yellow/red)
  - Improved loading states and skeleton screens
  - Better accessibility with ARIA labels and keyboard navigation
  - Dark mode support for all new components

### 🐛 **Fixed**
- **Error Handling**: Unified error parsing for v1.2.x and v1.3.0 formats
- **Type Safety**: Strict TypeScript types for all new API endpoints
- **Performance**: Debounced validation reduces unnecessary API calls
- **Reliability**: ErrorBoundary prevents app crashes from rendering errors

### 📚 **Documentation**
- Added comprehensive test coverage for all new features
- Updated CLIENT_SERVER_COLLABORATION_GUIDE with v1.3.0 endpoints
- JSDoc comments for all new functions, hooks, and components
- Detailed inline code documentation

### 🧪 **Testing**
- **Unit Tests**: 70+ tests for new features (job cancellation, validation, error handling)
- **Component Tests**: 40+ tests for UI components (ErrorDisplay, ServerDiagnostics, TokenSetup)
- **Integration Tests**: End-to-end flows for all user stories
- **Test Coverage**: >80% for new code

### 🔄 **Backward Compatibility**
- Full backward compatibility with VideoAnnotator v1.2.x servers
- Feature detection and graceful degradation
- UI elements hidden when server doesn't support features
- No breaking changes to existing functionality
- **Note**: VideoAnnotator server v1.3.0+ (October 2025) now automatically whitelists port 19011 (this web app) and port 18011 (server). Simple setup: `uv run videoannotator` - no configuration needed. For custom development or remote testing, use `--dev` flag.

### ⚠️ **Known Issues**
- Some timing-related tests for auto-refresh need refinement (React Query + fake timers interaction)
- ServerDiagnostics component tests: 16/24 passing (67%), remaining failures are timing-related edge cases

### 🚀 **Migration Guide**
No migration needed - all changes are additive and backward compatible. If using VideoAnnotator v1.3.0+, new features will automatically become available.

---

## [0.4.0] - 2025-09-26

### 🎯 **Major Changes**
- **Dynamic Pipeline Integration**: Complete VideoAnnotator v1.2.x pipeline discovery and introspection support
- **Capability-Aware UI**: Smart overlay controls that adapt to server capabilities and job-specific data availability
- **Dynamic Parameter Forms**: Auto-generated configuration forms based on server-provided parameter schemas
- **Enhanced Server Diagnostics**: Comprehensive pipeline catalog inspection and server capability detection

### ✨ **Added**
- **Pipeline Discovery System**:
  - Dynamic pipeline catalog fetching from VideoAnnotator server
  - Server capability detection and feature flag support
  - Automatic fallback to legacy mode for older servers
  - Pipeline cache management with TTL and manual refresh
- **Smart Job Creation Wizard**:
  - Parameter forms generated from server schemas with validation
  - Support for all parameter types (boolean, number, enum, multiselect, object, string)
  - Real-time validation with min/max constraints and required field checking
  - Pipeline selection based on server availability and defaults
- **Capability-Aware Viewer**:
  - OpenFace3 controls now check server capabilities and job pipelines
  - Helpful tooltips explaining why features are unavailable
  - Visual indicators for available vs unavailable overlay features
  - Job-specific pipeline awareness for overlay controls
- **Enhanced Settings & Diagnostics**:
  - Server pipeline catalog browser with version and model information
  - Feature flag display and server capability inspection
  - Pipeline refresh and cache management controls
  - Comprehensive server diagnostics and debugging tools

### 🔧 **Changed**
- **API Client Architecture**:
  - Refactored error handling into separate module (`src/api/handleError.ts`)
  - Enhanced pipeline catalog caching with intelligent invalidation
  - Improved server version detection and feature negotiation
- **React Architecture**:
  - Added `PipelineProvider` context for app-wide pipeline state management
  - New React hooks for pipeline data management (`usePipelineData`, `usePipelineContext`)
  - Enhanced error boundaries and loading state management
- **UI/UX Improvements**:
  - Pipeline controls show clear availability status with color coding
  - Improved parameter form UX with better validation feedback
  - Enhanced settings page with comprehensive server information

### 🐛 **Fixed**
- **Type Safety**: Improved TypeScript definitions for pipeline types and API responses
- **Error Handling**: Better error recovery for network failures and server incompatibilities
- **Performance**: Optimized pipeline data loading and caching strategies

### 🧪 **Testing**
- **E2E Tests**: Playwright smoke tests passing across all browsers
- **Unit Tests**: Pipeline integration test coverage added
- **QA Framework**: Comprehensive v0.4.0 testing checklist created

## [0.3.1] - 2025-09-26

### 🔧 **Changed**
- **UI Readability**: Improved text visibility in both light and dark themes, particularly for the JSON configuration editor in the job creation wizard.
- **User Guidance**: Enhanced descriptions for pipelines and clearer instructions for API token setup to improve user experience.

### 🐛 **Fixed**
- **Stability**: Resolved a crash on the job detail page when encountering invalid or missing job data.
- **Console Errors**: Significantly reduced console error spam by addressing React Router warnings and improving Server-Sent Events (SSE) connection handling.

## [0.3.0] - 2025-08-25

### 🎯 **Major Changes**
- **VideoAnnotator Job Creation**: Complete integration with VideoAnnotator API server for creating and managing annotation jobs
- **Professional Job Management**: Full job lifecycle management with creation wizard, real-time monitoring, and results integration
- **API Server Integration**: Native VideoAnnotator API client with authentication, error handling, and Server-Sent Events
- **Enhanced User Experience**: Consistent branding, improved readability, and comprehensive error feedback

### ✨ **Added**
- **Job Creation Wizard**:
  - Multi-step job creation interface (`/create/new`)
  - Video file upload with batch processing support
  - Pipeline selection (scene detection, person tracking, face analysis, audio processing)
  - JSON configuration editor with theme-aware styling
  - Real-time form validation and user guidance
- **Job Management System**:
  - Job listing page with status indicators (`/create/jobs`)
  - Individual job detail pages with progress tracking (`/create/jobs/:id`)
  - Real-time status updates via Server-Sent Events (SSE)
  - Job artifact management and file handling
- **API Integration**:
  - Complete VideoAnnotator API client (`src/api/client.ts`)
  - Token-based authentication with localStorage persistence
  - Comprehensive error handling and user feedback
  - Network resilience and connection status monitoring
- **Professional Interface**:
  - VideoAnnotator icon integration across all pages
  - Enhanced footer with clear VideoAnnotator vs GitHub distinction
  - Consistent branding and color scheme throughout
  - Theme-aware text colors for perfect readability
- **Developer Tools**:
  - Enhanced debug utilities with API testing capabilities
  - `VideoAnnotatorDebug.runAllTests()` for comprehensive API validation
  - Better error logging and troubleshooting guidance
  - Console utilities for both demo data and API testing

### 🔧 **Changed**
- **Application Architecture**:
  - Added React Router for multi-page navigation
  - Implemented page-based architecture with layout components
  - Added React Query for server state management
  - Integrated Server-Sent Events for real-time updates
- **User Interface**:
  - All text now uses theme-aware colors (`text-foreground`, `text-muted-foreground`)
  - Configuration text editors properly readable in all themes
  - Pipeline descriptions enhanced with detailed explanations
  - Button functionality improved with proper user feedback
- **Error Handling**:
  - SSE connections gracefully handle unavailable endpoints
  - React Router future flags added to eliminate console warnings
  - Job detail pages include proper error boundaries
  - API errors provide actionable user guidance
- **Footer Design**:
  - Three-section layout: Version Info | Powered by VideoAnnotator | Source & Docs
  - Clear distinction between VideoAnnotator (processing) and GitHub (viewer source)
  - Enhanced tooltips explaining each link's purpose

### 🐛 **Fixed**
- **Critical Readability Issues**:
  - White-on-white text in JSON configuration editors
  - Dark-on-dark text issues across pipeline selection pages
  - Theme inconsistencies throughout job creation interface
- **Functionality Issues**:
  - Submit button now has comprehensive error handling and logging
  - View Results buttons provide clear feedback instead of silent redirects
  - Pipeline checkboxes correctly default to all enabled
  - Job detail page crashes (`Cannot access 'job' before initialization`)
- **Console Errors**:
  - React Router future flag warnings eliminated
  - SSE connection errors properly handled when endpoints unavailable
  - React forwardRef warnings in Badge component resolved
- **User Experience**:
  - All buttons now functional with appropriate feedback
  - Error messages provide specific troubleshooting steps
  - Job creation workflow provides clear progress indicators

### 📚 **Documentation**
- **Comprehensive v0.3.0 Updates**:
  - Main README updated with job creation workflow and v0.3.0 features
  - Enhanced VideoAnnotator integration explanation with workflow diagram
  - Citations, Credits & Contact section with Zenodo DOI placeholder
  - All documentation links updated to current v0.3.0 structure
- **Developer Resources**:
  - Complete API integration section in Developer Guide
  - CLIENT_SERVER_COLLABORATION_GUIDE.md for VideoAnnotator API
  - Updated project structure documentation with all v0.3.0 components
  - QA testing checklists for both comprehensive and focused testing
- **Navigation Improvements**:
  - Documentation index completely restructured for v0.3.0
  - Clear separation of current vs historical documentation
  - Enhanced hyperlinking throughout all documentation
  - Professional quick start guides for developers

### 🔗 **Integration**
- **VideoAnnotator API Server**:
  - Full REST API integration for job management
  - Server-Sent Events for real-time job monitoring
  - Token-based authentication with configuration UI
  - Health checks and connectivity validation
- **Enhanced Ecosystem Integration**:
  - Clear workflow: VideoAnnotator (processing) → Video Annotation Viewer (visualization)
  - API client handles all VideoAnnotator server communication
  - Professional onboarding for API token setup
  - Comprehensive error handling for server integration

### 🎨 **User Experience**
- **Professional Branding**: VideoAnnotator icon and consistent visual identity across all pages
- **Improved Readability**: All text properly visible in both light and dark themes
- **Better Navigation**: Clear distinction between viewer and job management interfaces
- **Enhanced Feedback**: Users always know the status of their actions with clear messaging
- **Intuitive Workflows**: Job creation wizard guides users through complete annotation process

### ⚡ **Performance**
- **Optimized API Calls**: Efficient server communication with proper caching
- **Smart Error Recovery**: SSE connections intelligently handle server unavailability
- **Enhanced Loading States**: Better user feedback during API operations
- **Memory Management**: Proper cleanup of SSE connections and API clients

### 🔒 **Security**
- **Token Management**: Secure API token storage and transmission
- **Input Validation**: Comprehensive validation of all user inputs
- **Error Sanitization**: Safe error messages without exposing sensitive information

### 🗑️ **Removed**
- Basic file-only workflow (enhanced with job creation capabilities)
- Hardcoded gray text colors (replaced with theme-aware styling)
- Silent error handling (replaced with comprehensive user feedback)

## [0.2.0] - 2025-08-06

### 🎯 **Major Changes**
- **Unified Interface Design**: Complete UI overhaul with elegant two-column layout and professional controls
- **Enhanced File Detection**: Sophisticated dual-method JSON detection supporting all VideoAnnotator formats
- **Professional Debug System**: Built-in debugging panel with automated testing capabilities
- **VEATIC Dataset Integration**: Added support for longer duration silent video analysis

### ✨ **Added**
- **Unified Controls Panel**: 
  - Combined overlay and timeline controls into single elegant interface
  - Custom colored circle buttons replacing basic toggles
  - Padlock functionality for synchronized control modes
  - Individual JSON viewer buttons for each pipeline component
- **Advanced File Detection**:
  - Dual-method JSON detection (fileUtils + sophisticated merger fallback)
  - Support for VEATIC dataset JSON files that were previously undetected
  - Enhanced error reporting with specific file type confidence levels
- **Debug Panel System**:
  - Professional debugging interface accessible via Ctrl+Shift+D or button
  - Automated file detection testing for VEATIC datasets
  - Data integrity checking for all demo datasets
  - Terminal-style logging with real-time test feedback
- **Navigation Improvements**:
  - "← Home" button for returning to landing page from viewer
  - VideoAnnotator documentation link in footer with explanatory tooltip
  - Updated browser favicon using VideoAnnotationViewer.png
- **New Demo Dataset**: VEATIC Silent Video (3.mp4) for pose tracking analysis

### 🔧 **Changed**
- **Interface Layout**: 
  - Redesigned from three-column to elegant two-column layout
  - Video Player + Controls on left, Unified Controls on right
  - Timeline positioned below for optimal space utilization
  - Proper responsive behavior and alignment
- **Control System**:
  - Replaced toggle switches with intuitive colored circle buttons
  - Color-coded components matching their overlay colors
  - Bulk controls renamed to "All On/All Off" for clarity
  - Lock functionality creates single-column elegant mode
- **File Processing**:
  - Enhanced JSON detection with larger sample sizes (10KB vs 500 bytes)
  - String-based content indicators for better format recognition
  - Comprehensive debug output for unknown file analysis
- **User Experience**:
  - Improved subtitle positioning centered at video bottom
  - Better error handling with user-friendly messages
  - Enhanced progress feedback during file processing

### 🐛 **Fixed**
- **Layout Issues**: Resolved three-column alignment problems and control visibility
- **File Detection**: Fixed "Unknown file type" errors for VEATIC dataset JSON files
- **Control Functionality**: Fixed Timeline Lock to Overlays button operation
- **Visual Issues**: Corrected color visibility problems with overlay toggles
- **Debug Access**: Fixed debug panel availability on file loading page
- **Timeline Controls**: Standardized JSON button naming and functionality

### 📚 **Documentation**
- **CLAUDE.md**: Updated with comprehensive v0.2.0 architecture and patterns
- **QA Checklist**: Complete testing checklist with manual verification steps
- **Implementation Guides**: Detailed v0.2.0 implementation status and progress tracking
- **Debug Utils Guide**: Enhanced with new debugging panel capabilities

### 🔗 **Integration**
- **VideoAnnotator Ecosystem**: 
  - Footer link to VideoAnnotator documentation
  - Explanation that VideoAnnotator generates the data files
  - Updated copyright text referencing VideoAnnotator pipeline outputs
- **Developer Experience**:
  - Restored window.debugUtils with enhanced capabilities
  - Data integrity checking functions for all datasets
  - Automated testing workflows for file detection

### 🎨 **User Experience**
- **Professional Interface**: Clean, modern design with consistent color theming
- **Intuitive Controls**: Color-coded buttons with clear visual feedback
- **Enhanced Navigation**: Easy return to landing page and external documentation
- **Better Onboarding**: Improved demo loading with multiple dataset options
- **Debug-Friendly**: Professional debugging tools for developers and testers

### ⚡ **Performance**
- **File Detection**: Optimized JSON parsing with smart sampling techniques
- **UI Responsiveness**: Smoother control interactions and state management
- **Error Handling**: Graceful degradation with helpful user feedback

### 🗑️ **Removed**
- Basic toggle switches (replaced with colored circle buttons)
- Three-column layout constraints (replaced with flexible two-column design)
- Simple file detection (enhanced with sophisticated fallback system)

## [0.1.0] - 2025-08-06

### ✨ **Initial Release**
- **Core Architecture**: React + TypeScript + Vite + Tailwind CSS + shadcn/ui foundation
- **VideoAnnotator Integration**: Full support for VideoAnnotator v1.1.1 pipeline outputs
- **Multimodal Support**: 
  - COCO format pose detection with 17-point skeleton rendering
  - WebVTT speech recognition with synchronized subtitles
  - RTTM speaker diarization with timeline visualization
  - Scene detection with boundary markers
- **File Handling**:
  - Drag-and-drop multi-file upload interface
  - Intelligent file type detection and validation
  - Support for MP4, WebM, AVI, MOV video formats
- **Interactive Features**:
  - Real-time synchronized video playback
  - Multi-track interactive timeline
  - Overlay toggle controls for all annotation types
  - Professional video controls with frame stepping
- **Demo System**: Built-in demonstration with VideoAnnotator sample data
- **Developer Experience**: 
  - Bun runtime development server
  - Comprehensive type safety with Zod validation
  - Extensible parser system for future format support
- **Documentation**: Complete user and developer guides

---

## Legend

- 🎯 **Major Changes**: Significant feature additions or architectural changes
- ✨ **Added**: New features and capabilities
- 🔧 **Changed**: Changes to existing functionality
- 🐛 **Fixed**: Bug fixes and error corrections
- 📚 **Documentation**: Documentation updates and improvements
- 🔗 **Integration**: External integrations and connectivity
- 🎨 **User Experience**: UI/UX improvements and visual enhancements
- ⚡ **Performance**: Performance improvements and optimizations
- 🔒 **Security**: Security-related changes
- 🗑️ **Removed**: Removed features or deprecated functionality

---

For more details about any release, check the [GitHub Releases](https://github.com/InfantLab/video-annotation-viewer/releases) page.
