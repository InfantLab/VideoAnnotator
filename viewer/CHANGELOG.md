# Changelog

All notable changes to Video Annotation Viewer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
