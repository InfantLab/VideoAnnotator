# Changelog

All notable changes to Video Annotation Viewer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-08-08

### 🎯 **Major Changes**
- **Project Rebranding**: Renamed from "Video Action Viewer" to "Video Annotation Viewer" for consistency with VideoAnnotator ecosystem
- **Repository Naming**: GitHub repository renamed to `video-annotation-viewer` following npm conventions
- **Interface Showcase**: Added actual interface screenshot demonstrating multimodal annotation capabilities

### ✨ **Added**
- New interface preview image (`VideoAnnotationViewer.png`) showcasing:
  - Real-time pose detection overlays with COCO keypoints
  - Facial emotion recognition visualization
  - Speech recognition subtitles
  - Speaker diarization timeline
  - Scene detection markers
  - Interactive timeline with multiple annotation tracks
- Social media integration with Open Graph and Twitter Card meta tags
- Enhanced README with interface screenshot and detailed feature descriptions
- Comprehensive documentation updates across all files

### 🔧 **Changed**
- **Package Configuration**: 
  - Updated `package.json` name to maintain npm compatibility
  - Updated repository URLs to point to `video-annotation-viewer`
  - Version bumped to 0.2.0
- **Branding Updates**:
  - Application title in `index.html` updated to "Video Annotation Viewer v0.2.0"
  - Welcome screen now uses new interface preview image
  - All documentation references updated to new project name
- **File Organization**:
  - Moved interface preview to `public/` folder for GitHub accessibility
  - Updated import paths in components to use new image asset

### 📚 **Documentation**
- **README.md**: Complete overhaul with:
  - Professional interface screenshot with descriptive caption
  - Updated project name and branding throughout
  - Enhanced feature descriptions and use cases
  - Corrected GitHub repository links
- **Developer Guide**: Updated file structure and repository references
- **Debug Utils Guide**: Updated project name references
- **File Formats Documentation**: Updated tool name references

### 🔗 **Integration**
- GitHub repository properly renamed and configured
- Automatic redirects maintained for old repository URL
- Social sharing optimized with interface preview images
- All documentation links updated for consistency

### 🎨 **User Experience**
- Welcome screen now displays actual interface preview
- Improved visual representation of tool capabilities
- Better onboarding experience with realistic interface preview
- Enhanced social media sharing with interface screenshots

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
