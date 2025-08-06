# Video Annotation Viewer Roadmap

## ✅ Version 0.2.0 - COMPLETED (2025-08-06)

### ✅ **Interface & Controls**
- ✅ **Unified Controls**: Combined overlay and timeline controls into single elegant interface
- ✅ **Toggle All Buttons**: "All On/All Off" bulk controls implemented
- ✅ **Lock Functionality**: "Lock to Overlay" padlock synchronizes control modes
- ✅ **Color-coded Controls**: Intuitive colored circle buttons for each component
- ✅ **JSON Viewers**: Individual data inspection buttons for each pipeline component
- ✅ **Consistent Naming**: Standardized between Timeline and Overlay controls

### ✅ **Video & Overlays**
- ✅ **Person Tracking**: COCO skeleton overlays with proper rendering
- ✅ **Subtitle Positioning**: Fixed alignment and positioning issues
- ✅ **Professional Debug Panel**: Ctrl+Shift+D debugging interface
- ✅ **Enhanced File Detection**: Dual-method JSON detection for VEATIC datasets
- ✅ **Navigation**: Home button and VideoAnnotator documentation links

### ✅ **Developer Experience**
- ✅ **Debug Utilities**: Restored window.debugUtils with data integrity checking
- ✅ **Automated Testing**: File detection testing and dataset validation
- ✅ **Documentation**: Updated all guides and references for v0.2.0

---

## 🎯 Version 0.3.0 - PLANNED

### 🔄 **Enhanced Person Tracking**
- [ ] **Sub-options UI**: Person IDs, bounding boxes, and COCO skeleton toggles
- [ ] **Skeleton Review**: Verify YOLO connections with additional test videos
- [ ] **Track Visualization**: Improved track ID display and person following

### 🎭 **Advanced Face Analysis**  
- [ ] **Model Selection**: Support for different face detection models
- [ ] **DeepFace Integration**: Age, gender, facial emotion sub-options
- [ ] **OpenFace3 Support**: Full OpenFace3 pipeline integration

### 📊 **Timeline Enhancements**
- [ ] **Motion Intensity**: Industry-standard motion analysis visualization
- [ ] **Multi-person Tracks**: Individual person tracking lanes
- [ ] **Audio Waveforms**: Visual audio representation in timeline

### 🎪 **User Experience**
- [ ] **Combined Speech/Speaker**: "SPEAKER_00: Hello baby" format with color coding
- [ ] **Persistent State**: Reload last video/annotations on page refresh  
- [ ] **Performance Optimization**: Smooth playback with multiple people (3+)

### 🔧 **Technical Improvements**
- [ ] **Scene Duration**: 1-second scene boundary label display
- [ ] **Large File Support**: Handle >1000 person tracking entries efficiently
- [ ] **Memory Management**: Prevent memory leaks during extended use

---

## 💡 Future Considerations

### 📱 **Mobile & Responsive**
- [ ] Mobile-friendly interface adaptation
- [ ] Touch gesture support for timeline navigation

### 🌐 **Integration & Export**
- [ ] Export functionality for annotation segments
- [ ] Integration with other analysis tools
- [ ] Batch processing capabilities

### 📈 **Analytics & Metrics**
- [ ] Usage analytics and user behavior insights
- [ ] Performance metrics and optimization targets 

