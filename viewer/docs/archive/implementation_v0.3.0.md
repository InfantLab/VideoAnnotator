# Video Annotation Viewer v0.3.0 - Implementation Plan
### Planning Phase: 2025-08-07
### Target Release: TBD

## 🎯 **VERSION 0.3.0 OVERVIEW**

Building on the **successful v0.2.0 foundation**, v0.3.0 **expands the project scope** to include a **VideoAnnotator GUI integration** as the primary focus, enabling users to create annotation jobs via the VideoAnnotator API and then view results in the existing playback interface. Secondary priorities include performance optimization and enhanced user experience based on comprehensive QA testing results from August 2025.

### **🎉 v0.2.0 ACHIEVEMENTS** 
Based on QA Checklist testing (✅ 2025-08-06), v0.2.0 successfully delivered:
- **Complete Core Functionality**: All person tracking, overlays, and video controls working
- **Multi-Pipeline Support**: VideoAnnotator v1.1.1 complete_results.json format fully supported  
- **Professional Visualization**: YOLO skeleton rendering with proper connections and colors
- **Comprehensive Timeline**: Subtitle, speaker, and scene tracks functional
- **Debug Infrastructure**: `window.debugUtils` available for testing and development
- **Robust Browser Support**: Firefox and Edge fully functional

---

## 📋 **QA TESTING RESULTS - AUGUST 2025**

### **✅ VERIFIED WORKING (Passed in QA Testing)**
- **All Core Video Loading**: Demo datasets load correctly ✅ 2025-08-06
- **Person Tracking Overlays**: YOLO skeleton rendering fully functional ✅ 2025-08-06 
- **COCO Keypoints**: All 17 keypoints render with proper connections ✅ 2025-08-06
- **Track IDs & Bounding Boxes**: Multi-person tracking works correctly ✅ 2025-08-06
- **Speech Recognition**: WebVTT timing and positioning working ✅ 2025-08-06
- **All Overlay Toggles**: Pose, subtitles, speakers, scenes, faces, emotions ✅ 2025-08-06
- **Video Controls**: Play/pause, seeking, frame stepping, playback speed ✅ 2025-08-06
- **Timeline Tracks**: Subtitle, speaker, and scene tracks functional ✅ 2025-08-06
- **Browser Compatibility**: Firefox and Edge fully functional ✅ 2025-08-06
- **Debug Utils**: `window.debugUtils` available and datasets accessible ✅ 2025-08-06

### **🔴 CRITICAL ISSUES (Must Fix for v0.3.0 Release)**

#### **1. SSE Endpoint Missing - Server Side** 🔴 BLOCKING
- **Issue**: `/api/v1/events/stream` returns 404 Not Found - real-time job monitoring completely broken
- **QA Evidence**: `SSE Error: Error: SSE connection failed after maximum retry attempts`
- **Impact**: Job progress tracking, live logs, completion notifications all non-functional
- **Files**: Server-side implementation required (not client issue)
- **Status**: ⏳ **AWAITING SERVER FIX**

#### **2. CreateJobDetail Page Crashes** 🔴 HIGH PRIORITY
- **Issue**: `Uncaught ReferenceError: Cannot access 'job' before initialization`
- **QA Evidence**: Component crashes with React error boundary activation
- **Impact**: Cannot view individual job details
- **Files**: `CreateJobDetail.tsx:22:22`
- **Effort**: 2-3 hours
- **Status**: 🔧 **NEEDS CLIENT FIX**

### **🟠 HIGH PRIORITY ISSUES (Should Fix for Release)**

#### **3. UI Accessibility & Color Issues** 🟠 HIGH PRIORITY
- **Issue**: "Dark gray text on black background - hard to read" + "White text on white background in JSON editor"
- **QA Evidence**: Configuration step unusable, poor readability throughout UI
- **Impact**: Users cannot effectively use configuration and pipeline selection
- **Files**: CSS/Tailwind classes across wizard components
- **Effort**: 4-6 hours
- **Status**: 🔧 **NEEDS CLIENT FIX**

#### **4. Pipeline Selection Enhancement** 🟠 HIGH PRIORITY
- **Issue**: "Very basic" pipeline selection, missing detailed information, incomplete face/audio options
- **QA Evidence**: "Face pipeline doesn't list all options. Audio pipeline doesn't separate pyannote, whisper and LAION"
- **Impact**: Users can't make informed pipeline choices
- **Files**: `PipelineSelectionStep` component + requires enhanced server API
- **Effort**: 8-12 hours (client) + server-side work
- **Status**: 🔧 **NEEDS CLIENT + SERVER FIX**

### **🟡 MEDIUM PRIORITY ISSUES (Could Fix)**

#### **5. Branding & Visual Consistency** 🟡 MEDIUM
- **Issue**: "All pages should show our icon next to name" + "match colour scheme to complement icon"
- **QA Evidence**: Inconsistent branding across pages, missing footer on wizard pages
- **Impact**: Unprofessional appearance, poor brand consistency
- **Files**: Layout components, favicon, CSS theme
- **Effort**: 3-4 hours
- **Status**: 🎨 **POLISH IMPROVEMENT**

#### **6. Configuration User Experience** 🟡 MEDIUM
- **Issue**: "Page is no use to a naive user" - JSON configuration too technical
- **QA Evidence**: Users need simple UI controls rather than raw JSON editing
- **Impact**: Poor user experience for non-technical users
- **Files**: `ConfigurationStep` component
- **Effort**: 8-10 hours for UI-based config editor
- **Status**: 🎯 **UX ENHANCEMENT**

### **🟢 LOW PRIORITY / FUTURE VERSIONS**

#### **7. Enhanced Features for v0.4.0**
- **Select Folder Option**: Bulk video selection from directory
- **Output Location Control**: UI for database and output directory configuration  
- **Persistent User Preferences**: Remember pipeline and config choices
- **Advanced Pipeline Controls**: Granular parameter adjustment UI
- **Chrome/Safari Testing**: Complete cross-browser compatibility verification

---

## 🚀 **UPDATED v0.3.0 IMPLEMENTATION PLAN**

### **IMMEDIATE FIXES (This Sprint)**

#### **Client-Side Fixes (Can implement now)**
1. **Fix CreateJobDetail crash** - Variable initialization issue
2. **UI Color/Accessibility fixes** - Dark text readability, white-on-white JSON editor
3. **Add branding consistency** - Icon, footer, color scheme alignment

#### **Awaiting Server-Side (Blocking release)**
1. **SSE endpoint implementation** - Critical for job monitoring
2. **Enhanced pipeline API** - Detailed pipeline information and options

---

## 🚀 **v0.3.0 FEATURE ROADMAP**

### **Phase 1: VideoAnnotator GUI Integration** (Week 1-4) 🔴 **TOP PRIORITY**

#### **1.1 Core API Integration Infrastructure** 
- [ ] **API Client & Type Generation**
  - Generate TypeScript types from VideoAnnotator OpenAPI schema (`http://localhost:18011/openapi.json`)
  - Create API client wrapper with authentication and error handling
  - Implement Server-Sent Events (SSE) connection for real-time job updates
  - Set up React Query or Zustand for state management

- [ ] **Routing & Navigation Structure**
  - Add `/create` route section to existing React Router setup
  - Implement sub-routes: `/create/datasets`, `/create/new`, `/create/jobs`, `/create/jobs/:id`
  - Update main navigation to include "Create Annotations" section
  - Maintain existing viewer functionality at root routes

#### **1.2 Dataset Management Interface**
- [ ] **Dataset Registration & Management**
  - `DatasetManager.tsx` component for listing registered datasets (`GET /datasets`)
  - "Register Dataset" form with name and base_path (`POST /datasets`)
  - Optional video scanning functionality (`POST /datasets/{id}/videos/scan`)
  - Dataset details view with video file listings

#### **1.3 Annotation Job Creation Wizard**
- [ ] **Multi-Step Job Creation Wizard**
  - Step 1: Dataset selection and video file picker with checkboxes
  - Step 2: Pipeline configuration using presets (`GET /presets`) or custom parameters
  - Step 3: Job confirmation with estimated runtime/storage display
  - `VideoPicker.tsx`, `PipelinePicker.tsx`, `PresetEditor.tsx` components

- [ ] **Pipeline Configuration Interface** 
  - Load available pipelines from VideoAnnotator API (`GET /pipelines`)
  - Support for 4 core pipelines: Scene Detection, Person Tracking, Face Analysis, Audio Processing
  - Parameter editors for each pipeline with validation
  - Preset management (save/load/edit custom pipeline configurations)

#### **1.4 Job Monitoring & Results**
- [ ] **Job Management Interface**
  - `JobTable.tsx` with real-time status updates via SSE (`/events/stream`)
  - Job filtering by state, tags, creation date
  - Columns: ID, Created, State, Progress, Tags, Actions
  - `JobDetail.tsx` with progress bars, logs tail, and artifact listing

- [ ] **Results Integration**
  - Fetch job artifacts when completed (`GET /jobs/{id}/artifacts`)
  - "Open in Viewer" deep link integration using artifact manifest
  - Seamless transition from job completion to existing playback interface
  - Support for COCO format outputs from all VideoAnnotator pipelines

### **Phase 2: Performance & Minor Fixes** (Week 5-6)

#### **2.1 Performance Optimization** 🔴 HIGH PRIORITY
- [ ] **Fix Multi-Person Rendering Performance**
  - Optimize canvas rendering for 3+ people in frame
  - Resolve timing issues with VEATIC video datasets (Patch Adams video)
  - Implement efficient frame-based overlay caching
  - Add performance monitoring and metrics

#### **2.2 Minor UI Improvements** 🟠 MEDIUM PRIORITY  
- [ ] **Audio Button Functionality**
  - Fix speaker button behavior (currently doesn't toggle audio)
  - Clarify audio control functionality or remove if not applicable
  - Improve audio control user feedback

- [ ] **Error Message Enhancement**
  - Replace generic "unknown file type" with specific error messages
  - Provide helpful guidance for malformed JSON files
  - Improve user feedback for file loading errors

- [ ] **Browser Compatibility Testing**
  - Complete Chrome functionality verification
  - Complete Safari functionality testing (if macOS available)
  - Document any browser-specific issues or limitations

### **Phase 2: Enhanced Person Tracking** (Week 3-4)

#### **2.1 Advanced Skeleton Visualization** 
- [ ] **Sub-options UI Implementation**
  - Person ID toggle (show/hide track IDs)
  - Bounding box toggle (independent of skeleton)
  - COCO skeleton toggle (independent of boxes)
  - Confidence score display option

- [ ] **Skeleton Connection Review**
  - Verify YOLO connections with additional test videos
  - Fix ear-to-shoulder connections (should connect to head)
  - Test across diverse pose datasets
  - Document connection accuracy improvements

- [ ] **Track Visualization Enhancements**
  - Improved track ID display styling
  - Person following/tracking across frames
  - Color consistency for person identification
  - Track persistence visualization

### **Phase 3: Advanced Face Analysis** (Week 5-6)

#### **3.1 Enhanced Face Detection**
- [ ] **Model Selection Support**
  - Support for different face detection models
  - Model confidence score display
  - Face detection quality indicators

- [ ] **DeepFace Integration Planning**
  - Age estimation display sub-options
  - Gender classification sub-options  
  - Enhanced facial emotion recognition
  - Face attribute visualization controls

- [ ] **OpenFace3 Support Research**
  - Investigate OpenFace3 pipeline integration
  - Face landmark visualization
  - Facial expression analysis compatibility

### **Phase 4: Timeline Enhancements** (Week 7-8)

#### **4.1 Motion Analysis Visualization**
- [ ] **Industry-Standard Motion Intensity**
  - Implement motion intensity algorithms
  - Visual motion representation in timeline
  - Per-person motion tracking lanes
  - Motion heatmap visualization

- [ ] **Multi-Person Timeline Tracks**
  - Individual person tracking lanes
  - Person-specific motion intensity
  - Track-based navigation and selection
  - Synchronized multi-person analysis

- [ ] **Audio Waveform Integration**
  - Visual audio representation in timeline
  - Sync with speech and speaker tracks
  - Audio-visual correlation indicators
  - Timeline audio navigation

### **Phase 5: User Experience Improvements** (Week 9-10)

#### **5.1 Enhanced Interface**
- [ ] **Combined Speech/Speaker Display**
  - "SPEAKER_00: Hello baby" format implementation
  - Color-coded speaker identification
  - Improved subtitle presentation
  - Speaker-speech correlation visualization

- [ ] **Persistent State Management**
  - Reload last video/annotations on page refresh
  - Session state preservation
  - User preference persistence
  - Improved workflow continuity

- [ ] **Layout Optimization**
  - Better screen space utilization
  - Responsive design improvements
  - Mobile-friendly adaptations (research phase)
  - Accessibility enhancements

#### **5.2 Performance & Memory**
- [ ] **Large File Support**
  - Handle >1000 person tracking entries efficiently  
  - Memory optimization for extended use
  - Streaming/chunked data loading
  - Memory leak prevention

- [ ] **Scene Duration Improvements**
  - 1-second scene boundary label display
  - Scene transition visualization
  - Scene-based navigation
  - Scene duration analysis tools

---

## 🔧 **TECHNICAL IMPROVEMENTS**

### **Code Quality & Maintainability**
- [ ] **Debug Code Cleanup**
  - Remove DEBUG comments from merger.ts (8 instances)
  - Clean up console.log statements
  - Implement proper logging system
  - Add development/production build differences

- [ ] **TODO Item Resolution**
  - Implement toast notifications in FileViewer.tsx
  - Complete any remaining technical debt
  - Code documentation improvements
  - Type safety enhancements

### **Testing & Quality Assurance**
- [ ] **Comprehensive Browser Testing**
  - Chrome, Firefox, Safari compatibility
  - Canvas scaling on browser resize
  - Cross-platform testing
  - Performance benchmarking

- [ ] **Large Dataset Testing**
  - >10MB annotation file performance
  - Memory usage during extended use
  - Stress testing with multiple datasets
  - Performance regression testing

### **Documentation Updates**
- [ ] **User Documentation**
  - Update all guides for v0.3.0 features
  - Screenshot updates with new interface
  - Feature demonstration videos
  - Troubleshooting guides

- [ ] **Developer Documentation**
  - Architecture documentation updates
  - API reference improvements
  - Extension point documentation
  - Contributing guidelines enhancement

---

## 📊 **v0.3.0 SUCCESS METRICS**

### **Performance Targets**
- [ ] **Rendering Performance**: Smooth 30fps with 5+ people in frame
- [ ] **Loading Performance**: Large files (>10MB) load within 15 seconds
- [ ] **Memory Usage**: No memory leaks during 4+ hour sessions
- [ ] **Response Time**: <50ms for all UI interactions

### **Functionality Targets**
- [ ] **All Critical Bugs**: 100% resolution of v0.2.0 deferred issues
- [ ] **Feature Completeness**: All planned v0.3.0 features implemented
- [ ] **Browser Compatibility**: 100% functionality across Chrome, Firefox, Safari, Edge
- [ ] **User Experience**: Streamlined workflow with persistent state

### **Quality Targets**
- [ ] **Code Coverage**: >90% test coverage for critical components
- [ ] **Documentation**: 100% feature documentation coverage
- [ ] **Accessibility**: WCAG 2.1 AA compliance research
- [ ] **Performance**: Industry-standard motion analysis algorithms

---

## 🗓️ **DEVELOPMENT TIMELINE**

### **Sprint 1 (Week 1-2): VideoAnnotator API Foundation**
- **Goal**: Establish core API integration infrastructure and routing
- **Deliverables**: API client generation, SSE setup, routing structure, navigation updates
- **Success Criteria**: Functional API connection, basic routing to `/create` section

### **Sprint 2 (Week 3-4): Dataset & Job Management Core**  
- **Goal**: Implement dataset management and job creation wizard
- **Deliverables**: Dataset registration, video picker, pipeline configuration, job submission
- **Success Criteria**: Complete job creation workflow from dataset to submission

### **Sprint 3 (Week 5-6): Job Monitoring & Results Integration**
- **Goal**: Complete job monitoring interface and results integration
- **Deliverables**: Real-time job tracking, progress displays, "Open in Viewer" deep linking
- **Success Criteria**: Full end-to-end workflow from job creation to viewing results

### **Sprint 4 (Week 7-8): Performance & Critical Fixes**
- **Goal**: Address performance issues and critical UX problems from v0.2.0 QA
- **Deliverables**: Multi-person rendering optimization, audio button fix, error handling
- **Success Criteria**: Smooth performance with 3+ people, resolved VEATIC timing issues

### **Sprint 5 (Week 9-10): Browser Compatibility & Polish**
- **Goal**: Complete cross-browser testing and UX improvements
- **Deliverables**: Chrome/Safari verification, combined speech/speaker display, persistent state
- **Success Criteria**: 100% browser compatibility, enhanced user experience

### **Sprint 6 (Week 11-12): Testing & Documentation**
- **Goal**: Comprehensive quality assurance and documentation
- **Deliverables**: Regression testing, performance validation, updated documentation
- **Success Criteria**: Release-ready quality with full feature documentation

---

## 🎯 **PRIORITIZATION FRAMEWORK**

### **Priority 1: VideoAnnotator GUI Integration (Must Have for v0.3.0)**
- VideoAnnotator API client and type generation
- Dataset management interface (register, list, scan)
- Annotation job creation wizard (dataset → videos → pipelines → confirm)
- Job monitoring with real-time SSE updates
- Results integration with "Open in Viewer" deep linking
- Support for all 4 VideoAnnotator pipelines (Scene, Person, Face, Audio)

### **Priority 2: Performance Critical (Should Have)**
- Multi-person rendering performance optimization (3+ people lag)
- VEATIC video timing issues resolution
- Canvas rendering optimization for complex scenes

### **Priority 3: UX Improvements (Should Have)**
- Audio button functionality clarification/fix
- Error message improvements for file loading
- Combined speech/speaker display enhancement
- Persistent state management (reload last video on refresh)

### **Priority 4: Enhanced Features (Could Have)**
- Motion intensity algorithms and visualization
- Enhanced person tracking sub-options
- Audio waveform integration
- Advanced face analysis features

### **Priority 5: Future Enhancements (Won't Have in v0.3.0)**
- Navigation back to landing page
- Online documentation system
- Legacy format support removal
- Mobile responsiveness research
- Export functionality
- Batch processing capabilities

---

## � **v0.3.0 SUCCESS METRICS**

### **Performance Targets** 
- [ ] **Rendering Performance**: Smooth playback with 5+ people in frame (currently fails with 3+)
- [ ] **VEATIC Video Support**: Resolve timing issues with complex videos like Patch Adams
- [ ] **Loading Performance**: Maintain current <5 second demo loading performance
- [ ] **Memory Usage**: No memory leaks during extended use (deferred from v0.2.0)

### **Functionality Targets**
- [ ] **VideoAnnotator Integration**: Complete end-to-end workflow from job creation to result viewing
- [ ] **API Connectivity**: Stable connection to VideoAnnotator backend with error handling
- [ ] **Real-time Updates**: Live job progress monitoring via Server-Sent Events
- [ ] **Pipeline Support**: Full integration with all 4 VideoAnnotator pipelines
- [ ] **Core Features**: Maintain 100% functionality of all ✅ verified features from v0.2.0
- [ ] **Performance Issues**: Resolve all identified performance bottlenecks
- [ ] **Browser Compatibility**: Complete Chrome and Safari testing and verification

### **Quality Targets**
- [ ] **Regression Prevention**: Ensure no v0.2.0 working features break
- [ ] **Performance Benchmarking**: Establish baseline metrics for future optimization
- [ ] **Error Handling**: Improve user feedback for all error conditions
- [ ] **Cross-Browser**: 100% functionality across all major browsers