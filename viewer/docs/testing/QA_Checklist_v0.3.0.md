# Video Annotation Viewer v0.3.0 - QA Testing Checklist

## 🎯 **OVERVIEW**

This comprehensive QA checklist covers the new VideoAnnotator pipeline running functionality added in v0.3.0, along with regression testing for all v0.2.0 features. The primary focus is on the new annotation job creation and management system that integrates with the VideoAnnotator API.

**Testing Date:** ___________  
**Tester:** ___________  
**Browser:** ___________  
**OS:** ___________

---

## 📋 **SECTION 1: VideoAnnotator API Integration**

### **1.1 API Client & Connection**
- [ ] **API Health Check**: `http://localhost:8000/health` returns status  
- [ ] **Detailed Health**: `/api/v1/system/health` provides system information  
- [ ] **Authentication**: API token authentication works correctly  
- [ ] **Error Handling**: Network errors display appropriate messages  
- [ ] **Environment Variables**: VITE_API_BASE_URL and VITE_API_TOKEN configurable  

**Issues Found:**
```
[List any API connectivity issues]
```

### **1.2 Server-Sent Events (SSE)**
- [ ] **SSE Connection**: EventSource connects to `/api/v1/events/stream`  
- [ ] **Job-Specific SSE**: Job ID filtering works correctly  
- [ ] **Event Types**: Handles job.update, job.log, job.complete, job.error events  
- [ ] **Reconnection**: Automatic reconnection with exponential backoff  
- [ ] **Connection Status**: isConnected state updates correctly  
- [ ] **Event History**: Maintains last 100 events in memory  

**Issues Found:**
```
[List any SSE-related issues]
```

---

## 📋 **SECTION 2: Navigation & Routing**

### **2.1 Route Structure** 
- [ ] **Main Routes**: `/` (viewer), `/create` (annotation jobs) work  
- [ ] **Create Sub-routes**: `/create/jobs`, `/create/new`, `/create/datasets` accessible  
- [ ] **Job Details**: `/create/jobs/:id` displays job-specific information  
- [ ] **Navigation Menu**: Links to Create Annotations section present  
- [ ] **Back Navigation**: Back buttons work correctly throughout wizard  

**Issues Found:**
```
[List any routing/navigation issues]
```

---

## 📋 **SECTION 3: Job Creation Wizard**

### **3.1 Step 1: Video Upload**
- [ ] **File Selection**: Video file picker accepts MP4, WebM, AVI, MOV  
- [ ] **File Information**: Displays name, size, type after selection  
- [ ] **File Validation**: Rejects non-video files appropriately  
- [ ] **Large Files**: Handles files >100MB without crashing  
- [ ] **Progress Indicator**: Step progress shows 25% completion  
- [ ] **Next Button**: Disabled until file selected, enabled after  

**Test Files:** Upload various video formats and sizes  
**Issues Found:**
```
[List any file upload issues]
```

### **3.2 Step 2: Pipeline Selection**
- [ ] **Pipeline Options**: Scene Detection, Person Tracking, Face Analysis, Audio Processing shown  
- [ ] **Checkboxes**: All pipelines checked by default  
- [ ] **Descriptions**: Each pipeline shows appropriate description text  
- [ ] **Toggle States**: Can check/uncheck individual pipelines  
- [ ] **Progress Indicator**: Shows 50% completion  
- [ ] **Navigation**: Previous/Next buttons functional  

**Issues Found:**
```
[List any pipeline selection issues]
```

### **3.3 Step 3: Configuration**
- [ ] **Default Config**: Shows JSON configuration preview  
- [ ] **Config Display**: Contains scene_detection, person_tracking, face_analysis, audio_processing  
- [ ] **Parameters**: Default values match expected pipeline settings  
- [ ] **Progress Indicator**: Shows 75% completion  
- [ ] **Config Validation**: (Future: ensure valid JSON structure)  

**Issues Found:**
```
[List any configuration issues]
```

### **3.4 Step 4: Review & Submit**
- [ ] **File Summary**: Shows selected video file name  
- [ ] **Pipeline Summary**: Lists all selected pipelines  
- [ ] **Time Estimate**: Displays "~5-10 minutes" estimate  
- [ ] **Submit Button**: Currently shows "Submit Job (Coming Soon)"  
- [ ] **Progress Indicator**: Shows 100% completion  
- [ ] **Final Review**: All information accurate  

**Issues Found:**
```
[List any review step issues]
```

---

## 📋 **SECTION 4: Job Management Interface**

### **4.1 Jobs List Page (`/create/jobs`)**
- [ ] **Page Load**: `/create/jobs` loads without errors  
- [ ] **Job Table**: Displays jobs list (when available)  
- [ ] **Create New Job**: Button links to `/create/new`  
- [ ] **Job Status**: Shows job states and progress  
- [ ] **Real-time Updates**: SSE updates job statuses live  
- [ ] **Job Navigation**: Can click jobs to view details  

**Issues Found:**
```
[List any jobs list issues]
```

### **4.2 Job Detail Page (`/create/jobs/:id`)**
- [ ] **Job Info**: Displays job ID, created date, status  
- [ ] **Progress Bar**: Shows job completion percentage  
- [ ] **Live Logs**: Displays real-time job logs via SSE  
- [ ] **Artifacts**: Lists job output files when completed  
- [ ] **Open in Viewer**: Button to transition to playback interface  
- [ ] **Error Handling**: Graceful display of job errors  

**Issues Found:**
```
[List any job detail issues]
```

---

## 📋 **SECTION 5: Pipeline Integration**

### **5.1 Pipeline API Endpoints**
- [ ] **Get Pipelines**: `/api/v1/pipelines` returns available pipelines  
- [ ] **Pipeline Info**: Each pipeline has name, description, parameters  
- [ ] **Job Submission**: POST to `/api/v1/jobs` accepts video + config  
- [ ] **Job Retrieval**: GET `/api/v1/jobs/:id` returns job details  
- [ ] **Jobs List**: GET `/api/v1/jobs` returns paginated job list  

**Issues Found:**
```
[List any pipeline API issues]
```

### **5.2 VideoAnnotator Pipeline Support**
- [ ] **Scene Detection**: PySceneDetect + CLIP pipeline recognized  
- [ ] **Person Tracking**: YOLO11 + ByteTrack pipeline supported  
- [ ] **Face Analysis**: OpenFace 3.0 pipeline available  
- [ ] **Audio Processing**: Whisper + diarization pipeline included  
- [ ] **Configuration**: Pipeline parameters correctly formatted  

**Issues Found:**
```
[List any pipeline support issues]
```

---

## 📋 **SECTION 6: Results Integration**

### **6.1 Job Completion Flow**
- [ ] **Completion Detection**: job.complete SSE event triggers UI update  
- [ ] **Artifact Retrieval**: Fetches job output files automatically  
- [ ] **Format Support**: Handles COCO format outputs from pipelines  
- [ ] **Deep Linking**: "Open in Viewer" transitions to playback interface  
- [ ] **Data Loading**: Completed annotations load in existing viewer  

**Issues Found:**
```
[List any results integration issues]
```

### **6.2 Viewer Transition**
- [ ] **Seamless Transition**: No page reload when opening results in viewer  
- [ ] **Data Preservation**: All annotation data properly loaded  
- [ ] **Format Compatibility**: Pipeline outputs compatible with existing parsers  
- [ ] **State Management**: Viewer state properly initialized  

**Issues Found:**
```
[List any viewer transition issues]
```

---

## 📋 **SECTION 7: Performance & Stability**

### **7.1 Performance Optimization (v0.2.0 Fixes)**
- [ ] **Multi-Person Rendering**: Smooth with 3+ people in frame  
- [ ] **VEATIC Video Support**: Patch Adams video plays without timing issues  
- [ ] **Canvas Optimization**: No lag during complex scene playback  
- [ ] **Memory Usage**: No memory leaks during extended use  
- [ ] **Frame Rate**: Maintains 30fps during overlay rendering  

**Test with:** VEATIC dataset, multiple person scenarios  
**Issues Found:**
```
[List any performance issues]
```

### **7.2 Large File Handling**
- [ ] **Large Videos**: >100MB video files upload successfully  
- [ ] **Large Annotations**: >10MB annotation files load efficiently  
- [ ] **Processing Time**: Reasonable upload/processing times  
- [ ] **Memory Management**: No browser crashes with large files  

**Issues Found:**
```
[List any large file issues]
```

---

## 📋 **SECTION 8: Error Handling & User Feedback**

### **8.1 Error Messages (v0.2.0 Fixes)**
- [ ] **File Type Errors**: Specific messages for malformed JSON  
- [ ] **Upload Errors**: Clear feedback for failed uploads  
- [ ] **API Errors**: Meaningful messages for API connectivity issues  
- [ ] **Network Errors**: Graceful handling of network failures  
- [ ] **Validation Errors**: Clear guidance for invalid inputs  

**Issues Found:**
```
[List any error handling issues]
```

### **8.2 User Experience**
- [ ] **Loading States**: Appropriate loading indicators throughout  
- [ ] **Progress Feedback**: Clear progress indication in wizard  
- [ ] **Success Messages**: Confirmation when actions complete  
- [ ] **Help Text**: Sufficient guidance for each step  
- [ ] **Responsive Design**: Interface adapts to different screen sizes  

**Issues Found:**
```
[List any UX issues]
```

---

## 📋 **SECTION 9: Regression Testing (v0.2.0 Features)**

### **9.1 Core Video Playback** ✅ Previously Verified
- [ ] **Video Loading**: Demo datasets load correctly  
- [ ] **Playback Controls**: Play/pause/seek/speed controls work  
- [ ] **Frame Stepping**: Forward/backward frame stepping  
- [ ] **Timeline Sync**: Video and annotations stay synchronized  

### **9.2 Annotation Overlays** ✅ Previously Verified  
- [ ] **COCO Keypoints**: 17-point skeleton rendering  
- [ ] **Person Tracking**: Track IDs and bounding boxes  
- [ ] **Speech Recognition**: WebVTT subtitle positioning  
- [ ] **Speaker Diarization**: RTTM speaker segments  
- [ ] **Scene Detection**: Scene boundary indicators  

### **9.3 Timeline Features** ✅ Previously Verified
- [ ] **Subtitle Track**: Speech recognition timeline  
- [ ] **Speaker Track**: Speaker diarization timeline  
- [ ] **Scene Track**: Scene boundary timeline  
- [ ] **Timeline Navigation**: Click-to-seek functionality  

### **9.4 Unified Controls** ✅ Previously Verified
- [ ] **Overlay Toggles**: All overlay on/off switches  
- [ ] **Lock Functionality**: Synchronized overlay/timeline settings  
- [ ] **Color Coding**: Component-specific colors maintained  
- [ ] **Bulk Controls**: "All On/All Off" buttons  

**Regression Issues:**
```
[List any features that broke from v0.2.0]
```

---

## 📋 **SECTION 10: Browser Compatibility**

### **10.1 Cross-Browser Testing**
- [ ] **Chrome**: Full functionality verified  
- [ ] **Firefox**: All features working (✅ Previously verified)  
- [ ] **Edge**: Complete compatibility (✅ Previously verified)  
- [ ] **Safari**: Functionality tested (if macOS available)  

**Browser-Specific Issues:**
```
[List any browser compatibility issues]
```

### **10.2 Browser Features**
- [ ] **EventSource Support**: SSE works across browsers  
- [ ] **File Upload**: File API consistent across browsers  
- [ ] **Canvas Rendering**: WebGL/Canvas2D performance consistent  
- [ ] **LocalStorage**: Settings persistence works  

---

## 📋 **SECTION 11: Integration Testing**

### **11.1 End-to-End Workflow**
- [ ] **Complete Flow**: Upload → Configure → Submit → Monitor → View Results  
- [ ] **State Persistence**: Wizard state maintained during navigation  
- [ ] **Data Integrity**: No data loss throughout workflow  
- [ ] **Session Management**: Proper cleanup on page refresh  

### **11.2 API Integration Stability**
- [ ] **Connection Recovery**: Handles API server restarts  
- [ ] **Authentication**: Token refresh/validation working  
- [ ] **Concurrent Jobs**: Multiple jobs can be created/monitored  
- [ ] **Long-Running Jobs**: SSE connection stable for extended periods  

**Integration Issues:**
```
[List any integration problems]
```

---

## 📋 **SECTION 12: Debug & Development Tools**

### **12.1 Debug Interface** ✅ Previously Verified
- [ ] **Debug Panel**: Ctrl+Shift+D opens debug interface  
- [ ] **Terminal UI**: Black background with green text  
- [ ] **Automated Tests**: File detection and integrity checking  
- [ ] **VEATIC Support**: Specialized dataset testing  

### **12.2 Console Access** ✅ Previously Verified
- [ ] **Window.debugUtils**: Available for testing  
- [ ] **Demo Data**: `window.debugUtils.DEMO_DATA_SETS` accessible  
- [ ] **Version Info**: `window.version.getAppTitle()` works  

---

## 📊 **OVERALL ASSESSMENT**

### **Critical Issues (Must Fix Before Release)**
```
[List any critical bugs that prevent core functionality]
```

### **High Priority Issues (Should Fix)**  
```
[List important issues that significantly impact user experience]
```

### **Medium Priority Issues (Could Fix)**
```
[List minor issues that don't prevent usage]
```

### **Low Priority Issues (Future Enhancement)**
```
[List cosmetic or enhancement-level issues]
```

---

## ✅ **SIGN-OFF**

### **Feature Completeness**
- [ ] All planned v0.3.0 features implemented
- [ ] VideoAnnotator API integration functional  
- [ ] Job creation wizard complete
- [ ] Real-time monitoring working
- [ ] Results integration successful

### **Quality Gates**
- [ ] No critical bugs present
- [ ] Performance meets v0.2.0 standards or better
- [ ] All v0.2.0 features still functional (no regressions)
- [ ] Cross-browser compatibility verified
- [ ] Error handling comprehensive

### **Final Approval**
- [ ] **QA Tester Approval**: _________________ Date: _________
- [ ] **Development Lead Approval**: _________________ Date: _________  
- [ ] **Ready for Release**: _________________ Date: _________

---

## 📝 **TESTING NOTES**

**Environment Setup:**
```
VideoAnnotator API: http://localhost:8000
Video Annotation Viewer: http://localhost:8080
API Token: [configured in .env]
Test Dataset: [specify datasets used]
```

**Additional Notes:**
```
[Any additional observations, recommendations, or context]
```

---

**Checklist Version:** v0.3.0  
**Last Updated:** 2025-08-23  
**Total Items:** 150+ verification points