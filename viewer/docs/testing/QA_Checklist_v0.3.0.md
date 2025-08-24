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
- [x] **API Health Check**: `http://localhost:8000/health` returns status  
- [f] **Detailed Health**: `/api/v1/system/health` provides system information  
- [p] **Authentication**: API token authentication works correctly  
- [x] **Error Handling**: Network errors display appropriate messages  
- [x] **Environment Variables**: VITE_API_BASE_URL and VITE_API_TOKEN configurable  

**Issues Found:**
```
intitially detailed health url gave {"detail":"Not Found"}. But then gave full check details.
how do tell that API token is working? What does user do if it isn't?
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
SSE Disconnected
:8000/api/v1/events/stream?token=dev-token:1   Failed to load resource: the server responded with a status of 404 (Not Found)
SSEContext.tsx:33 SSE Disconnected
:8000/api/v1/events/stream?token=dev-token:1   Failed to load resource: the server responded with a status of 404 (Not Found)
hook.js:608  SSE Error: Error: SSE connection failed after maximum retry attempts
    at eventSource.onerror (useSSE.ts:140:25)
```

---

## 📋 **SECTION 2: Navigation & Routing**

### **2.1 Route Structure** 
- [ ] **Main Routes**: `/` (viewer), `/create` (annotation jobs) work  
- [ ] **Create Sub-routes**: `/create/jobs`, `/create/new`, `/create/datasets` accessible  
- [ ] **Job Details**: `/create/jobs/:id` displays job-specific information  
- [x] **Navigation Menu**: Links to Create Annotations section present  
- [x] **Back Navigation**: Back buttons work correctly throughout wizard  

**Issues Found:**
```
[List any routing/navigation issues]
```

---

## 📋 **SECTION 3: Job Creation Wizard**

### **3.1 Step 1: Video Upload**
- [x] **File Selection**: Video file picker accepts MP4, WebM, AVI, MOV  
- [x] **File Information**: Displays name, size, type after selection  
- [x] **File Validation**: Rejects non-video files appropriately  
- [>] **Large Files**: Handles files >100MB without crashing  
- [x] **Progress Indicator**: Step progress shows 25% completion  
- [x] **Next Button**: Disabled until file selected, enabled after  

**Test Files:** Upload various video formats and sizes  
**Issues Found:**
```
All pages in v-a-v ought to show our icon next to name. 
Let's match colour scheme of UI to complement main colours in icon. (do this with minimal hardcoding)
All pages should have our standard footer.
version 0.4.0: Would be useful to have a 'select folder' option. 
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
+ UI uses lots of dark gray text on black background - hard to read.
+ The Select Pipelines page is very basic. Needs to give detailed information about each possible pipeline
and give control over all of them. 
Face pipeline doesn't list all opions. 
Audio pipeline doesn't separate pyannote, whisper and LAION.
I guess we need a mechanism for VideoAnnotator to provide this information. 


```

### **3.3 Step 3: Configuration**
- [ ] **Default Config**: Shows JSON configuration preview  
- [ ] **Config Display**: Contains scene_detection, person_tracking, face_analysis, audio_processing  
- [ ] **Parameters**: Default values match expected pipeline settings  
- [ ] **Progress Indicator**: Shows 75% completion  
- [ ] **Config Validation**: (Future: ensure valid JSON structure)  

**Issues Found:**
```
+ Configure Pipelines json box has white text on white background! More generally, page is no use to a naive user. As first step we must give simple on screen example of available options. As a further step we should make this into a proper UI. For example many pipelines have some params in common (predictions per second - i think) and so could make some controls as well as raw json. 
Finally, we ought to have a mechanism to remember users last choices or preferences.

```

### **3.4 Step 4: Review & Submit**
- [x] **File Summary**: Shows selected video file name  
- [x] **Pipeline Summary**: Lists all selected pipelines  
- [x] **Time Estimate**: Displays "~5-10 minutes" estimate  
- [p] **Submit Button**: Does it work  
- [x] **Progress Indicator**: Shows 100% completion  
- [p] **Final Review**: All information accurate  

**Issues Found:**
```
Feels like this page also ought to specify and let us modify db location and output directory location. 

```

---

## 📋 **SECTION 4: Job Management Interface**

### **4.1 Jobs List Page (`/create/jobs`)**
- [p] **Page Load**: `/create/jobs` loads without errors  
- [x] **Job Table**: Displays jobs list (when available)  
- [x] **Create New Job**: Button links to `/create/new`  
- [f] **Job Status**: Shows job states and progress  
- [f] **Real-time Updates**: SSE updates job statuses live  
- [f] **Job Navigation**: Can click jobs to view details  

**Issues Found:**
```
+ hook.js:608 
 ⚠️ React Router Future Flag Warning: React Router will begin wrapping state updates in `React.startTransition` in v7. You can use the `v7_startTransition` future flag to opt-in early. For more information, see https://reactrouter.com/v6/upgrading/future#v7_starttransition. Error Component Stack
    at SSEProvider (SSEContext.tsx:21:3)
overrideMethod	@	hook.js:608
+ :8000/api/v1/events/stream?token=dev-token:1   Failed to load resource: the server responded with a status of 404 (Not Found)
+ hook.js:608 
 The above error occurred in the <CreateJobDetail> component:

    at CreateJobDetail (http://localhost:8080/src/pages/CreateJobDetail.tsx:33:23)
    at RenderedRoute (http://localhost:8080/node_modules/.vite/deps/react-router-dom.js?v=dbc932f2:4088:5)
    at Outlet (http://localhost:8080/node_modules/.vite/deps/react-router-dom.js?v=dbc932f2:4494:26)
    at div
    at _c (http://localhost:8080/src/components/ui/card.tsx:23:53)
    at div
    at div
    at CreateLayout (http://localhost:8080/src/pages/Create.tsx:29:22)
    at RenderedRoute (http://localhost:8080/node_modules/.vite/deps/react-router-dom.js?v=dbc932f2:4088:5)
    at Routes (http://localhost:8080/node_modules/.vite/deps/react-router-dom.js?v=dbc932f2:4558:5)
    at Router (http://localhost:8080/node_modules/.vite/deps/react-router-dom.js?v=dbc932f2:4501:15)
    at BrowserRouter (http://localhost:8080/node_modules/.vite/deps/react-router-dom.js?v=dbc932f2:5247:5)
    at Provider (http://localhost:8080/node_modules/.vite/deps/chunk-RS7FYEF2.js?v=dbc932f2:48:15)
    at TooltipProvider (http://localhost:8080/node_modules/.vite/deps/@radix-ui_react-tooltip.js?v=dbc932f2:62:5)
    at SSEProvider (http://localhost:8080/src/contexts/SSEContext.tsx:25:31)
    at QueryClientProvider (http://localhost:8080/node_modules/.vite/deps/@tanstack_react-query.js?v=dbc932f2:2934:3)
    at App

Consider adding an error boundary to your tree to customize error handling behavior.
Visit https://reactjs.org/link/error-boundaries to learn more about error boundaries.
react-dom.development.js:12056 
 Uncaught ReferenceError: Cannot access 'job' before initialization
    at CreateJobDetail (CreateJobDetail.tsx:22:22)

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