# Video Annotation Viewer v0.3.0 - QA Testing Checklist

## 🎯 **OVERVIEW**

This comprehensive QA checklist covers the new VideoAnnotator pipeline running functionality added in v0.3.0, along with regression testing for all v0.2.0 features. The primary focus is on the new annotation job creation and management system that integrates with the VideoAnnotator API.

**Testing Date:** 2025-08-26  
**Tester:** Caspar  
**Browser:** Edge  
**OS:** Win11

### Testing checkbox format
- [ ] unchecked boxes for tests that need doing
- [x] ticked boxes for passed tests ✅ 2025-08-06
- [f] f is for fail
      with explanatory comments below
- [>] > for next minor version
- [>>] >> for next major version
---

## 📋 **SECTION 1: VideoAnnotator API Integration**

### **1.1 API Client & Connection**
- [x] **API Health Check**: `http://localhost:8000/health` returns status  
- [f] **Detailed Health**: `/api/v1/system/health` provides system information  
- [x] **Authentication**: API token authentication works correctly  
- [x] **Error Handling**: Network errors display appropriate messages  
- [x] **Environment Variables**: VITE_API_BASE_URL and VITE_API_TOKEN configurable  

**Issues Found:**
```
Let's 
```

### **1.2 Server-Sent Events (SSE)**
- [>] **SSE Connection**: EventSource connects to `/api/v1/events/stream`  
- [>] **Job-Specific SSE**: Job ID filtering works correctly  
- [>] **Event Types**: Handles job.update, job.log, job.complete, job.error events  
- [>] **Reconnection**: Automatic reconnection with exponential backoff  
- [>] **Connection Status**: isConnected state updates correctly  
- [>] **Event History**: Maintains last 100 events in memory  

**Issues Found:**
```
TO test these I think we need a both an events viewer page and a user friendly debug page.
Let's defer these to 0.3.1
```

---

## 📋 **SECTION 2: Navigation & Routing**

### **2.1 Route Structure** 
- [x] **Main Routes**: `/` (viewer), `/create` (annotation jobs) work  
- [x] **Create Sub-routes**: `/create/jobs`, `/create/new`, `/create/datasets` accessible  
- [x] **Job Details**: `/create/jobs/:id` displays job-specific information  
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
[fixed] All pages should have our standard footer.
version 0.4.0: Would be useful to have a 'select folder' option. 
Refresh page forgets all the steps so far and resets to first one. OUght to better than this.
```

### **3.2 Step 2: Pipeline Selection**
- [>] **Pipeline Options**: Scene Detection, Person Tracking, Face Analysis, Audio Processing shown  
- [p] **Checkboxes**: All pipelines checked by default  
- [p] **Descriptions**: Each pipeline shows appropriate description text  
- [x] **Toggle States**: Can check/uncheck individual pipelines  
- [x] **Progress Indicator**: Shows 50% completion  
- [x] **Navigation**: Previous/Next buttons functional  

**Issues Found:**
```
check box fonts still hard to read
pipeline names are black font on black background 
[v0.3.1]
 The Select Pipelines page is very basic. Needs to give detailed information about each possible pipeline and give control over all of them. 
 Face pipeline doesn't list all opions (deepface, openface, etc)
 Audio pipeline doesn't separate pyannote, whisper and LAION.
 I guess we need a mechanism for VideoAnnotator to provide this information. See how far we get reading info currently available from server and we can request more if it's not sufficient
 
```

### **3.3 Step 3: Configuration**
- [x] **Default Config**: Shows JSON configuration preview  
- [x] **Config Display**: Contains scene_detection, person_tracking, face_analysis, audio_processing  
- [p] **Parameters**: Default values match expected pipeline settings  
- [x] **Progress Indicator**: Shows 75% completion  
- [>] **Config Validation**: (Future: ensure valid JSON structure)  

**Issues Found:**
```
[FIXED} Configure Pipelines json box has white text on white background! More generally, page is no use to a naive user. As first step we must give simple on screen example of available options. 
[V 0.4.0] As a further step we should make this into a proper UI. For example many pipelines have some params in common (predictions per second - i think) and so could make some controls as well as raw json. 
[v0.3.1] Finally, we ought to have a mechanism to remember users last choices or preferences.

```

### **3.4 Step 4: Review & Submit**
- [x] **File Summary**: Shows selected video file name  
- [x] **Pipeline Summary**: Lists all selected pipelines  
- [x] **Time Estimate**: Displays "~5-10 minutes" estimate  
- [x] **Submit Button**: Does it work  
- [x] **Progress Indicator**: Shows 100% completion  
- [x] **Final Review**: All information accurate  

**Issues Found:**
```
[v 0.4.0] Feels like this page also ought to indicate what db we are using, with link to settings page that let us modify db location and output directory location. 

```

---

## 📋 **SECTION 4: Job Management Interface**

### **4.1 Jobs List Page (`/create/jobs`)**
- [x] **Page Load**: `/create/jobs` loads without errors  
- [x] **Job Table**: Displays jobs list (when available)  
- [x] **Create New Job**: Button links to `/create/new`  
- [x] **Job Status**: Shows job states and progress  
- [x] **Real-time Updates**: SSE updates job statuses live  
- [x] **Job Navigation**: Can click jobs to view details  

**Issues Found:**
```
[FIXED] + hook.js:608 
 ⚠️ React Router Future Flag Warning: React Router will begin wrapping state updates in `React.startTransition` in v7. You can use the `v7_startTransition` future flag to opt-in early. For more information, see https://reactrouter.com/v6/upgrading/future#v7_starttransition. Error Component Stack
    at SSEProvider (SSEContext.tsx:21:3)

[v0.4.0] -
Now we have api job management, can we make the buttons on the details page implement features like delete job (can we rerun too? shouldn't need to but worth implementing).
And deleting should be possible from the main jobs page, with accompanying select all option.  And from job detail page.
```

### **4.2 Job Detail Page (`/create/jobs/:id`)**
- [x] **Job Info**: Displays job ID, created date, status  
- [x] **Progress Bar**: Shows job completion percentage  
- [x] **Live Logs**: Displays real-time job logs via SSE  
- [f] **Artifacts**: Lists job output files when completed  
- [f] **Open in Viewer**: Button to transition to playback interface  
- [x] **Error Handling**: Graceful display of job errors  

**Issues Found:**
```
Pipeline config is white text on white background.
[v0.3.1] - Pipeline config json text box should be collapsable. and default to collapse.
[v0.4.0] pipeline config should have some option to show conplete config from server with all the defaults it has assumed for this job (pps, etc)
[CRITICAL] Results section doesn't list output files - none of the buttons do anything 
```

---

## 📋 **SECTION 5: Pipeline Integration**

### **5.1 Pipeline API Endpoints**
- [>>] **Get Pipelines**: `/api/v1/pipelines` returns available pipelines  
- [>>] **Pipeline Info**: Each pipeline has name, description, parameters  
- [x] **Job Submission**: POST to `/api/v1/jobs` accepts video + config  
- [>>] **Job Retrieval**: GET `/api/v1/jobs/:id` returns job details  
- [>>] **Jobs List**: GET `/api/v1/jobs` returns paginated job list  

**Issues Found:**
```
[v0.4.0] I assume most of these work because job creation works but user testing can't easily check these directly or more technical challenges like rate limiting. Let's create debug script for testing this. In fact, can we have a user-friendly troubleshooting page that shows live logs and has a panel of packaged tests for debugging/finding issues. 
```

### **5.2 VideoAnnotator Pipeline Support**
- [p] **Scene Detection**: PySceneDetect + CLIP pipeline recognized  
- [p] **Person Tracking**: YOLO11 + ByteTrack pipeline supported  
- [p] **Face Analysis**: OpenFace 3.0 pipeline available  
- [p] **Audio Processing**: Whisper + diarization pipeline included  
- [p] **Configuration**: Pipeline parameters correctly formatted  

**Issues Found:**
```
Yes, as far as it goes but see above comments about limited scope of current pipeline support. 
```

---

## 📋 **SECTION 6: Results Integration**

### **6.1 Job Completion Flow**
- [x] **Completion Detection**: job.complete SSE event triggers UI update  
- [f] **Artifact Retrieval**: Fetches job output files automatically  
- [x] **Format Support**: Handles COCO format outputs from pipelines  
- [p] **Deep Linking**: "Open in Viewer" transitions to playback interface  
- [f] **Data Loading**: Completed annotations load in existing viewer  

**Issues Found:**
```
Can't see any results yet. View buttons not implemented.
[v0.4.0] The viewer itself needs to be pipeline aware. At the moment it is still setup around local preprocessed files. This needs some careful thought to make it as flexible [local raw files and server awareness] and as integrated as possible.
```

### **6.2 Viewer Transition**
- [x] **Seamless Transition**: No page reload when opening results in viewer  
- [x] **Data Preservation**: All annotation data properly loaded  
- [x] **Format Compatibility**: Pipeline outputs compatible with existing parsers  
- [ ] **State Management**: Viewer state properly initialized  

**Issues Found:**
```
[List any viewer transition issues]
```

---

## 📋 **SECTION 7: Performance & Stability**

### **7.1 Performance Optimization (v0.2.0 Fixes)**
- [>] **Multi-Person Rendering**: Smooth with 3+ people in frame  
- [>] **VEATIC Video Support**: Patch Adams video plays without timing issues  
- [>] **Canvas Optimization**: No lag during complex scene playback  
- [>] **Memory Usage**: No memory leaks during extended use  
- [>] **Frame Rate**: Maintains 30fps during overlay rendering  

**Test with:** VEATIC dataset, multiple person scenarios  
**Issues Found:**
```
Will check in 0.3.1 
```

### **7.2 Large File Handling**
- [>] **Large Videos**: >100MB video files upload successfully  
- [>] **Large Annotations**: >10MB annotation files load efficiently  
- [>] **Processing Time**: Reasonable upload/processing times  
- [] **Memory Management**: No browser crashes with large files  

**Issues Found:**
```
Will check in 0.3.1 
```

---

## 📋 **SECTION 8: Error Handling & User Feedback**

### **8.1 Error Messages (v0.2.0 Fixes)**
- [p] **File Type Errors**: Specific messages for malformed JSON  
- [p] **Upload Errors**: Clear feedback for failed uploads  
- [p] **API Errors**: Meaningful messages for API connectivity issues  
- [p] **Network Errors**: Graceful handling of network failures  
- [p] **Validation Errors**: Clear guidance for invalid inputs  

**Issues Found:**
```
[v0.4.0] These seem good but we ought to have some explicit tests in our troubleshooting page when we make that. 
```

### **8.2 User Experience**
- [x] **Loading States**: Appropriate loading indicators throughout  
- [x] **Progress Feedback**: Clear progress indication in wizard  
- [x] **Success Messages**: Confirmation when actions complete  
- [x] **Help Text**: Sufficient guidance for each step  
- [x] **Responsive Design**: Interface adapts to different screen sizes  

**Issues Found:**
```
[v0.4.0] We will design much more detailed and integerated help/documentation in next major version. 
```

---

## 📋 **SECTION 9: Regression Testing (v0.2.0 Features)**

### **9.1 Core Video Playback** ✅ Previously Verified
- [x] **Video Loading**: Demo datasets load correctly  
- [x] **Playback Controls**: Play/pause/seek/speed controls work  
- [x] **Frame Stepping**: Forward/backward frame stepping  
- [x] **Timeline Sync**: Video and annotations stay synchronized  

### **9.2 Annotation Overlays** ✅ Previously Verified  
- [x] **COCO Keypoints**: 17-point skeleton rendering  
- [x] **Person Tracking**: Track IDs and bounding boxes  
- [x] **Speech Recognition**: WebVTT subtitle positioning  
- [x] **Speaker Diarization**: RTTM speaker segments  
- [x] **Scene Detection**: Scene boundary indicators  

### **9.3 Timeline Features** ✅ Previously Verified
- [x] **Subtitle Track**: Speech recognition timeline  
- [x] **Speaker Track**: Speaker diarization timeline  
- [x] **Scene Track**: Scene boundary timeline  
- [x] **Timeline Navigation**: Click-to-seek functionality  

### **9.4 Unified Controls** ✅ Previously Verified
- [x] **Overlay Toggles**: All overlay on/off switches  
- [x] **Lock Functionality**: Synchronized overlay/timeline settings  
- [x] **Color Coding**: Component-specific colors maintained  
- [x] **Bulk Controls**: "All On/All Off" buttons  

**Regression Issues:**
```
[List any features that broke from v0.2.0]
```

---

## 📋 **SECTION 10: Browser Compatibility**

### **10.1 Cross-Browser Testing**
- [ ] **Chrome**: Full functionality verified  
- [x] **Firefox**: All features working (✅ Previously verified)  
- [x] **Edge**: Complete compatibility (✅ Previously verified)  
- [ ] **Safari**: Functionality tested (if macOS available)  

**Browser-Specific Issues:**
```
[List any browser compatibility issues]
```

### **10.2 Browser Features**
- [>] **EventSource Support**: SSE works across browsers  
- [>] **File Upload**: File API consistent across browsers  
- [>] **Canvas Rendering**: WebGL/Canvas2D performance consistent  
- [>] **LocalStorage**: Settings persistence works  

---

## 📋 **SECTION 11: Integration Testing**

### **11.1 End-to-End Workflow**
- [p] **Complete Flow**: Upload → Configure → Submit → Monitor → View Results  
- [f] **State Persistence**: Wizard state maintained during navigation  
- [x] **Data Integrity**: No data loss throughout workflow  
- [f] **Session Management**: Proper cleanup on page refresh  

### **11.2 API Integration Stability**
- [>] **Connection Recovery**: Handles API server restarts  
- [x] **Authentication**: Token refresh/validation working  
- [x] **Concurrent Jobs**: Multiple jobs can be created/monitored  
- [x] **Long-Running Jobs**: SSE connection stable for extended periods  

**Integration Issues:**
```
Create jobs should not lose progress on refresh
```

---

## 📋 **SECTION 12: Debug & Development Tools**

### **12.1 Debug Interface** ✅ Previously Verified
- [>>] **Debug Panel**: Ctrl+Shift+D opens debug interface  
- [x] **Terminal UI**: Black background with green text  
- [x] **Automated Tests**: File detection and integrity checking  
- [x] **VEATIC Support**: Specialized dataset testing  

### **12.2 Console Access** ✅ Previously Verified
- [x] **Window.debugUtils**: Available for testing  
- [x] **Demo Data**: `window.debugUtils.DEMO_DATA_SETS` accessible  
- [x] **Version Info**: `window.version.getAppTitle()` works  

---
[v0.4.0] existing debug should be folded into our upcoming troubleshooting page
---



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