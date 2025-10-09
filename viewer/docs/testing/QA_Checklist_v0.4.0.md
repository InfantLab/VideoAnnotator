# Video Annotation Viewer v0.4.0 - Comprehensive Pipeline Integration QA Testing

## 🎯 **OVERVIEW**

This comprehensive testing guide covers the new **v0.4.0 Pipeline Integration** features that enable dynamic pipeline discovery, parameter introspection, and capability-aware UI components.

**Target Build:** v0.4.0 (dynamic VideoAnnotator v1.2.x integration)  
**Last Updated:** 2025-09-26  
**Testing Date:** 2025-09-26  
**Tester:** Caspar Addyman
**Server Version:** VideoAnnotator v1.2.2
**Browser:** Edge 141.0.3537.38 
**OS:** __Windows 

## 🔧 **TEST ENVIRONMENT SETUP**

### **Prerequisites**
- **VideoAnnotator Server**: v1.2.1 or v1.2.2 running at `http://localhost:18011`
- **Environment Variables**: `VITE_API_BASE_URL=http://localhost:18011` and `VITE_API_TOKEN=dev-token`
- **Token Configuration**: Use the "Reset to Defaults" button in Settings → API Configuration if having token issues (much easier than manual localStorage manipulation)
- **Demo Data Available**: Located in `demo/videos_out/` directory

### **Server Setup Commands**
```bash
# Start VideoAnnotator server (if available)
cd /path/to/videoannotator
python -m videoannotator.server --port 18011

# Alternative: Use debug API at scripts/test_api_quick.py
cd video-annotation-viewer
python scripts/test_api_quick.py
```

### **Browser Dev Tools Setup**
1. Open browser Dev Tools (`F12`)
2. Go to **Console** tab for error monitoring
3. Go to **Network** tab to monitor API calls
4. Enable "Preserve log" to keep logs across navigation

### Testing checkbox format
- [ ] unchecked boxes for tests that need doing
- [x] ticked boxes for passed tests ✅ 2025-09-26
- [f] f is for fail with explanatory comments below
- [>] deferred to future minor release 1.3.x
- [>>] deferred to future release 1.4.x

---
## Set up

- [x] Launch with `bun run dev` and confirm minimal console errors on load
- [x] Runs on ports that complement server (e.g., 19011 if server is 18011)

## 🚨 **CONSOLE ERROR TESTING**

### **Expected vs Unexpected Errors**
- [ ] **404 Errors (Expected)**: Some 404s for missing endpoints are normal
- [ ] **404 Error Count**: Should be minimal - not dozens of duplicate requests to same endpoints
- [ ] **401 Errors (Problem)**: No "401 Unauthorized" errors should appear
- [ ] **React DOM Warnings (Problem)**: No "validateDOMNesting" warnings about invalid HTML structure
- [ ] **JavaScript Errors (Problem)**: No uncaught exceptions or runtime errors

## 🚀 **SECTION 1: PIPELINE CATALOG INTEGRATION**

### **1.1 Dynamic Pipeline Discovery**
- [x] **Settings Page Navigation**: Navigate to `/create/settings` → Click "Server Info" tab
- [x] **Pipeline Catalog Display**: Verify "Pipeline Catalog Overview" section shows server-fetched pipelines
- [f] **Server Version Detection**: Check server version displays (e.g., "1.2.1" or "1.2.2")
- [>] **Feature Flags Display**: Verify feature badges show (Pipeline Catalog, Parameter Schemas, etc.)
  - Working but needs UI improvement: color them green for enabled, gray for disabled  
  - Job SSE shows as disabled - investigate if this is correct
- [x] **Refresh Functionality**: Click blue "Refresh" button → catalog updates without page reload
- [x] **Cache Management**: Click "Clear Cache" button → cache cleared, forces next refetch

Gives console error
client.ts:86   GET http://localhost:18011/api/v1/pipelines/catalog 404 (Not Found)

- [>>] **Offline Mode**: Shows cached/stale data when server unavailable

Extra comments:
We should have a link to http://10.5.0.2:19011/create/settings on main landing page. 

**Detailed Test Steps:**
1. **Navigate to Settings**:
   - Click "Create" in top navigation
   - Click "Settings" in sidebar or navigate to `http://localhost:19011/create/settings`
   - Click "Server Info" tab (second tab with server icon)

**Expected Behavior:**
- Server Info tab should load without JavaScript errors
- Pipeline catalog should display available pipelines
- Token status should show as valid
- Some 404 errors are expected (missing v1.2.x endpoints)

### **Critical Tests**
- [x] **Token Status**: Should show "Valid" 
- [x] **Console Errors**: No "401 Unauthorized" errors in browser console
- [x] **Pipeline Loading**: Pipeline catalog displays 6 pipelines
- [x] **Port Check**: Client runs on `localhost:19011`, server on `localhost:18011`

### **Network Validation**
- [x] **Request Targets**: All API requests go to `localhost:18011` (not `localhost:19011`)
- [x] **Error Types**: 404 errors are expected; 401 errors indicate problems
- [x] **Fallback**: Missing endpoints fall back gracefully

### **UI/UX Tests**
- [f] **Version Display**: Check if server version appears (currently shows "Unknown")
- [>] **Feature Flags**: Verify badges display and suggest color improvements

### **Stability Tests**
- [x] **Fresh Browser**: Test with cleared cache/localStorage
- [x] **Server Restart**: Test after restarting VideoAnnotator server
- [x] **Multiple Refresh**: Click "Refresh" button several times
- [f] **Network Issues**: Test when server becomes unavailable
Fine when server restarts. Without server runing the red error could be more helpful - suggest user checks server is running instead of:
Server diagnostics issue
Network error: Failed to fetch


### **Browser Testing**
- [x] **Chrome/Edge**: Primary testing browser
- [ ] **Firefox**: Secondary validation
- [ ] **Safari**: If available

2. **Verify Server Integration Status Card**:
   - Look for "Server Integration Status" card with server URL and token status
   - Verify "Pipelines Detected" shows count with database icon
   - Check "Server Features" section shows colored badges for available features

3. **Test Pipeline Catalog Section**:
   - Scroll to "Pipeline Catalog Overview" section
   - Verify pipelines are grouped (Face, Person, Audio, Scenes, etc.)
   - Each pipeline should show: name, description, version badge, model badge
   - Look for "Updated X time ago" timestamp

4. **Test Refresh Functionality**:
   - Click blue "Refresh" button (with RefreshCw icon)
   - Watch Network tab for `/api/v1/pipelines/catalog` or similar calls
   - Verify timestamp updates to "Updated just now" or similar

5. **Test Cache Management**:
   - Click "Clear Cache" button (with Trash2 icon)
   - Next page load should show loading states again
   - Network tab should show fresh API calls

6. **Test Offline/Error Handling**:
   - Stop VideoAnnotator server or disconnect network
   - Refresh page → should show cached data marked as stale
   - Try "Refresh Catalog" → should show error but not crash

**Expected Results:**
- Server version appears (not "Unknown")
- Pipeline count > 0 (typically 4-8 pipelines)
- Feature flags show at least "Pipeline Catalog: true"
- Pipelines grouped and show metadata (version, model, description)

**Issues Found:**
```
[Document any pipeline discovery issues, include screenshots if possible]
```

### **1.2 Server Capability Detection**
- [ ] **Feature Flag Badges**: Verify colored badges in "Server Features" section
- [ ] **Version Detection**: Server version not "Unknown" in diagnostics
- [ ] **Pipeline Metadata**: Each pipeline shows version/model badges where available  
- [ ] **Capability Detection**: Individual pipeline capabilities properly detected
- [ ] **Fallback Mode**: Graceful degradation when server lacks v1.2.x endpoints

**Detailed Test Steps:**
1. **Feature Flags Verification**:
   - In Settings > Server Info, locate "Server Features" section
   - Expected badges: "Pipeline Catalog", "Parameter Schemas", "Pipeline Health", "Job SSE", "Artifact Listing"
   - Green/colored badges = enabled, gray badges = disabled
   - Most should be colored for v1.2.x servers

2. **Version Detection**:
   - Check "Server Version" field shows actual version (e.g., "1.2.1")
   - If shows "Unknown", server may not support introspection

3. **Pipeline Capability Details**:
   - In pipeline cards, look for version badges (e.g., "v1.0.2") 
   - Look for model badges (e.g., "YOLO11", "OpenFace3")
   - Verify descriptions are meaningful (not generic)

4. **Test Different Server Scenarios** (if possible):
   - **Full v1.2.x server**: All features enabled, rich metadata
   - **Limited server**: Some features disabled, basic info only
   - **Legacy server**: Fallback mode, minimal feature set

5. **Browser Console Check**:
   - Open Dev Tools Console
   - Look for successful API calls: `/api/v1/system/info`, `/api/v1/pipelines/catalog`
   - No red error messages related to pipeline discovery

**Expected Results:**
- At least "Pipeline Catalog" feature should be enabled
- Server version detected (not "Unknown")
- Most pipelines show version and/or model information
- No console errors for supported endpoints

**Issues Found:**
```
[Document capability detection issues, note which features are missing]
```

---

## 🔧 **SECTION 2: DYNAMIC JOB CREATION**

### **2.1 Pipeline Parameter Forms**
- [ ] **Job Creation Navigation**: Navigate to `/create/new` → reach Step 3 "Configure"
- [ ] **Dynamic Form Generation**: Parameter forms auto-generated from server schemas
- [ ] **Boolean Parameters**: Toggle switches for true/false parameters
- [ ] **Number Parameters**: Number inputs with min/max/step constraints  
- [ ] **Enum Parameters**: Dropdowns with server-provided options
- [ ] **Multiselect Parameters**: Checkbox groups for multi-value selection
- [ ] **String Parameters**: Text inputs for string values
- [ ] **Object Parameters**: JSON text areas for complex objects
- [ ] **Input Validation**: Invalid inputs show errors and block submission
- [ ] **Required Field Enforcement**: Missing required fields prevent submission

**Detailed Test Steps:**
1. **Start Job Creation Wizard**:
   - Navigate to `/create/new` or click "Create New Job" from jobs page
   - **Step 1 - Upload Videos**: Use test video from `demo/videos/3.mp4`
   - Click "Next" to proceed

2. **Step 2 - Select Pipelines**:
   - Verify pipeline list populated from server (not hardcoded)
   - Pipelines should be grouped: Face, Person, Audio, Scenes
   - Check pipeline descriptions are informative
   - Select 2-3 pipelines that have parameters (e.g., face_analysis, person_tracking)
   - Click "Next"

3. **Step 3 - Configure Parameters**:
   - Should see cards for each selected pipeline
   - Each card shows: pipeline name, description, version/model badges
   - **Test Parameter Types**:

   **Boolean Parameters** (look for switches):
   - Toggle switches should flip on/off
   - Label should clearly describe the parameter
   - Default values should match server schema

   **Number Parameters** (look for number inputs):
   - Try entering values outside min/max range
   - Use step values (e.g., if step=0.1, try 0.15, 0.2)
   - Should show validation errors for invalid ranges
   - Look for unit labels (e.g., "seconds", "pixels")

   **Enum Parameters** (look for dropdowns):
   - Click dropdown → should show server-provided options
   - Options should have labels, not just values
   - Select different options → value should update

   **Required Fields**:
   - Leave required fields empty
   - Try to proceed → should show validation errors
   - Fill required fields → errors should clear

4. **Test Form Validation**:
   - Enter invalid values (negative numbers, out of range)
   - Click "Next" or "Submit" → should show helpful error messages
   - Error messages should reference specific field names
   - Fix errors → submission should proceed

5. **Step 4 - Review & Submit**:
   - Verify configuration JSON reflects your parameter choices
   - Parameters for unselected pipelines should not appear
   - Click "Submit" → should create job successfully

**Test Data Files:**
- Use `demo/videos/3.mp4` or any MP4 file for testing
- Small files work better for quick testing

**Expected Parameter Types:**
- **Face Analysis**: confidence thresholds (number), enable features (boolean)
- **Person Tracking**: detection confidence (number), tracking mode (enum)
- **Audio Processing**: sample rate (number), language (enum)

**Issues Found:**
```
[Document parameter form issues, include parameter names and expected vs actual behavior]
```

### **2.2 Pipeline Selection Interface**
- [ ] **Dynamic Pipeline Loading**: Pipeline list populated from server (not static defaults)
- [ ] **Auto-Selection**: Pipelines with `defaultEnabled: true` pre-checked
- [ ] **Pipeline Grouping**: Pipelines organized by category with clear headers
- [ ] **Rich Descriptions**: Each pipeline shows informative description
- [ ] **Metadata Display**: Version and model badges visible where available
- [ ] **Selection Persistence**: Selections persist when navigating between steps
- [ ] **Loading States**: Appropriate loading indicators while fetching
- [ ] **Error Handling**: Graceful handling when server unavailable

**Detailed Test Steps:**
1. **Start Job Wizard**:
   - Navigate to `/create/new`
   - Upload any video file (use `demo/videos/3.mp4`)
   - Click "Next" to reach Step 2 "Select Pipelines"

2. **Verify Dynamic Loading**:
   - Should see loading spinner initially
   - Pipelines should populate from server (not hardcoded list)
   - If server offline, should show error with retry option

3. **Check Pipeline Organization**:
   - **Expected Groups**: Face, Person, Audio, Scenes (may vary by server)
   - Each group should have header and pipeline count
   - Pipelines within groups should be sorted alphabetically

4. **Verify Pipeline Information**:
   - **Checkbox + Name**: Each pipeline has checkbox and clear name
   - **Description**: Meaningful description (not generic text)
   - **Version Badge**: Shows version if available (e.g., "v1.0.2")
   - **Model Badge**: Shows model if available (e.g., "YOLO11", "OpenFace3")

5. **Test Selection Behavior**:
   - **Default Selection**: Some pipelines pre-checked (server's defaultEnabled)
   - **Toggle Selection**: Click checkboxes → selection changes
   - **Select All/None**: Use any bulk selection controls if available
   - Navigate to Step 3 and back → selections should persist

6. **Test Error Scenarios**:
   - **Server Offline**: Stop server → should show error, not crash
   - **Slow Network**: Throttle network → should show loading states
   - **Empty Response**: Should handle edge cases gracefully

7. **Performance Check**:
   - Pipeline loading should be fast (< 2 seconds)
   - No flickering or layout shifts during loading
   - Smooth interaction with checkboxes

**Expected Pipeline Examples:**
- **Face Group**: Face Analysis, Emotion Detection, Age Estimation
- **Person Group**: Person Detection, Pose Estimation, Activity Recognition  
- **Audio Group**: Speech Recognition, Speaker Identification
- **Scene Group**: Scene Detection, Object Detection

**UI Layout Expectations:**
```
┌─ Face (2 pipelines) ────────────────┐
│ ☑ Face Analysis v1.2 OpenFace3     │
│   Detect and analyze faces...       │
│ ☐ Emotion Detection v1.0           │
│   Classify facial emotions...       │
└─────────────────────────────────────┘
```

**Issues Found:**
```
[Document pipeline selection issues, note which pipelines appear and their grouping]
```

---

## 👁️ **SECTION 3: CAPABILITY-AWARE VIEWER**

### **3.1 OpenFace3 Controls Integration**
- [ ] **Load Demo Data**: Load demo data with OpenFace3 results successfully
- [ ] **Server Capability Detection**: Controls adapt to server-declared capabilities
- [ ] **Job Pipeline Awareness**: Controls reflect which pipelines were actually run
- [ ] **Feature Availability Indicators**: Clear visual coding for available/unavailable
- [ ] **Informative Tooltips**: Hover messages explain why features are disabled
- [ ] **Color-Coded Labels**: Green for available, gray for unavailable features
- [ ] **Master Toggle Logic**: Master OpenFace3 toggle controls child features appropriately

**Detailed Test Steps:**
1. **Load Demo Data with Face Results**:
   - Navigate to main viewer page (`/`)
   - Click "Get Started" → "Load Demo Data"
   - Select demo with face data: **"2UWdXP.joke1.rep2.take1.Peekaboo_h265"**
   - Or use browser console: `window.debugUtils.testAllDatasets()` → select one with faces

2. **Access OpenFace3 Controls**:
   - In right sidebar, look for "OpenFace3 Controls" panel
   - Should be expanded by default when face data is available
   - Look for master "OpenFace3" toggle at top

3. **Test Server Capability Awareness**:
   - **With v1.2.x Server**: All supported features should be available
   - **Without Server/Legacy**: Should fall back to data-based availability
   - Check browser console for pipeline catalog API calls

4. **Verify Feature Availability Indicators**:
   - **Available Features** (green labels):
     - "Landmarks (98pts)" - if landmark data exists
     - "Face Boxes" - if face detection data exists  
     - "Emotions (8 types)" - if emotion data exists
   - **Unavailable Features** (gray labels):
     - Features without data should be grayed out
     - Features not supported by server should be disabled

5. **Test Tooltip Information**:
   - **Hover over disabled toggles** → should show helpful tooltip:
     - "Server doesn't support OpenFace3" (no server capability)
     - "Face analysis not run for this job" (pipeline not executed)
     - "No landmark data available" (data missing)
   - **Available toggles** → no tooltip or positive message

6. **Test Toggle Functionality**:
   - **Master Toggle**: Turn off → all child features should disable
   - **Individual Features**: Toggle on/off → only affects that feature
   - **Auto-Enable Logic**: Turn on child feature → master should auto-enable
   - **Visual Feedback**: Changes should be immediate in video player

7. **Test Different Data Scenarios**:
   - **Rich Face Data**: Demo with all OpenFace3 features → most toggles available
   - **Limited Face Data**: Demo with only basic detection → fewer toggles available
   - **No Face Data**: Demo without faces → all toggles disabled with explanations

**Demo Data Files with Face Results:**
- `2UWdXP.joke1.rep2.take1.Peekaboo_h265` - Good face detection data
- `3dC3SQ.joke1.rep1.take1.TearingPaper_h265` - Multiple faces
- Use `window.debugUtils.testAllDatasets()` to see all available demos

**Expected Control Layout:**
```
┌─ OpenFace3 Controls ────────────────┐
│ ☑ OpenFace3                        │
│ ──────────────────────────────────── │
│ Core Features:                      │
│ ☑ Landmarks (98pts)     ☑ Face Boxes│
│ ☑ Emotions (8 types)    ☐ Action Units│
│ ──────────────────────────────────── │
│ Advanced Features:                  │
│ ☐ Head Pose (3D)       ☐ Gaze Direction│
└─────────────────────────────────────┘
```

**Color Coding Legend:**
- **Green text**: Feature available and can be toggled
- **Gray text**: Feature unavailable (hover for reason)
- **Switch enabled**: Feature is currently active
- **Switch disabled**: Feature cannot be activated

**Issues Found:**
```
[Document viewer capability issues, include which features show as available/unavailable]
```

### **3.2 Complete Overlay System Testing**
- [ ] **Data-Driven Availability**: All overlay toggles reflect actual data availability
- [ ] **Consistent Messaging**: Clear "Not available" messages across all controls
- [ ] **Visual Consistency**: Similar styling for available/unavailable states
- [ ] **Logical Grouping**: Related features grouped appropriately
- [ ] **Performance Impact**: Toggling overlays doesn't cause lag or crashes

**Detailed Test Steps:**
1. **Test Multiple Demo Datasets**:
   - Load different demos with varying data types
   - **Rich Dataset**: `2UWdXP.joke1.rep2.take1.Peekaboo_h265` (faces, poses, audio)
   - **Basic Dataset**: One with limited annotation types
   - **Audio-Only**: Dataset with transcripts but no visual annotations

2. **Verify All Overlay Controls**:
   - **Main Controls Panel** (left sidebar):
     - "Pose Keypoints" toggle
     - "Subtitles" toggle  
     - "Speaker Identification" toggle
     - "Scene Boundaries" toggle
     - "Face Detection" toggle
     - "Emotion Overlays" toggle

3. **Test Availability Logic**:
   - **With Data**: Toggle should be enabled, colored normally
   - **Without Data**: Toggle should be disabled, grayed out
   - **Partial Data**: Some features available, others not

4. **Check Messaging Consistency**:
   - Disabled toggles should show consistent messaging
   - Look for "Not available for this video" or similar
   - Messages should be helpful, not technical

5. **Test Visual Feedback**:
   - **Toggle On**: Overlay appears immediately on video
   - **Toggle Off**: Overlay disappears immediately  
   - **Multiple Overlays**: Can enable multiple simultaneously
   - **No Conflicts**: Overlays don't interfere with each other

6. **Performance Testing**:
   - Toggle multiple overlays rapidly → should be responsive
   - Scrub timeline with overlays on → should stay smooth
   - Play video with all overlays → acceptable performance

7. **Test Edge Cases**:
   - **No Annotation Data**: All overlays disabled with explanation
   - **Corrupted Data**: Graceful handling of bad data
   - **Large Datasets**: Performance with many detected objects

**Expected Overlay Types:**
- **Pose Keypoints**: Skeleton overlays on detected people
- **Face Boxes**: Rectangles around detected faces
- **Emotion Labels**: Text labels showing detected emotions
- **Subtitles**: WebVTT transcript display
- **Speaker Labels**: RTTM speaker identification
- **Scene Markers**: Visual indicators of scene changes

**Visual States to Check:**
- **Available + Off**: Normal checkbox, can be toggled
- **Available + On**: Checked checkbox, overlay visible on video  
- **Unavailable**: Grayed checkbox, explanatory text, disabled

**Issues Found:**
```
[Document overlay behavior issues, note which data types are missing or not working]
```

---

## ⚡ **SECTION 4: PERFORMANCE & ERROR HANDLING**

### **4.1 Loading States & Performance**
- [ ] **Settings Loading**: Appropriate spinners/skeletons while fetching pipeline catalog
- [ ] **Job Creation Loading**: Loading indicators during schema fetch and validation
- [ ] **Fast Response Times**: Pipeline operations complete within reasonable time
- [ ] **Smooth Interactions**: No blocking UI during background operations
- [ ] **Memory Efficiency**: No memory leaks during repeated operations

**Detailed Test Steps:**
1. **Test Loading States**:
   - **Settings Page**: 
     - Navigate to `/create/settings` → should show skeleton loading
     - Watch for smooth transition from loading to content
     - No layout shift when content loads
   
   - **Job Creation**:
     - Start new job → Step 2 should show pipeline loading
     - Step 3 parameter forms should load smoothly
     - Submission should show progress indicators

2. **Performance Benchmarks**:
   - **Pipeline Catalog Load**: < 2 seconds for typical server
   - **Parameter Schema Fetch**: < 1 second per pipeline
   - **Job Submission**: < 5 seconds for small video
   - **Overlay Toggles**: Immediate response (< 100ms)

3. **Test Under Slow Conditions**:
   - Use browser Dev Tools → Network → Throttling → "Slow 3G"
   - All operations should still work, just slower
   - Loading states should be visible and helpful
   - No timeouts or failures under normal slow conditions

4. **Memory Usage Testing**:
   - Open browser Task Manager (Shift+Esc in Chrome)
   - Navigate through app extensively
   - Memory usage should be reasonable (< 200MB typical)
   - No significant memory growth over time

**Browser Dev Tools Checklist:**
1. **Console Tab**: Should be clean (no red errors)
2. **Network Tab**: API calls should succeed (200 status)
3. **Performance Tab**: No long tasks blocking UI
4. **Memory Tab**: Reasonable memory usage patterns

**Issues Found:**
```
[Document performance issues, include specific timing measurements]
```

### **4.2 Error Recovery & Resilience**
- [ ] **Network Failures**: Clear, actionable error messages for connection issues
- [ ] **Server Errors**: Graceful handling of 404/500/timeout responses
- [ ] **Validation Errors**: Specific, helpful messages for form validation failures
- [ ] **Graceful Degradation**: App remains usable when advanced features unavailable
- [ ] **Recovery Mechanisms**: Retry options and manual refresh capabilities
- [ ] **Error Boundaries**: No white screen crashes, proper error pages

**Detailed Test Steps:**
1. **Network Error Testing**:
   - **Disconnect Internet**: Verify app shows meaningful offline message
   - **Stop VideoAnnotator Server**: 
     - Settings should show "Server unreachable" or similar
     - Cached data should still be available
     - Retry mechanisms should be offered
   
   - **Slow/Unstable Network**:
     - Use Dev Tools → Network → Add throttling
     - Operations should eventually succeed or fail gracefully
     - No infinite loading states

2. **Server Error Simulation**:
   - **Wrong URL**: Change API URL to invalid address → clear error message
   - **Wrong Token**: Use invalid token → authentication error message
   - **Server Errors**: If possible, simulate 500 errors → app shouldn't crash

3. **Form Validation Testing**:
   - **Job Creation Parameters**:
     - Enter invalid numbers (negative, too large) → clear field-specific errors
     - Leave required fields empty → helpful "This field is required" messages
     - Enter malformed JSON in object fields → JSON parsing error message
   
   - **Error Message Quality Check**:
     - Messages should be specific ("Value must be between 0 and 1")
     - Avoid technical jargon ("API call failed")
     - Include suggested actions ("Please check your network connection")

4. **Fallback Mode Testing**:
   - **No Pipeline Catalog**: App should work with static pipeline list
   - **No Parameter Schemas**: Should show basic configuration options
   - **No Server Info**: Should show "Unknown" but continue functioning

5. **Recovery Testing**:
   - **After Network Restored**: 
     - Manual refresh should work
     - Auto-retry should kick in for failed operations
     - Fresh data should load properly
   
   - **After Fixing Errors**:
     - Correct form values → errors should clear immediately
     - Valid configuration → submission should proceed

6. **Error Boundary Testing**:
   - Try to trigger JavaScript errors (invalid data, edge cases)
   - App should show error page, not blank screen
   - Error reporting should be helpful for debugging

**Error Message Examples (Good vs Bad):**
```
✅ Good: "Connection failed. Please check that VideoAnnotator server is running at http://localhost:18011"
❌ Bad: "Network error"

✅ Good: "Value must be between 0.0 and 1.0 for confidence threshold"
❌ Bad: "Invalid input"

✅ Good: "Face analysis pipeline not available. Enable in server configuration."
❌ Bad: "Pipeline error"
```

**Test Scenarios to Force Errors:**
- Change API URL to `http://localhost:9999` (nonexistent)
- Set invalid token like `invalid-token-123`
- Enter parameter values outside allowed ranges
- Upload unsupported file types
- Attempt operations with server stopped

**Issues Found:**
```
[Document error handling issues, include specific error messages and user experience problems]
```

---

## 🔄 **SECTION 5: INTEGRATION & REGRESSION**

### **5.1 Core Functionality Preservation**
- [ ] **Demo Data Loading**: All existing demo datasets load without issues
- [ ] **File Upload System**: Video and annotation file upload works unchanged
- [ ] **Video Playback**: Play/pause/scrub functionality intact
- [ ] **Timeline Navigation**: Timeline scrubbing and controls functional  
- [ ] **Data Export**: Export features work for all annotation types
- [ ] **Debug Console**: Browser console utilities still functional
- [ ] **Navigation**: All routing and page navigation works

**Detailed Test Steps:**
1. **Demo Data Regression Test**:
   - Navigate to main page (`/`)
   - Click "Get Started" → "Load Demo Data"
   - **Test Each Demo Dataset**:
     - `2UWdXP.joke1.rep2.take1.Peekaboo_h265`
     - `3dC3SQ.joke1.rep1.take1.TearingPaper_h265`  
     - `4JDccE.joke5.rep2.take1.TearingPaper_h265`
     - (And all others in demo list)
   - Each should load video + annotations successfully
   - No console errors during loading

2. **File Upload Testing**:
   - **Video Upload**: Drag/drop or select video file → should load and play
   - **Annotation Upload**: Upload existing .json annotation files
   - **Multi-file Upload**: Video + annotations together
   - **Error Handling**: Invalid file types should show appropriate errors

3. **Core Playback Features**:
   - **Play/Pause**: Space bar and click controls work
   - **Timeline Scrubbing**: Click and drag on timeline
   - **Speed Controls**: Playback rate adjustment (0.5x, 1x, 2x, etc.)
   - **Volume Controls**: Audio level adjustment
   - **Fullscreen**: Fullscreen mode works properly

4. **Timeline and Navigation**:
   - **Timeline Display**: Shows duration and current position
   - **Annotation Markers**: Timeline shows annotation events
   - **Jump to Events**: Click timeline annotations → video jumps
   - **Keyboard Controls**: Arrow keys for frame-by-frame navigation

5. **Export Functionality**:
   - **Export Options**: Should still have export buttons/menus
   - **Data Formats**: JSON, CSV, or other export formats work
   - **Annotation Export**: Can export detected annotations
   - **Video Export**: If available, video with overlays export

6. **Debug Console Tools**:
   - Open browser console (`F12`)
   - **Test Debug Commands**:
     ```javascript
     // These should work without errors
     window.debugUtils.testAllDatasets()
     VideoAnnotatorDebug.runAllTests()
     window.debugUtils.loadDemoDataset('2UWdXP.joke1.rep2.take1.Peekaboo_h265')
     ```

7. **Navigation and Routing**:
   - **Main Navigation**: All navigation links work
   - **URL Navigation**: Direct URLs load correctly
   - **Browser Back/Forward**: History navigation works
   - **404 Handling**: Invalid URLs show proper 404 page

**Key Demo Datasets to Test:**
- **Rich Multi-modal**: `2UWdXP.joke1.rep2.take1.Peekaboo_h265` (faces, poses, audio)
- **Multiple People**: `3dC3SQ.joke1.rep1.take1.TearingPaper_h265` 
- **Audio-Heavy**: Any dataset with good transcript/speaker data
- **Basic Detection**: Simple object/scene detection datasets

**Console Commands to Verify:**
```javascript
// Check debug utilities are available
typeof window.debugUtils !== 'undefined'

// Check VideoAnnotator debug tools
typeof VideoAnnotatorDebug !== 'undefined'

// Load specific dataset
window.debugUtils.loadDemoDataset('2UWdXP.joke1.rep2.take1.Peekaboo_h265')

// Test all datasets
window.debugUtils.testAllDatasets()
```

**Regression Issues:**
```
[List any v0.3.x features that stopped working, include specific functionality]
```

### **5.2 Cross-Browser Compatibility Testing**
- [ ] **Chrome/Chromium**: Full functionality in latest Chrome (v118+)
- [ ] **Firefox**: Complete feature parity in Firefox (v118+)  
- [ ] **Safari**: Core features work in Safari (v16+)
- [ ] **Edge**: Microsoft Edge compatibility maintained (v118+)
- [ ] **Mobile Browsers**: Basic functionality on mobile devices

**Detailed Test Steps:**
1. **Test Each Browser Systematically**:
   - Install/update to latest stable versions
   - Clear cache and cookies before testing  
   - Test same scenarios across all browsers

2. **Core Feature Parity Check**:
   - **Pipeline Discovery**: Settings page loads pipeline catalog
   - **Job Creation**: Wizard works through all steps
   - **Parameter Forms**: All input types render and function
   - **Video Playback**: Play/pause/scrub works smoothly
   - **Overlay Toggles**: All overlay controls functional

3. **UI Rendering Consistency**:
   - **Layout**: No major layout breaks or misalignments
   - **Fonts**: Text renders clearly and consistently
   - **Colors**: Color scheme consistent across browsers
   - **Responsive**: Mobile/tablet layouts work properly
   - **Icons**: All icons display correctly (Lucide icons)

4. **JavaScript API Compatibility**:
   - **Fetch API**: Network requests work (no IE11 issues)
   - **Modern JS**: ES6+ features supported
   - **React**: No React-specific browser issues
   - **LocalStorage**: Settings persistence works

5. **Video/Media Handling**:
   - **Video Codecs**: MP4/WebM support across browsers
   - **Audio Playback**: Audio tracks play properly
   - **Canvas Rendering**: Overlay graphics render correctly
   - **Performance**: Acceptable performance in all browsers

6. **Known Browser Differences to Check**:
   - **Safari**: May have stricter video codec requirements
   - **Firefox**: May handle canvas rendering differently
   - **Mobile Safari**: Touch interactions, viewport issues
   - **Edge**: Generally Chrome-compatible but verify

**Browser Version Requirements:**
- **Chrome**: v118+ (supports latest Web APIs)
- **Firefox**: v118+ (equivalent feature support)
- **Safari**: v16+ (modern JavaScript support)  
- **Edge**: v118+ (Chromium-based)

**Mobile Testing** (if available):
- **iOS Safari**: Basic functionality on iPhone/iPad
- **Chrome Mobile**: Android Chrome browser
- **Responsive Design**: App adapts to mobile screen sizes

**Critical Features to Verify Per Browser:**
1. Settings page loads and displays pipeline data
2. Job creation wizard completes successfully  
3. Video playback with overlays works
4. Pipeline parameter forms render correctly
5. No console errors in developer tools

**Browser Issues:**
```
[Document browser-specific problems, include browser version and specific functionality affected]
```

---

## ✅ **FINAL VERIFICATION & INTEGRATION TESTING**

### **6.1 End-to-End Workflow Testing**
- [ ] **Complete Job Lifecycle**: Create job → Monitor progress → View results
- [ ] **Settings to Job to Viewer**: Flow from server diagnostics through complete workflow
- [ ] **Cache Consistency**: Pipeline changes reflect across all app areas
- [ ] **Data Persistence**: Settings and selections persist across browser sessions

**Detailed Test Steps:**
1. **Full Workflow Test**:
   - Start at Settings → verify server connection and pipeline catalog
   - Create new job → use dynamic pipelines → configure parameters → submit
   - Monitor job progress (if SSE available)
   - Open completed job results → verify capability-aware overlays
   - Check that overlays match job's selected pipelines

2. **Cross-Component Consistency**:
   - Change server settings → verify job creation reflects changes
   - Clear pipeline cache → all areas should refresh consistently
   - Pipeline updates should appear in both settings and job creation

**Issues Found:**
```
[Document end-to-end workflow issues]
```

### **6.2 Automated Test Verification**
- [ ] **Unit Tests Pass**: Run `bun run test:run` → all tests pass
- [ ] **E2E Tests Pass**: Run `bun run e2e` → smoke tests pass
- [ ] **Build Success**: Run `bun run build` → clean build without errors
- [ ] **Type Checking**: Run `bunx tsc --noEmit` → no TypeScript errors

**Commands to Run:**
```bash
# Unit tests
bun run test:run

# E2E tests  
bun run e2e

# Production build
bun run build

# Type checking
bunx tsc --noEmit

# Lint check
bun run lint
```

**Results:**
```
[Document test results and any failures]
```

---

## 🎯 **CRITICAL PIPELINE INTEGRATION FEATURES**

### **Must-Pass Checklist**
- [ ] **Dynamic Pipeline Discovery**: Settings show server-fetched pipelines
- [ ] **Parameter Form Generation**: Job wizard creates forms from server schemas
- [ ] **Server Capability Detection**: Features adapt to server capabilities
- [ ] **Capability-Aware Overlays**: Viewer controls reflect pipeline availability
- [ ] **Graceful Fallback**: App works with limited/no server introspection
- [ ] **Error Recovery**: Clear error messages and retry mechanisms

### **Quality Standards Verification**
- [ ] **No Critical Regressions**: All v0.3.x core features work
- [ ] **Performance Standards**: Operations complete within acceptable time
- [ ] **Error Handling Quality**: User-friendly error messages throughout
- [ ] **UI/UX Consistency**: Consistent design patterns and interactions
- [ ] **Cross-Browser Compatibility**: Works in all major browsers

---

## 📝 **COMPREHENSIVE TESTING SUMMARY**

### **Testing Environment Used**
- **VideoAnnotator Server Version**: ___________
- **Browser(s) Tested**: ___________
- **OS**: ___________
- **Testing Duration**: ___________

### **Pipeline Integration Features Verified**
```
✅ Working Features:
- [ ] Dynamic pipeline catalog loading
- [ ] Server capability detection  
- [ ] Parameter form generation
- [ ] Capability-aware overlays
- [ ] Error handling and fallback
- [ ] Cache management
- [ ] (List all working features)

❌ Issues Found:
- [ ] (List any broken or problematic features)
```

### **Regression Testing Results**
```
✅ Preserved v0.3.x Features:
- [ ] Demo data loading
- [ ] Video playback
- [ ] File upload
- [ ] Export functionality
- [ ] Debug tools
- [ ] (List all preserved features)

❌ Broken v0.3.x Features:
- [ ] (List any regressions)
```

### **Performance Assessment**
```
- Pipeline catalog loading: _____ seconds
- Job creation workflow: _____ seconds  
- Overlay toggle response: _____ milliseconds
- Memory usage: _____ MB
- Overall performance: Acceptable / Needs improvement
```

### **Critical Issues Found**
```
Priority 1 (Blocking):
- [ ] (List any issues that prevent release)

Priority 2 (Important):  
- [ ] (List issues that should be fixed before release)

Priority 3 (Minor):
- [ ] (List nice-to-have improvements)
```

### **Browser Compatibility Results**
```
✅ Fully Compatible: (List browsers)
⚠️  Minor Issues: (List browsers with minor issues)
❌ Major Issues: (List browsers with major problems)
```

### **Recommendations for Release**
```
Ready for Release: Yes / No / With conditions

Conditions (if any):
- [ ] Fix critical issue X
- [ ] Address performance concern Y
- [ ] Update documentation for Z

Post-Release Monitoring:
- [ ] Monitor server load with new pipeline features
- [ ] Watch for user feedback on parameter forms
- [ ] Track performance metrics in production
```

---

## 🚀 **FINAL SIGN-OFF**

### **Release Readiness Checklist**
- [ ] **All Critical Features Work**: Pipeline integration fully functional
- [ ] **No Blocking Regressions**: Core app functionality preserved
- [ ] **Quality Standards Met**: Performance, usability, and reliability acceptable
- [ ] **Cross-Browser Tested**: Works in all supported browsers
- [ ] **Error Handling Verified**: Graceful failure modes and recovery
- [ ] **Documentation Updated**: User and developer docs reflect v0.4.0 changes

### **QA Approval**
- [ ] **Functional Testing Complete**: All test scenarios executed
- [ ] **Integration Testing Passed**: End-to-end workflows verified
- [ ] **Performance Testing Acceptable**: No significant performance regressions
- [ ] **User Experience Approved**: Interface changes enhance usability

**QA Tester Signature:** _________________ **Date:** _________  
**Release Recommendation:** ✅ Approved / ⚠️ Conditional / ❌ Not Ready

---

**Comprehensive Test Guide Version:** v0.4.0  
**Total Test Items:** 65+ verification points  
**Estimated Testing Time:** 6-8 hours for complete testing  
**Focus:** Pipeline integration, capability awareness, and regression prevention

**Quick Test (2 hours)**: Sections 1, 2, 3.1, 5.1, and Final Verification  
**Full Test (6-8 hours)**: All sections with detailed verification