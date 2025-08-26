# Video Annotation Viewer v0.3.1 - Focused QA Checklist

## 🎯 **OVERVIEW**

This is a **focused testing checklist** for v0.3.1 bug fixes and minor improvements. Unlike the comprehensive v0.3.0 checklist, this focuses only on the specific fixes implemented and critical regression testing.

**Testing Date:** ___________  
**Tester:** ___________  
**Browser:** ___________  
**OS:** ___________

### Testing checkbox format
- [ ] unchecked boxes for tests that need doing
- [x] ticked boxes for passed tests ✅ 2025-08-25
- [f] f is for fail
      with explanatory comments below

---

## 🚨 **SECTION 1: CRITICAL BUG FIXES**

### **1.1 Configuration Display Fix**
- [ ] **Navigate to**: `/create/new` → Step 3 (Configure Pipelines)
- [ ] **Text Visibility**: JSON configuration text is readable (not white-on-white)
- [ ] **Dark Mode**: Text remains visible in dark theme
- [ ] **Light Mode**: Text remains visible in light theme  
- [ ] **Editing**: Can modify JSON values without visibility issues

**Test Steps:**
1. Start job creation wizard
2. Upload any video file
3. Select pipelines  
4. Verify Step 3 configuration display
5. Toggle between light/dark themes (if available)

**Issues Found:**
```
[Document any remaining visibility issues]
```

### **1.2 Console Error Reduction**
- [ ] **React Router Warnings**: No future flag warnings in browser console
- [ ] **SSE Connection**: No infinite retry errors when API unavailable  
- [ ] **Clean Console**: Major reduction in console spam/errors
- [ ] **Error Handling**: Graceful degradation when endpoints missing

**Test Steps:**
1. Open browser dev tools → Console tab
2. Navigate through all major pages: `/`, `/create`, `/create/new`, `/create/jobs`
3. Monitor console for recurring warnings/errors
4. Test with API server off (stop VideoAnnotator server)

**Issues Found:**
```
[List any remaining console errors or warnings]
```

---

## 🔧 **SECTION 2: STABILITY IMPROVEMENTS**

### **2.1 Job Detail Page Error Handling**
- [ ] **Page Load**: `/create/jobs/any-id` doesn't crash with error boundary
- [ ] **Missing Job**: Graceful handling of non-existent job IDs
- [ ] **Navigation**: Can navigate back from job detail without crashes
- [ ] **Error Display**: Clear error message when job data unavailable

**Test Steps:**
1. Navigate to `/create/jobs/fake-job-id`
2. Navigate to `/create/jobs/` (without ID)  
3. Try various invalid job ID formats
4. Ensure no React error boundaries triggered

**Issues Found:**
```
[Document any crashes or unhandled errors]
```

### **2.2 Pipeline Information Enhancement**
- [ ] **Descriptions Present**: Each pipeline shows descriptive text
- [ ] **Scene Detection**: Clear explanation of functionality
- [ ] **Person Tracking**: Mentions YOLO11 + ByteTrack technology
- [ ] **Face Analysis**: Describes OpenFace3 capabilities  
- [ ] **Audio Processing**: Explains speech + speaker identification

**Test Steps:**
1. Start new job creation
2. Reach Step 2 (Select Pipelines)
3. Verify each pipeline checkbox has helpful description
4. Descriptions help user understand what each pipeline does

**Issues Found:**
```
[Note any missing or unclear descriptions]
```

---

## 🔑 **SECTION 3: TOKEN SETUP IMPROVEMENTS**

### **3.1 User Guidance Enhancement**
- [ ] **Setup Instructions**: Clear steps for obtaining/configuring token
- [ ] **Validation Feedback**: Immediate feedback when testing token
- [ ] **Error Messages**: Specific guidance when authentication fails
- [ ] **Success State**: Clear confirmation when token works

**Test Steps:**
1. Navigate to `/create/settings`
2. Try configuring API token
3. Test with invalid token
4. Test with valid token (if available)
5. Verify feedback quality

**Issues Found:**
```
[Document any confusing or missing guidance]
```

---

## 📋 **SECTION 4: REGRESSION TESTING**

### **4.1 Core Functionality** (Quick Smoke Test)
- [ ] **Home Page**: Main viewer loads without errors
- [ ] **File Upload**: Can select and upload video files  
- [ ] **Demo Data**: Debug utils and demo loading still work
- [ ] **Video Playback**: Basic play/pause functionality intact
- [ ] **Navigation**: All main navigation links work

### **4.2 Job Creation Flow** (Critical Path)
- [ ] **Step 1**: Video file selection works
- [ ] **Step 2**: Pipeline selection functional  
- [ ] **Step 3**: Configuration display and editing works
- [ ] **Step 4**: Review and submit process intact
- [ ] **Flow Navigation**: Previous/Next buttons throughout wizard

### **4.3 Debug Tools** (Developer Features)
- [ ] **Console Access**: `VideoAnnotatorDebug.runAllTests()` still works
- [ ] **Demo Data**: `window.debugUtils.testAllDatasets()` functional
- [ ] **API Testing**: Debug utilities connect to API correctly

**Regression Issues:**
```
[List any v0.3.0 features that stopped working]
```

---

## 📊 **SECTION 5: PERFORMANCE CHECK**

### **5.1 Loading Performance** (Quick Verification)
- [ ] **Page Load**: No noticeable slowdown from fixes
- [ ] **Navigation**: Routing performance unchanged
- [ ] **Memory**: No obvious memory leaks from error handling changes

**Performance Issues:**
```
[Note any performance regressions]
```

---

## ✅ **FINAL VERIFICATION**

### **Critical Issues Resolution**
- [ ] White-on-white text fixed in configuration
- [ ] Console error spam significantly reduced
- [ ] Job detail page crashes eliminated  
- [ ] Pipeline descriptions improve user understanding
- [ ] Token setup provides better guidance

### **No New Issues Introduced**
- [ ] All v0.3.0 functionality preserved
- [ ] No new errors or warnings introduced
- [ ] Performance remains acceptable
- [ ] User workflows still complete successfully

---

## 📝 **TESTING SUMMARY**

### **Issues Fixed Successfully**
```
[List confirmed fixes]
```

### **Remaining Issues**
```
[List any unfixed issues]
```

### **New Issues Found**
```
[List any new problems introduced]
```

### **Recommendations**
```
[Any suggestions for future improvements]
```

---

## 🚀 **SIGN-OFF**

- [ ] **All Critical Fixes Verified**: Configuration display, console errors, page stability
- [ ] **No Regressions**: Core v0.3.0 functionality preserved  
- [ ] **Quality Acceptable**: Improvement in user experience confirmed
- [ ] **Ready for Release**: v0.3.1 approved for deployment

**QA Tester Signature:** _________________ **Date:** _________

---

**Checklist Version:** v0.3.1  
**Total Test Items:** 25 focused verification points  
**Estimated Testing Time:** 2-3 hours  
**Focus:** Bug fixes and critical regression testing only