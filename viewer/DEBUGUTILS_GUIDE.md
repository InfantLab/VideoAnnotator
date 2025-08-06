# Debug Utils Guide

This guide explains the debugging and testing utilities available in the Video Action Viewer console.

## Overview

The Video Action Viewer provides comprehensive debugging utilities accessible through the browser console via `window.debugUtils`. These utilities are automatically loaded when the application starts and are designed for developers to test, debug, and inspect the VideoAnnotator pipeline data loading system.

## Available Functions

### Dataset Management

#### `window.debugUtils.listDatasets()`
Lists all available demo datasets with their file paths.

```javascript
window.debugUtils.listDatasets()
```

**Output**: Console listing of all datasets with video, data, speech, and speaker file paths.

#### `window.debugUtils.DEMO_DATA_SETS`
Direct access to the raw dataset configuration object.

```javascript
console.log(window.debugUtils.DEMO_DATA_SETS)
```

### Data Loading (Console Testing Only)

> ⚠️ **Important**: These functions load data for console inspection only. They do **NOT** update the UI. To view datasets in the application, use the demo buttons in the FileUploader or the "View Demo" button.

#### `window.debugUtils.loadDemoAnnotations(datasetKey)`
Loads annotation data for console testing and debugging.

```javascript
// Load annotations only
const annotations = await window.debugUtils.loadDemoAnnotations('peekaboo-rep3-v1.1.1')
console.log(annotations)
```

**Returns**: `StandardAnnotationData` object or `null` if loading fails.

#### `window.debugUtils.loadDemoVideo(datasetKey)`
Loads video file for console testing.

```javascript
// Load video file only  
const videoFile = await window.debugUtils.loadDemoVideo('peekaboo-rep3-v1.1.1')
console.log(videoFile.name, videoFile.size)
```

**Returns**: `File` object or `null` if loading fails.

#### `window.debugUtils.loadDemoDataset(datasetKey)`
Loads both video and annotation data for console testing.

```javascript
// Load complete dataset (console only)
await window.debugUtils.loadDemoDataset('peekaboo-rep3-v1.1.1')
```

**Returns**: Promise that resolves when loading is complete. Check console for detailed loading information.

### Testing and Quality Assurance

#### `window.debugUtils.testAllDatasets()`
Runs comprehensive tests on all available datasets.

```javascript
// Test all datasets for loading issues
const results = await window.debugUtils.testAllDatasets()
console.log(results)
```

**Returns**: Object with test results for each dataset (`✅ SUCCESS` or `❌ FAILED`).

#### `window.debugUtils.checkDataIntegrity(datasetKey)`
Performs detailed integrity checks on a specific dataset.

```javascript
// Check integrity of specific dataset
const report = await window.debugUtils.checkDataIntegrity('peekaboo-rep2-v1.1.1')
console.log(report)
```

**Returns**: Object with `{ valid: boolean, issues: string[] }`.

**Checks performed**:
- JSON validity of `complete_results.json`
- File accessibility for all dataset files
- Structure validation of VideoAnnotator v1.1.1 format
- Detailed error reporting for malformed files

## Available Datasets

The following VideoAnnotator v1.1.1 demo datasets are available:

| Dataset Key | Description | Status |
|-------------|-------------|--------|
| `peekaboo-rep3-v1.1.1` | Peekaboo joke, repetition 3 | ✅ Working |
| `peekaboo-rep2-v1.1.1` | Peekaboo joke, repetition 2 | ⚠️ Malformed JSON (test case) |
| `tearingpaper-rep1-v1.1.1` | Tearing paper joke, repetition 1 | ✅ Working |
| `thatsnotahat-rep1-v1.1.1` | "That's not a hat" joke, repetition 1 | ✅ Working |

## Example Usage Scenarios

### 1. Quick Dataset Overview
```javascript
// See what datasets are available
window.debugUtils.listDatasets()

// Test all datasets quickly
await window.debugUtils.testAllDatasets()
```

### 2. Debug Specific Loading Issues
```javascript
// Check integrity of problematic dataset
await window.debugUtils.checkDataIntegrity('peekaboo-rep2-v1.1.1')

// Load and inspect annotation structure
const data = await window.debugUtils.loadDemoAnnotations('thatsnotahat-rep1-v1.1.1')
console.log('Pipelines:', data?.metadata?.pipelines)
console.log('Person tracking entries:', data?.person_tracking?.length)
```

### 3. Performance Testing
```javascript
// Time the loading of a specific dataset
console.time('Dataset Load')
await window.debugUtils.loadDemoDataset('tearingpaper-rep1-v1.1.1')
console.timeEnd('Dataset Load')
```

### 4. Data Structure Inspection
```javascript
// Load data and inspect VideoAnnotator v1.1.1 structure
const data = await window.debugUtils.loadDemoAnnotations('peekaboo-rep3-v1.1.1')

console.log('Video info:', data.video_info)
console.log('Processing config:', data.metadata.processing_config)
console.log('Pipeline results:', {
  person_tracking: data.person_tracking?.length,
  face_analysis: data.face_analysis?.length,
  speech_recognition: data.speech_recognition?.length,
  speaker_diarization: data.speaker_diarization?.length,
  scene_detection: data.scene_detection?.length
})
```

## UI Integration vs Console Testing

### Console Functions (Debug Only)
- `loadDemoAnnotations()`, `loadDemoVideo()`, `loadDemoDataset()`
- **Purpose**: Testing, debugging, data inspection
- **UI Impact**: None - data loaded to console only
- **Use Case**: Development, QA testing, troubleshooting

### UI Functions (Actual Viewing)
- **Demo buttons in FileUploader**: Four dataset selection buttons
- **"View Demo" button**: Quick demo loading (uses first dataset)
- **Manual file upload**: Drag & drop or file selection
- **Purpose**: Actual application usage
- **UI Impact**: Loads data into viewer interface

## Error Handling and Troubleshooting

The debug utilities include comprehensive error handling:

### JSON Parsing Errors
When `complete_results.json` is malformed (like in `peekaboo-rep2-v1.1.1`):
```javascript
await window.debugUtils.checkDataIntegrity('peekaboo-rep2-v1.1.1')
// Shows: JSON parse error with first 300 characters of malformed content
```

### Network Issues
When files are inaccessible:
```javascript
await window.debugUtils.testAllDatasets()
// Shows: File accessibility status for each dataset component
```

### Data Structure Issues  
When VideoAnnotator format is unexpected:
```javascript
const data = await window.debugUtils.loadDemoAnnotations('dataset-key')
// Console shows: Pipeline detection, structure validation, entry counts
```

## Development Workflow

### 1. Initial Setup Testing
```javascript
// Verify all datasets load correctly
await window.debugUtils.testAllDatasets()
```

### 2. Feature Development
```javascript
// Load test data for feature development
const testData = await window.debugUtils.loadDemoAnnotations('peekaboo-rep3-v1.1.1')
// Use testData for testing new features in console
```

### 3. Quality Assurance
```javascript
// Check data integrity before releases
for (const key of Object.keys(window.debugUtils.DEMO_DATA_SETS)) {
  const report = await window.debugUtils.checkDataIntegrity(key)
  console.log(`${key}:`, report.valid ? '✅' : '❌', report.issues)
}
```

### 4. Performance Monitoring
```javascript
// Monitor loading performance
console.time('Full Dataset Test')
await window.debugUtils.testAllDatasets()
console.timeEnd('Full Dataset Test')
```

## Notes

- Debug utilities are automatically available after application load
- All functions return Promises - use `await` or `.then()`
- Console functions are read-only and safe for production use
- Malformed JSON in `peekaboo-rep2-v1.1.1` is intentional for testing error handling
- Functions include detailed console logging for transparency

---

*Generated for Video Action Viewer v0.2.0*