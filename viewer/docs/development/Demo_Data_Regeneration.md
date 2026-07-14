# Demo Data Regeneration Task

**Priority:** 🔴 Critical  
**Status:** Not Started  
**Target:** v0.5.0 Phase 1  
**Owner:** TBD  
**Created:** 2025-10-09

---

## 🎯 **OBJECTIVE**

Regenerate all demo datasets using VideoAnnotator v1.2.x to ensure:
1. Job metadata is included (pipeline IDs, versions, parameters)
2. Capability-aware controls work correctly in the viewer
3. All annotation types are properly represented
4. Demo data matches current server capabilities

---

## ⚠️ **PROBLEM STATEMENT**

**Current Issue:**
- Existing demo data in `demo/videos_out/` lacks job pipeline metadata
- OpenFace3 individual toggles are disabled for demo data because `jobPipelines` array is empty
- Cannot properly test/demo capability-aware features (v0.4.0 features)
- QA testing blocked for Section 3: Capability-Aware Viewer

**Impact:**
- Users/testers cannot see full functionality of v0.4.0 features
- QA checklist cannot be completed
- Demo experience doesn't showcase latest capabilities

---

## 📋 **REQUIREMENTS**

### **Must Have:**
1. **Job Metadata**: Each demo result must include:
   - List of pipelines that were run (`job.pipelines: string[]`)
   - Pipeline versions used
   - Parameters/configuration for each pipeline
   - Server version used for generation

2. **Annotation Coverage**: Ensure demos include:
   - ✅ Face detection & analysis (OpenFace3)
   - ✅ Pose estimation (COCO keypoints)
   - ✅ Audio transcription (WebVTT)
   - ✅ Speaker identification (RTTM)
   - ✅ Scene detection
   - ✅ Emotion analysis

3. **Variety**: Multiple demos showing:
   - Single person vs multiple people
   - Clear audio vs noisy audio
   - Good lighting vs challenging conditions
   - Different video lengths (5s to 60s)

### **Should Have:**
- Consistent naming convention
- README documenting each demo
- Generation scripts for reproducibility
- Validation that all expected data is present

---

## 🔧 **IMPLEMENTATION STEPS**

### **Step 1: Prepare VideoAnnotator Server**
```bash
# Ensure server is v1.2.x
cd /path/to/videoannotator
python -m videoannotator.server --port 18011

# Verify version
curl http://localhost:18011/api/v1/system/health
```

### **Step 2: Select/Organize Demo Videos**
Current videos in `demo/videos/`:
- `2UWdXP.joke1.rep2.take1.Peekaboo_h265.mp4`
- `2UWdXP.joke1.rep3.take1.Peekaboo_h265.mp4`
- `3.mp4`
- `3dC3SQ.joke1.rep1.take1.TearingPaper_h265.mp4`
- `3dC3SQ.joke1.rep2.take1.TearingPaper_h265.mp4`
- `4JDccE.joke5.rep2.take1.TearingPaper_h265.mp4`
- `4JDccE.joke5.rep3.take1.TearingPaper_h265.mp4`
- `6c6MZQ.joke1.rep1.take1.ThatsNotAHat_h265.mp4`
- `6c6MZQ.joke1.rep2.take1.ThatsNotAHat_h265.mp4`

**Action:** Keep all, organize by complexity/use case

### **Step 3: Create Processing Script**
```python
# scripts/regenerate_demos.py

import requests
import json
from pathlib import Path

API_BASE = "http://localhost:18011"
TOKEN = "dev-token"
DEMO_VIDEOS = Path("demo/videos")
OUTPUT_DIR = Path("demo/videos_out")

def submit_job(video_path, pipelines):
    """Submit a job with specified pipelines"""
    url = f"{API_BASE}/api/v1/jobs"
    headers = {"Authorization": f"Bearer {TOKEN}"}
    
    files = {"video": open(video_path, "rb")}
    data = {
        "pipelines": json.dumps(pipelines),
        # Add any pipeline-specific config
    }
    
    response = requests.post(url, headers=headers, files=files, data=data)
    return response.json()

def wait_for_completion(job_id):
    """Poll job status until complete"""
    # Implementation...

def download_results(job_id, output_path):
    """Download and organize job results"""
    # Implementation...

# Process each demo video
demos = [
    {
        "video": "3.mp4",
        "pipelines": ["face_analysis", "person_tracking", "audio_transcription"],
        "description": "Multi-modal demo with all features"
    },
    # ... more demos
]

for demo in demos:
    print(f"Processing {demo['video']}...")
    job = submit_job(DEMO_VIDEOS / demo["video"], demo["pipelines"])
    wait_for_completion(job["id"])
    download_results(job["id"], OUTPUT_DIR / demo["video"].stem)
```

### **Step 4: Process Each Demo**
For each video, submit jobs with different pipeline combinations:
- **Full pipeline**: All available pipelines
- **Face-only**: Just face analysis
- **Audio-only**: Just transcription + speaker ID
- **Pose-only**: Just pose estimation

### **Step 5: Enhance Output Format**
Modify result structure to include job metadata:
```json
{
  "video_info": { ... },
  "job_metadata": {
    "job_id": "job_123",
    "pipelines": ["face_analysis", "person_tracking"],
    "server_version": "1.2.2",
    "created_at": "2025-10-09T12:00:00Z",
    "pipeline_configs": {
      "face_analysis": {
        "version": "1.0.2",
        "model": "OpenFace3",
        "parameters": { ... }
      }
    }
  },
  "annotations": {
    "faces": [ ... ],
    "poses": [ ... ],
    ...
  }
}
```

### **Step 6: Update Demo Loader**
Update `src/utils/demoDataLoader.ts` (or equivalent) to:
- Parse job metadata from demo files
- Pass `jobPipelines` to viewer components
- Display job info in UI

### **Step 7: Validation**
For each regenerated demo:
- [ ] Verify all expected annotation types present
- [ ] Confirm job metadata is included
- [ ] Test in viewer - capability-aware controls work
- [ ] Check file sizes are reasonable
- [ ] Ensure videos play correctly

---

## 📁 **OUTPUT STRUCTURE**

```
demo/
├── videos/                      # Original video files
│   └── *.mp4
├── videos_out/                  # Generated annotations
│   ├── demo_1_multimodal/
│   │   ├── video.mp4           # Copy of original
│   │   ├── metadata.json       # Job metadata
│   │   ├── annotations.json    # All annotations combined
│   │   ├── faces.json          # Face data (COCO + OpenFace3)
│   │   ├── poses.json          # Pose data (COCO)
│   │   ├── transcript.vtt      # WebVTT transcript
│   │   ├── speakers.rttm       # Speaker identification
│   │   └── scenes.json         # Scene boundaries
│   └── README.md               # Demo catalog
└── README.md                    # Generation documentation
```

---

## ✅ **ACCEPTANCE CRITERIA**

- [ ] All demos regenerated with v1.2.x server
- [ ] Each demo includes job metadata with pipeline IDs
- [ ] OpenFace3 individual toggles work in viewer for all demos
- [ ] "Toggle All" button works correctly
- [ ] Capability badges show correctly based on job pipelines
- [ ] Demo README documents each dataset
- [ ] Generation process documented and reproducible
- [ ] QA checklist Section 3 can be completed successfully

---

## 🚧 **BLOCKERS & DEPENDENCIES**

- [ ] VideoAnnotator server v1.2.x must be stable and accessible
- [ ] API endpoints for job submission and result retrieval working
- [ ] Sufficient compute resources for processing all demos
- [ ] Clear understanding of required metadata format

---

## 📊 **SUCCESS METRICS**

- ✅ 100% of demo datasets have job metadata
- ✅ 0 capability-aware UI bugs with new demo data
- ✅ QA testing for Section 3 can be completed
- ✅ Demo loading time < 2 seconds per dataset

---

## 🔗 **RELATED DOCUMENTS**

- [v0.5.0 Roadmap](../archive/ROADMAP_v0.5.0.md)
- [QA Checklist v0.4.0](../archive/QA_Checklist_v0.4.0.md)
- [File Formats Guide](../FILE_FORMATS.md)
- [Client-Server Collaboration Guide](../CLIENT_SERVER_COLLABORATION_GUIDE.md)

---

## 📝 **NOTES**

### **QA Testing Discovery (2025-10-09)**
- Individual OpenFace3 toggles disabled for demo data
- Root cause: `jobPipelines` array empty for demo data
- Temporary fix applied to fall back to data-based availability
- Proper fix: regenerate demos with job metadata

### **Technical Considerations**
- Demo loader needs update to extract and pass `jobPipelines`
- Viewer components already support `jobPipelines` prop
- May need to update demo file format specification

---

**Last Updated:** 2025-10-09  
**Status:** Planning - Ready to start implementation
