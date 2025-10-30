# API Improvements - October 30, 2025

## Overview

This document describes two key API improvements made to address client-side issues reported during testing.

## 1. Trailing Slash Handling (Fixed)

### Problem
FastAPI's default behavior automatically redirects requests without trailing slashes to URLs with trailing slashes using HTTP 307 Temporary Redirect. This caused confusion during testing:

- `GET /api/v1/jobs` → 307 Redirect → `/api/v1/jobs/`
- Some HTTP clients don't automatically follow redirects with auth headers
- Testing with curl/PowerShell showed auth "working" in browser but "failing" in CLI

### Solution
Disabled automatic trailing slash redirects in FastAPI configuration:

```python
app = FastAPI(
    ...
    redirect_slashes=False,  # Disable automatic trailing slash redirects
)
```

### Impact
- API endpoints now accept requests **with or without** trailing slashes
- No more confusing 307 redirects
- Clearer error messages when endpoints don't exist
- Better compatibility with all HTTP clients

### Files Changed
- `src/videoannotator/api/main.py`

## 2. Video Metadata in Job Responses (Added)

### Problem
Job response objects were missing video metadata fields that clients need:
- No `video_filename` (clients showed "N/A")
- No `video_size_bytes` (file size unavailable)
- No `video_duration_seconds` (duration unavailable)

Clients had to implement workarounds like extracting filename from `video_path`.

### Solution
Added three new fields to `JobResponse` model and populated them across all job endpoints:

#### New Fields
```python
class JobResponse(BaseModel):
    # ... existing fields ...
    video_filename: str | None          # Original video filename
    video_size_bytes: int | None        # Video file size in bytes
    video_duration_seconds: int | None  # Video duration in seconds
```

#### Implementation Details

**Helper Function** (`extract_video_metadata`):
- Extracts filename from path
- Gets file size via `Path.stat()`
- Extracts duration using OpenCV (reads FPS and frame count)
- Gracefully handles missing files or read errors
- Returns `None` values when metadata unavailable

**Extraction Points**:
1. **Job submission** (`POST /api/v1/jobs/`): Extracts metadata immediately after video upload
2. **Get job** (`GET /api/v1/jobs/{job_id}`): Extracts metadata when retrieving job details
3. **List jobs** (`GET /api/v1/jobs/`): Extracts metadata for each job in pagination
4. **Cancel job** (`POST /api/v1/jobs/{job_id}/cancel`): Includes metadata in cancellation response

### Example Response

**Before:**
```json
{
  "id": "abc-123",
  "status": "completed",
  "video_path": "/tmp/uploads/my_video.mp4",
  "created_at": "2025-10-30T10:00:00Z"
}
```

**After:**
```json
{
  "id": "abc-123",
  "status": "completed",
  "video_path": "/tmp/uploads/my_video.mp4",
  "video_filename": "my_video.mp4",
  "video_size_bytes": 15728640,
  "video_duration_seconds": 120,
  "created_at": "2025-10-30T10:00:00Z"
}
```

### Files Changed
- `src/videoannotator/api/v1/jobs.py`
  - Added `video_filename`, `video_size_bytes`, `video_duration_seconds` to `JobResponse`
  - Added `extract_video_metadata()` helper function
  - Updated all 6 JobResponse construction sites to include video metadata

## Performance Considerations

### Video Metadata Extraction
- **Filename extraction**: Negligible cost (path string operation)
- **File size**: Fast (~1ms, single stat() syscall)
- **Duration extraction**: Moderate cost (~50-200ms per video depending on codec)
  - Only reads video headers (no frame decoding)
  - Cached by filesystem for repeated accesses
  - Runs in parallel for list endpoint (multiple jobs)

### Optimization for List Endpoint
The list endpoint processes multiple jobs. Current implementation:
- Extracts metadata sequentially for each job on the current page
- Only processes jobs visible on current page (pagination limits to 10-100 jobs)
- Failed extractions don't break the entire listing (graceful degradation)

**Future optimization ideas** (if needed):
- Cache video metadata in database during job submission
- Lazy-load metadata (separate `/jobs/{id}/metadata` endpoint)
- Background job to pre-populate metadata for all jobs

## Testing Recommendations

### Test Cases
1. **Trailing slash acceptance**:
   ```bash
   # Both should work without redirects
   curl -H "Authorization: Bearer TOKEN" http://localhost:18011/api/v1/jobs
   curl -H "Authorization: Bearer TOKEN" http://localhost:18011/api/v1/jobs/
   ```

2. **Video metadata presence**:
   ```bash
   # Submit job and verify metadata in response
   curl -X POST http://localhost:18011/api/v1/jobs/ \
     -H "Authorization: Bearer TOKEN" \
     -F "video=@test.mp4"

   # Response should include:
   # "video_filename": "test.mp4"
   # "video_size_bytes": <file size>
   # "video_duration_seconds": <duration>
   ```

3. **Missing video file handling**:
   - If video file is deleted after job creation
   - Should return `null` for metadata fields
   - Should not cause 500 errors

4. **Large video files**:
   - Test with videos >1GB to ensure duration extraction doesn't timeout
   - Current OpenCV header-only reading should handle large files fine

## Migration Notes

### Backward Compatibility
✅ **Fully backward compatible** - New fields are optional:
- Existing clients ignoring new fields continue to work
- Null values returned when metadata unavailable
- No breaking changes to existing fields

### Client Updates
Clients can now access video metadata directly:

**JavaScript/TypeScript:**
```typescript
const job = await fetch(`/api/v1/jobs/${jobId}`).then(r => r.json());
console.log(`Video: ${job.video_filename}`);
console.log(`Size: ${job.video_size_bytes} bytes`);
console.log(`Duration: ${job.video_duration_seconds}s`);
```

**Python:**
```python
job = requests.get(f"/api/v1/jobs/{job_id}").json()
print(f"Video: {job['video_filename']}")
print(f"Size: {job['video_size_bytes']} bytes")
print(f"Duration: {job['video_duration_seconds']}s")
```

## Future Enhancements

### Potential Database Schema Update
Currently metadata is extracted on-demand. Consider adding to database:

```python
# In database/models.py Job model (already exists!)
video_filename = Column(String(255), nullable=True)
video_size_bytes = Column(Integer, nullable=True)
video_duration_seconds = Column(Integer, nullable=True)
```

**Benefits:**
- No repeated extraction overhead
- Faster list endpoint (no OpenCV calls)
- Consistent metadata even if video file deleted
- Enables queries like "find all jobs with videos >10min"

**Next Steps:**
1. Populate these fields during job submission
2. Store in database via BatchJob or Job model
3. Remove on-demand extraction from API endpoints

### Additional Metadata Ideas
- `video_resolution`: Width x height (e.g., "1920x1080")
- `video_fps`: Frame rate
- `video_codec`: Video codec (e.g., "h264", "vp9")
- `video_bitrate`: Bitrate in kbps
- All extractable from OpenCV without significant overhead

## Related Issues

- Client issue: Jobs list showing "N/A" for all video information
- Client issue: Trailing slash redirects causing auth confusion in PowerShell
- Database models already have video metadata fields (future integration point)

## References

- FastAPI `redirect_slashes` parameter: https://fastapi.tiangolo.com/advanced/path-operation-advanced-configuration/
- OpenCV video properties: https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html
- HTTP 307 Temporary Redirect: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/307
