# = VideoAnnotator v1.2.0 API Upgrade Plan

## Overview

VideoAnnotator v1.2.0 introduces a major API modernization to transform the toolkit from a library-focused system into a production-ready service platform. This upgrade maintains backward compatibility while adding enterprise-grade features for scalable video annotation workflows.

**Target Release**: Q2 2025  
**Breaking Changes**: Minimal (opt-in new features)  
**Migration Path**: Gradual adoption with legacy support

---

## <¯ Core API Objectives

### 1. Service-Oriented Architecture
Transform VideoAnnotator from a Python library to a full-service platform:
- **REST API Server** - FastAPI-based HTTP service
- **Async Processing** - Non-blocking job queue system  
- **Microservice Ready** - Container-first deployment
- **Cloud Native** - Kubernetes and cloud provider integration

### 2. Enterprise Features
Production-ready capabilities for organizational deployment:
- **Multi-tenancy** - User isolation and resource management
- **Authentication** - OAuth2, JWT token management
- **Authorization** - Role-based access control (RBAC)
- **Rate Limiting** - API usage quotas and throttling
- **Audit Logging** - Comprehensive activity tracking

### 3. Developer Experience
Modern API design following industry standards:
- **OpenAPI/Swagger** - Auto-generated documentation
- **SDK Generation** - Multi-language client libraries
- **Webhook Support** - Event-driven integrations
- **Versioning Strategy** - Backward compatible API evolution

---

## <× New API Architecture

### Current Architecture (v1.1.1)
```python
# Direct pipeline usage
from src.pipelines.person_tracking import PersonTrackingPipeline
pipeline = PersonTrackingPipeline(config)
pipeline.initialize()
results = pipeline.process(video_path, 0, 30, output_dir)
pipeline.cleanup()
```

### New API Architecture (v1.2.0)
```python
# Option 1: Service Client (Recommended)
from videoannotator import VideoAnnotatorClient
client = VideoAnnotatorClient(api_url="https://api.videoannotator.ai")
job = client.submit_job(video_path, pipelines=["person_tracking"])
results = client.get_results(job.id)

# Option 2: Direct Library (Backward Compatible)  
from videoannotator import VideoAnnotator
annotator = VideoAnnotator()
results = annotator.process(video_path, pipelines=["person_tracking"])
```

---

## =€ New REST API Endpoints

### Core Processing Endpoints
```
POST   /api/v1/jobs                    # Submit processing job
GET    /api/v1/jobs/{job_id}          # Get job status
GET    /api/v1/jobs/{job_id}/results  # Download results
DELETE /api/v1/jobs/{job_id}          # Cancel/delete job
GET    /api/v1/jobs                   # List user jobs
```

### Pipeline Management
```
GET    /api/v1/pipelines              # List available pipelines
GET    /api/v1/pipelines/{name}       # Get pipeline details
POST   /api/v1/pipelines/{name}/test  # Test pipeline configuration
```

### Data Management  
```
POST   /api/v1/videos                 # Upload video for processing
GET    /api/v1/videos/{video_id}      # Get video metadata
DELETE /api/v1/videos/{video_id}      # Delete uploaded video
GET    /api/v1/annotations/{job_id}   # Export annotations (various formats)
```

### System Management
```
GET    /api/v1/health                 # System health check
GET    /api/v1/metrics                # Processing metrics
GET    /api/v1/config                 # Available configurations
POST   /api/v1/auth/login             # User authentication
POST   /api/v1/auth/refresh           # Token refresh
```

---

## =' Enhanced CLI Interface

### Current CLI (v1.1.1)
```bash
# Multiple entry points
python -m src.pipelines.scene_detection.scene_pipeline --input video.mp4
python batch_demo.py --input_dir videos/ --output_dir results/
python demo.py
```

### New Unified CLI (v1.2.0)
```bash
# Single entry point with subcommands
videoannotator process video.mp4 --pipelines person_tracking,face_analysis
videoannotator batch process --input-dir videos/ --output-dir results/
videoannotator server start --host 0.0.0.0 --port 8000
videoannotator job submit video.mp4 --wait-for-completion
videoannotator job status 12345-abcd-6789
videoannotator pipeline list
videoannotator config validate my_config.yaml
```

---

## =Ê Database Integration

### New Persistent Storage
Replace file-based metadata with proper database:

```sql
-- Core Tables
CREATE TABLE users (
    id UUID PRIMARY KEY,
    email VARCHAR UNIQUE,
    role user_role,
    created_at TIMESTAMP
);

CREATE TABLE jobs (
    id UUID PRIMARY KEY,  
    user_id UUID REFERENCES users(id),
    video_path TEXT,
    config JSONB,
    status job_status,
    created_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE annotations (
    id UUID PRIMARY KEY,
    job_id UUID REFERENCES jobs(id),
    pipeline_name VARCHAR,
    data JSONB,
    format annotation_format,
    created_at TIMESTAMP
);
```

### Migration Strategy
- **Phase 1**: Dual storage (files + database)
- **Phase 2**: Database-first with file export
- **Phase 3**: Pure database storage

---

## = Authentication & Authorization

### Authentication Methods
```python
# API Key Authentication
headers = {"Authorization": "Bearer your-api-key"}

# OAuth2 with JWT
@app.get("/api/v1/protected")
async def protected_endpoint(user: User = Depends(get_current_user)):
    return {"user_id": user.id}
```

### Authorization Levels
- **Admin**: Full system access, user management
- **Researcher**: Create jobs, access all personal data
- **Analyst**: View-only access to assigned datasets
- **Guest**: Public demo access with limits

---

## =È Async Processing System

### Job Queue Implementation
```python
from celery import Celery
from videoannotator.tasks import process_video

# Submit job
task = process_video.delay(video_path, config)
job_id = task.id

# Check status
status = process_video.AsyncResult(job_id)
if status.ready():
    results = status.get()
```

### Scaling Strategy
- **Redis** - Job queue and caching
- **Celery** - Distributed task processing  
- **GPU Pools** - Managed GPU resource allocation
- **Auto-scaling** - Dynamic worker management

---

## = Migration Guide

### For Library Users
```python
# v1.1.1 (Still Supported)
from src.pipelines.person_tracking import PersonTrackingPipeline
# ... existing code works unchanged

# v1.2.0 (Recommended) 
from videoannotator import VideoAnnotator
annotator = VideoAnnotator()  # Simplified interface
results = annotator.process(video, pipelines=["person_tracking"])
```

### For Production Deployments
```bash
# 1. Start API server
videoannotator server start

# 2. Submit jobs via API
curl -X POST "http://localhost:8000/api/v1/jobs" \
     -H "Authorization: Bearer $API_KEY" \
     -F "video=@video.mp4" \
     -F "config={\"pipelines\": [\"person_tracking\"]}"

# 3. Monitor and retrieve results
videoannotator job status $JOB_ID
videoannotator job results $JOB_ID --format coco
```

---

## =Å Implementation Timeline

### Month 1-2: Core API Foundation
- [ ] FastAPI server implementation
- [ ] Basic job submission and status endpoints
- [ ] Database schema and ORM setup
- [ ] Authentication system (API keys)

### Month 3-4: Enhanced Features
- [ ] Async processing with Celery
- [ ] File upload and management
- [ ] Multiple output format support
- [ ] Basic monitoring and metrics

### Month 5-6: Production Readiness
- [ ] Advanced authentication (OAuth2)
- [ ] Role-based access control
- [ ] Rate limiting and quotas
- [ ] Comprehensive testing and documentation

---

##   Breaking Changes & Compatibility

### Minimal Breaking Changes
- **Import paths**: New recommended imports, old paths deprecated but functional
- **CLI interface**: New unified CLI, old scripts work but show deprecation warnings  
- **Configuration**: Enhanced config format, backward compatible with v1.1.1 configs

### Deprecation Schedule
- **v1.2.0**: Deprecation warnings for old interfaces
- **v1.3.0**: Legacy interfaces moved to compatibility module
- **v2.0.0**: Full removal of deprecated interfaces (12+ months notice)

### Migration Support
- **Automated tools**: Scripts to upgrade configurations and code
- **Documentation**: Step-by-step migration guides
- **Support channels**: Dedicated help for migration issues

---

## >ê Testing & Validation

### API Testing Strategy
- **Unit Tests**: All new API endpoints and services
- **Integration Tests**: End-to-end workflow validation
- **Load Testing**: Performance under concurrent usage
- **Security Testing**: Authentication and authorization validation

### Compatibility Testing
- **Backward Compatibility**: Ensure v1.1.1 code works unchanged
- **Cross-Platform**: Linux, Windows, macOS validation
- **Container Testing**: Docker and Kubernetes deployment validation

---

This API upgrade positions VideoAnnotator as a modern, scalable video annotation platform suitable for both research and production environments while maintaining the ease of use that makes it valuable for individual researchers.

*Last Updated: January 2025*