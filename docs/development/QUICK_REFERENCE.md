# 🎯 VideoAnnotator v1.3.0 & v1.4.0 - Quick Reference

**Date**: October 9, 2025  
**Current Release**: v1.2.2  
**Active Development**: v1.3.0 (Production Reliability)  
**Next Major**: v1.4.0 (First Public Release + JOSS)

---

## ⚡ Quick Summary

### v1.3.0 (6-8 weeks) - Production Reliability
**CRITICAL FIXES ONLY** - No new features

**The Big 6 Issues**:
1. ❌ Fix pipeline naming (jobs failing with "audio_processing" not found)
2. ❌ Persistent storage (no more data loss on restart)
3. ❌ Job cancellation (stop GPU jobs, prevent OOM)
4. ⚠️ Config validation (reject bad configs early)
5. ⚠️ Security defaults (auth required, CORS restricted)
6. ⚠️ Error consistency (standard format everywhere)

**Also Includes**:
- Package namespace migration (`src/` → `videoannotator/`)
- Batch/job semantics fixes
- Test fixture improvements

---

### v1.4.0 (3-4 months after v1.3.0) - Public Release
**Research-ready platform with JOSS paper**

**Key Deliverables**:
- 📚 JOSS paper (complete manuscript)
- 🔬 4+ research workflow examples
- 🐳 Docker images (CPU/GPU, pinned deps)
- 📊 Benchmark suite + validation
- 📖 Publication-quality docs
- 📦 PyPI package (`pip install videoannotator`)
- 🔧 All deferred v1.3.0 items (version endpoint, enhanced health, logging)

---

## 📋 What Goes Where?

### v1.3.0: Critical Fixes
✅ **Include** if it:
- Blocks production use
- Causes data loss
- Crashes server
- Security risk
- Makes jobs fail

❌ **Defer** if it:
- Nice-to-have
- Documentation
- New feature
- Performance optimization (unless critical)
- Advanced capability

### v1.4.0: Public Release Polish
✅ **Include** if it:
- Needed for JOSS paper
- Improves reproducibility
- Documentation gap
- Installation friction
- Deferred from v1.3.0

❌ **Defer to v1.5.0+** if it:
- Advanced ML (active learning, multi-modal)
- Plugin system
- Enterprise features
- Real-time streaming
- GraphQL API

---

## 🚨 From Jerry's Testing

| Issue | v1.3.0 | v1.4.0 | v1.5.0+ |
|-------|--------|--------|---------|
| Pipeline naming failures | ✅ | | |
| Ephemeral storage | ✅ | | |
| Cannot cancel jobs | ✅ | | |
| Config validation | ✅ | | |
| Security defaults | ✅ | | |
| Error format consistency | ✅ | | |
| Version inconsistency | | ✅ | |
| Enhanced health endpoint | | ✅ | |
| YAML loader edge cases | | ✅ | |
| Legacy examples cleanup | | ✅ | |
| Structured logging | | ✅ | |

---

## 🔥 From Client Team (Video Annotation Viewer v0.4.0 QA)

| Issue | v1.3.0 | v1.4.0 |
|-------|--------|--------|
| Jobs failing ("audio_processing" not found) | ✅ | |
| Debug endpoint returns 401 | ✅ | |
| Pipeline catalog endpoint 404 | | ✅ |

**Root Cause**: Pipeline registry metadata doesn't match actual implementations.

**Fix**: Audit script to compare registry vs implementations, add startup validation.

---

## 📅 v1.3.0 Timeline

### Week 1-2: Critical Fixes
- Pipeline registry audit + name mapping fix
- Persistent storage implementation
- Job cancellation MVP

### Week 3-4: Quality & Security
- Schema-based config validation
- Secure-by-default configuration
- Standardized error envelope

### Week 5-6: Technical Debt
- Package namespace migration
- Batch/job semantics fixes
- Deterministic test fixtures

### Week 7-8: Beta & Release
- Client team integration testing
- Documentation updates
- Release preparation

---

## 📅 v1.4.0 Timeline

### Month 1: Examples & Reproducibility
- Research workflow examples (classroom, clinical, developmental, group dynamics)
- Docker images (CPU/GPU)
- Benchmark suite

### Month 2: Documentation
- Academic docs (methods, validation, comparison)
- User docs (quick start, tutorials, API reference)
- Developer docs (architecture, contributing)

### Month 3: Usability & Quality
- Installation simplification (PyPI)
- Progress indicators
- Export formats (FiftyOne, Label Studio)
- Quality assessment pipeline

### Month 4: Testing & Release
- Comprehensive testing (80%+ coverage)
- JOSS paper finalization
- Community preparation
- Release!

---

## ✅ Success Criteria

### v1.3.0
- [ ] Zero job failures due to pipeline naming
- [ ] Zero data loss on server restart
- [ ] All running jobs cancellable within 5 seconds
- [ ] Invalid configs rejected at submission
- [ ] Authentication required by default
- [ ] All errors use standard envelope
- [ ] Package namespace migrated

### v1.4.0
- [ ] JOSS paper submitted/accepted
- [ ] 3+ research examples work 100%
- [ ] PyPI package published
- [ ] Docker images published
- [ ] Documentation complete
- [ ] Test coverage ≥ 80%
- [ ] 1,000+ downloads in first month

---

## 🛠️ Immediate Actions (This Week)

### For v1.3.0 Development

1. **Create Pipeline Audit Script**
   ```bash
   # Compare registry metadata vs actual pipeline classes
   uv run python scripts/audit_pipeline_names.py
   ```

2. **Implement Persistent Storage**
   ```bash
   # Add STORAGE_DIR env var
   # Update file upload/download to use persistent paths
   ```

3. **Job Cancellation POC**
   ```bash
   # Add CANCELLED status to database
   # Implement cancel endpoint
   # Test GPU cleanup
   ```

### For Documentation

1. **Update AGENTS.md** - Add v1.3.0/v1.4.0 guidance
2. **Update CHANGELOG.md** - Prepare v1.3.0 section
3. **Create Migration Guide** - v1.2.x → v1.3.0 breaking changes

---

## 📞 Who to Contact

### Critical Issues (v1.3.0)
- **Pipeline failures**: Review with backend team
- **Storage design**: Backend + DevOps
- **Security defaults**: Security review needed

### JOSS Paper (v1.4.0)
- **Content**: Senior researchers for use cases
- **Validation**: Stats/methods experts
- **Figures**: Data visualization specialist

### Community (v1.4.0)
- **Website**: Front-end / design team
- **Videos**: Video production / screencasts
- **Examples**: Domain experts for each scenario

---

## 🔗 Key Documents

- [Roadmap Overview](roadmap_overview.md) - Complete strategy
- [v1.3.0 Roadmap](roadmap_v1.3.0.md) - Detailed v1.3.0 plan
- [v1.4.0 Roadmap](roadmap_v1.4.0.md) - Detailed v1.4.0 plan
- [Jerry's Issues](../testing/Jerry-issues.md) - Testing feedback analysis
- [Client Team Issues](../testing/Issues%20for%20Server%20Team.md) - QA findings
- [AGENTS.md](../../AGENTS.md) - AI assistant guidance

---

## 💡 Key Principles to Remember

1. **v1.3.0 = Fix, Don't Add** - Critical fixes only
2. **v1.4.0 = Polish, Don't Innovate** - Make what exists shine
3. **v1.5.0+ = Expand** - Advanced features come later
4. **JOSS First** - v1.4.0 is about the paper
5. **Community Ready** - Documentation and onboarding matter
6. **Research Focus** - We serve researchers, not enterprises (yet)

---

**Quick Links**:
- [Full Roadmap Overview](roadmap_overview.md)
- [GitHub Issues](https://github.com/InfantLab/VideoAnnotator/issues)
- [GitHub Discussions](https://github.com/InfantLab/VideoAnnotator/discussions)

---

**Last Updated**: October 9, 2025
