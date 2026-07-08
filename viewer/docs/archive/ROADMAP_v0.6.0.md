# Video Annotation Viewer v0.6.0 Roadmap (Superseded)

> This roadmap has been superseded by [ROADMAP_v0.6.1.md](./ROADMAP_v0.6.1.md).
> v0.6.0 was released in December 2025; the active JOSS release implementation plan is v0.6.1.

**Theme:** JOSS Paper & Public Release  
**Status:** 📋 PLANNED  
**Target Date:** Q2 2026 (after VideoAnnotator v1.4.0 release)  
**Previous Version:** v0.5.0 (VideoAnnotator v1.3.0 Integration - October 2025)

---

## 🎯 **OVERVIEW**

v0.6.0 is a **tightly scoped release** focused exclusively on JOSS (Journal of Open Source Software) paper submission and public release readiness. This version delivers the minimum viable documentation, demo data, and quality improvements required for academic publication.

**Dependency:** VideoAnnotator server v1.4.0 must be released first (provides stable API and complete feature set for documentation).

---

## 📦 **FEATURES (JOSS Requirements Only)**

### **1. JOSS Paper Preparation** 📄

**Priority:** 🔴 CRITICAL (Blocking release)

#### 1.1 Paper Content
- [ ] **Abstract & Introduction**
  - Clear problem statement
  - Solution overview
  - Target audience identification
  
- [ ] **Statement of Need**
  - Why this tool is needed in research community
  - Gap in existing tools
  - Research use cases
  
- [ ] **Software Architecture**
  - High-level system design with diagrams
  - Component relationships
  - Technology stack overview
  
- [ ] **Features & Functionality**
  - Comprehensive feature description
  - VideoAnnotator integration details
  - Annotation format support
  
- [ ] **Use Cases & Examples**
  - Real-world research applications
  - Example workflows
  - Results demonstration
  
- [ ] **Community Guidelines**
  - Contribution process
  - Support channels
  - Development workflow
  
- [ ] **Acknowledgments**
  - Contributors and funding sources
  - Related projects and dependencies

#### 1.2 Supporting Materials
- [ ] **Software DOI** (Zenodo integration)
- [ ] **CITATION.cff** review and update
- [ ] **License verification** for all dependencies
- [ ] **Test coverage report** (demonstrate code quality)
- [ ] **API documentation** (auto-generated from code)

#### 1.3 Peer Review Preparation
- [ ] JOSS reviewer checklist completion
- [ ] Cross-platform installation testing
- [ ] Example notebooks (if applicable)

**Estimated Effort:** Medium (2-3 weeks including revisions)

---

### **2. Demo Data Regeneration** 🎬

**Priority:** 🔴 CRITICAL (Required for JOSS examples)

- [ ] **Re-process all demo videos** with VideoAnnotator v1.4.x
  - Use latest stable pipeline configurations
  - Ensure all annotation types represented (faces, poses, audio, scenes)
  - Include complete job metadata (pipeline IDs, parameters, versions)
  - Verify capability-aware controls work correctly
  
- [ ] **Document generation process**
  - Reproducible pipeline configurations
  - Processing environment details
  - Expected outputs and formats
  
- [ ] **Validate demo quality**
  - Check all overlays render correctly
  - Verify metadata completeness
  - Test capability-aware features
  - Ensure files are properly organized

**Rationale:** Current demo data lacks job metadata from v1.3.0+, preventing proper testing of capability-aware features. JOSS reviewers need working examples.

**Estimated Effort:** Small-Medium (1-2 weeks)

---

### **3. Essential Documentation** 📚

**Priority:** 🔴 CRITICAL (JOSS requirement)

#### 3.1 User Documentation
- [ ] **Getting Started Guide**
  - Installation instructions
  - Token configuration
  - First job creation walkthrough
  - Viewing results tutorial
  
- [ ] **User Guide** (Core workflows only)
  - Video upload and annotation loading
  - Pipeline selection basics
  - Job creation and monitoring
  - Results viewing and interpretation
  
- [ ] **FAQ Section** (Common issues only)
  - Installation troubleshooting
  - Connection errors
  - Server setup basics
  - Browser compatibility

#### 3.2 Developer Documentation (Minimal)
- [ ] **Architecture Overview**
  - High-level system design
  - Component relationships
  - Data flow diagrams
  
- [ ] **Contribution Guidelines**
  - Development workflow
  - Testing requirements
  - Pull request process

**Estimated Effort:** Medium (2-3 weeks)

---

### **4. Quality Assurance** ✅

**Priority:** 🟠 HIGH (JOSS credibility)

- [ ] **Bug Triage & Fixes**
  - Fix all critical bugs
  - Address high-priority issues
  - Test on Windows, macOS, Linux
  
- [ ] **Cross-Browser Testing**
  - Chrome, Firefox, Safari, Edge
  - Document known limitations
  
- [ ] **Test Coverage**
  - Ensure >80% coverage maintained
  - Fix flaky tests
  - Add missing integration tests
  
- [ ] **Performance Validation**
  - Lighthouse CI passing
  - No memory leaks
  - Acceptable load times

**Estimated Effort:** Small-Medium (1-2 weeks)

---

### **5. Accessibility Basics** ♿

**Priority:** 🟠 HIGH (Ethical/legal requirement for public release)

- [ ] **Critical WCAG 2.1 AA Issues**
  - Color contrast compliance (automated fixes)
  - Keyboard navigation (focus indicators)
  - ARIA labels for key UI elements
  - Alt text for images
  
- [ ] **Basic Testing**
  - Automated accessibility audit (axe)
  - Manual keyboard navigation test
  - Screen reader smoke test

**Note:** Full accessibility compliance deferred to v0.7.0. This covers only critical issues that would block public release.

**Estimated Effort:** Small (1 week)

---

## 🚫 **OUT OF SCOPE (Deferred to v0.7.0)**

The following are explicitly **NOT** included in v0.6.0:

- ❌ Landing page redesign
- ❌ Visual configuration builder
- ❌ User preference persistence
- ❌ Advanced filtering and search
- ❌ Bulk operations and batch processing
- ❌ Folder selection
- ❌ Job deletion polish (basic implementation sufficient)
- ❌ Performance optimizations (beyond bug fixes)
- ❌ Multi-user features
- ❌ Offline support
- ❌ Full accessibility compliance (only critical issues)

See [ROADMAP_v0.7.0.md](./ROADMAP_v0.7.0.md) for these features.

---

## 📋 **DEVELOPMENT PHASES**

### **Phase 1: Foundation (Weeks 1-2)** 🏗️
- [ ] Regenerate demo data with v1.4.x metadata
- [ ] Set up JOSS paper structure
- [ ] Audit and triage critical bugs
- [ ] Begin user documentation

**Milestone:** Demo data complete, JOSS outline approved

---

### **Phase 2: Documentation (Weeks 3-5)** 📚
- [ ] Write JOSS paper first draft
- [ ] Complete getting started guide
- [ ] Write core user guide sections
- [ ] Basic developer documentation
- [ ] Create FAQ from common issues

**Milestone:** JOSS paper draft ready for feedback

---

### **Phase 3: Quality & Polish (Weeks 6-7)** ✨
- [ ] Fix critical bugs
- [ ] Cross-platform testing
- [ ] Accessibility critical issues
- [ ] Performance validation
- [ ] Test coverage improvements

**Milestone:** All tests passing, no critical bugs

---

### **Phase 4: Submission (Week 8)** 🚀
- [ ] Finalize JOSS paper
- [ ] Submit to JOSS
- [ ] Create Zenodo DOI
- [ ] Tag v0.6.0 release
- [ ] Public announcement

**Milestone:** JOSS submitted, v0.6.0 released

---

## 🎯 **SUCCESS CRITERIA**

### **Must-Have (Blocking JOSS Submission)** 🔴
- ✅ JOSS paper submitted and passing initial review
- ✅ Demo data regenerated with v1.4.x metadata
- ✅ Getting Started guide complete
- ✅ Core user documentation complete
- ✅ Developer contribution guidelines
- ✅ Zenodo DOI created
- ✅ No critical bugs
- ✅ Cross-browser compatible (major browsers)
- ✅ Test coverage >80%
- ✅ Critical accessibility issues fixed

### **Nice-to-Have** 🟢
- 🔮 Video tutorials
- 🔮 Example notebooks
- 🔮 Performance benchmarks documented
- 🔮 Mobile browser testing

---

## 🔀 **DEPENDENCIES**

### **Critical Dependency**
- **VideoAnnotator v1.4.0**: Must be released and stable before v0.6.0
  - Provides complete API documentation
  - Stable pipeline catalog
  - Production-ready features
  - No breaking changes expected

### **External Dependencies**
- **JOSS Review Process**: 2-6 weeks after submission
- **Zenodo**: Account setup and DOI generation
- **Community Feedback**: Beta testers for documentation review

---

## ⏱️ **TIMELINE ESTIMATE**

- **Development:** 8 weeks
- **JOSS Review:** 2-6 weeks (out of our control)
- **Total:** ~3-4 months from start to accepted paper

**Target Release:** Q2 2026 (April-June)

---

## 💡 **PRINCIPLES FOR v0.6.0**

1. **JOSS First**: Everything serves the paper submission
2. **Minimum Viable**: No feature creep, defer non-critical items
3. **Quality Over Features**: Focus on stability and documentation
4. **Dependency Awareness**: Wait for v1.4.0 server release
5. **Clear Scope**: Say "no" to anything not blocking JOSS

---

## 🚀 **NEXT STEPS**

After v0.6.0 release and JOSS submission, proceed to v0.7.0 for professional polish and enterprise features.

See [ROADMAP_v0.7.0.md](./ROADMAP_v0.7.0.md) for post-JOSS improvements.

---

**Document Version:** 2.0 (Split from original v0.6.0)  
**Created:** 2025-10-30  
**Last Updated:** 2025-10-30  
**Status:** 📋 Planning - Awaiting VideoAnnotator v1.4.0  
**Next Review:** When server v1.4.0 release timeline is known
