# v1.3.0 Specification Updates - JOSS & Documentation Spring Cleaning

**Date**: October 11, 2025
**Updated By**: AI Assistant
**Context**: Adding JOSS publication readiness and documentation/script cleanup to v1.3.0 scope

---

## Summary of Changes

### New User Stories Added

**User Story 7 - Submit Software for JOSS Publication (Priority: P1)**
- **Why**: JOSS publication is release goal; reviewer friction blocks acceptance
- **Scope**: Installation verification, comprehensive API docs, test coverage
- **Test**: External reviewer completes JOSS checklist without clarification requests

**User Story 8 - Navigate Documentation as New Contributor (Priority: P2)**
- **Why**: Good documentation accelerates community contributions
- **Scope**: Documentation reorganization, troubleshooting guides, contributor clarity
- **Test**: New contributor submits first PR within 2 hours using only docs

---

## New Functional Requirements

### JOSS Publication Readiness (P1) - FR-054 to FR-058

| ID | Requirement | Success Criteria |
|----|-------------|------------------|
| FR-054 | Installation verification script (`scripts/verify_installation.py`) | SC-029: Script succeeds on Ubuntu, macOS, Windows |
| FR-055 | Comprehensive docstrings with curl examples for all API endpoints | SC-030: 100% coverage for public endpoints |
| FR-056 | README documents test execution with CI status badge | SC-032: Reviewer can verify in 15 minutes |
| FR-057 | API docstrings include request/response examples | SC-030: All endpoints have examples |
| FR-058 | Test coverage verified and documented (target >80%) | SC-031: Core pipelines >80% coverage |

**Rationale**: JOSS reviewers need to install, test, and understand the software without external help. These requirements eliminate reviewer friction and ensure acceptance.

### Documentation Organization & Cleanup (P2) - FR-059 to FR-064

| ID | Requirement | Success Criteria |
|----|-------------|------------------|
| FR-059 | Reorganize docs for external users and OSS contributors | SC-033: Positive feedback from 2+ reviewers |
| FR-060 | Include troubleshooting guide for common issues | SC-034: Resolves 80% of issues without tickets |
| FR-061 | Distinguish user guides, contributor guides, API references | SC-035: Contributors can navigate to PR submission |
| FR-062 | Follow consistent hierarchy: installation → usage → development → testing → deployment | SC-033: Clear navigation |
| FR-063 | Include "Getting Started for Reviewers" guide | SC-036: JOSS reviewers complete checklist without clarifications |
| FR-064 | Generated docs include "DO NOT EDIT" warnings and regeneration instructions | SC-037: No manual edits to generated files |

**Rationale**: Current docs/ folder mixes user-facing guides with internal development notes. External contributors and JOSS reviewers need clear navigation paths. Spring cleaning removes confusion.

### Script Consolidation & Diagnostic Tools (P2) - FR-065 to FR-070

| ID | Requirement | Success Criteria |
|----|-------------|------------------|
| FR-065 | Migrate essential testing scripts to main test suite or remove | SC-039: No critical coverage gaps from removed scripts |
| FR-066 | Remove outdated or dangerous scripts with deprecation notices | SC-037: Only actively maintained, documented scripts remain |
| FR-067 | Provide diagnostic CLI: `videoannotator diagnose [system\|gpu\|storage\|database]` | SC-038: Identifies common issues with fix suggestions |
| FR-068 | Diagnostic tools validate installation, dependencies, GPU, storage | SC-038: Actionable diagnostics |
| FR-069 | Scripts have clear documentation headers (purpose and usage) | SC-040: All retained scripts documented |
| FR-070 | Script documentation indicates audience (developers, users, CI) | SC-040: Clear purpose statements |

**Rationale**: scripts/ folder has accumulated test scripts, one-off tools, and experimental code. Some are essential (should be in pytest), some are outdated (should be removed), some are useful (should be formal diagnostic tools). Spring cleaning removes confusion and makes testing more discoverable.

---

## New Success Criteria

### JOSS Publication Quality (P1) - SC-029 to SC-032

- **SC-029**: Installation verification script succeeds on Ubuntu, macOS, Windows
- **SC-030**: 100% of public API endpoints have docstrings with curl examples
- **SC-031**: Test coverage >80% for core pipelines (verified and documented)
- **SC-032**: Independent reviewer installs and runs sample pipeline in <15 minutes

### Documentation Quality (P2) - SC-033 to SC-036

- **SC-033**: Documentation structure receives positive feedback from 2+ external reviewers
- **SC-034**: Troubleshooting guide resolves 80% of common issues without support tickets
- **SC-035**: Contributors navigate from "want to contribute" to "PR submitted" using only CONTRIBUTING.md
- **SC-036**: JOSS reviewers complete checklist without external clarification requests

### Script & Tool Quality (P2) - SC-037 to SC-040

- **SC-037**: scripts/ contains only actively maintained, documented tools (no orphans)
- **SC-038**: Diagnostic CLI identifies common config issues with actionable fixes
- **SC-039**: Essential test functionality captured in pytest (no gaps from removed scripts)
- **SC-040**: All retained scripts have usage documentation and clear purpose

---

## Updated Scope Statement

**Added to "In Scope"**:
- **JOSS publication readiness**: installation verification, API documentation, test coverage validation
- **Documentation spring cleaning**: reorganization for external users, troubleshooting guides, contributor clarity
- **Script consolidation**: migrate essential tests, remove outdated scripts, create diagnostic tools

---

## Implementation Priorities

### Phase 1: JOSS Critical (P1) - Blocks Publication
1. **FR-054**: Create `scripts/verify_installation.py`
2. **FR-055-057**: Enhance API endpoint documentation
3. **FR-058**: Verify and document test coverage
4. **FR-056**: Add CI badge to README

**Timeline**: Week 1-2
**Owner**: TBD
**Validation**: External reviewer test

### Phase 2: Documentation Cleanup (P2) - Improves Contributor Experience
5. **FR-059-062**: Reorganize documentation structure
6. **FR-060**: Create troubleshooting guide
7. **FR-063**: Create "Getting Started for Reviewers" guide
8. **FR-064**: Add warnings to generated docs

**Timeline**: Week 3-4
**Owner**: TBD
**Validation**: 2+ external reviewers

### Phase 3: Script Consolidation (P2) - Removes Technical Debt
9. **FR-065-066**: Audit scripts/, migrate or remove
10. **FR-067-068**: Implement diagnostic CLI commands
11. **FR-069-070**: Document retained scripts

**Timeline**: Week 4-5
**Owner**: TBD
**Validation**: Script audit checklist

---

## Risks & Mitigations

### Risk: Scope Creep Delays JOSS Submission

**Impact**: HIGH - JOSS publication is time-sensitive for v1.3.0 announcement
**Likelihood**: MEDIUM - Documentation and script cleanup can expand indefinitely
**Mitigation**:
- Strict prioritization: P1 (JOSS) must complete before P2 (cleanup)
- Time-box documentation reorganization to 1 week
- Defer non-essential script cleanup to v1.4.0 if needed
- External reviewer validation gates release (prevents endless polishing)

### Risk: Breaking Existing Development Workflows

**Impact**: MEDIUM - Scripts in use by developers might be removed
**Likelihood**: MEDIUM - No clear ownership documentation for scripts
**Mitigation**:
- Survey team before removing any script
- Add deprecation notices before removal
- Document migration path for essential functionality
- Keep removed scripts in archive/ folder for 1 release

### Risk: Documentation Reorganization Breaks Links

**Impact**: MEDIUM - External references and internal cross-links could break
**Likelihood**: HIGH - docs/ has many interconnected files
**Mitigation**:
- Use redirects for moved files
- Scan all docs for cross-references before moving
- Test all links in README and CONTRIBUTING.md
- Update Video Annotation Viewer docs that reference VideoAnnotator

---

## Dependencies & Coordination

### Internal Dependencies
- Batch A/B/C cleanup should merge to master before starting JOSS work (clean baseline)
- JOSS paper (docs/joss.md) needs final updates during documentation reorganization
- CI pipeline must be stable for badge to be meaningful

### External Dependencies
- JOSS submission timeline coordinates with v1.3.0 release date
- Video Annotation Viewer team may need updated documentation links
- External reviewers must be available for documentation validation (2+ people)

### Timeline Coordination
```
Week 1-2:  JOSS Critical (P1) - Installation script, API docs, test coverage
Week 3-4:  Documentation Cleanup (P2) - Reorganization, troubleshooting
Week 4-5:  Script Consolidation (P2) - Audit, diagnostic tools
Week 6:    External validation, PR reviews, release preparation
Week 7-8:  v1.3.0 release + JOSS submission
```

---

## Testing Strategy

### JOSS Requirements Testing
- [ ] Run `scripts/verify_installation.py` on Ubuntu 20.04, 22.04, 24.04
- [ ] Run `scripts/verify_installation.py` on macOS Intel and Apple Silicon
- [ ] Run `scripts/verify_installation.py` on Windows 10/11 (WSL2 and native)
- [ ] External reviewer installs and runs sample pipeline (timed, documented)
- [ ] Verify all API endpoints have docstrings with examples (automated scan)
- [ ] Run pytest with coverage report, verify >80% for core pipelines

### Documentation Quality Testing
- [ ] External reviewer A navigates docs/ and provides feedback
- [ ] External reviewer B attempts contribution using only CONTRIBUTING.md
- [ ] JOSS reviewer completes review checklist (mock review)
- [ ] Check all cross-links in documentation (automated scan)
- [ ] Verify troubleshooting guide covers top 10 GitHub issues

### Script Consolidation Testing
- [ ] Run all scripts in scripts/ folder, document which fail or are obsolete
- [ ] Compare script test coverage to main pytest suite (identify gaps)
- [ ] Test diagnostic CLI on fresh installation (no config files)
- [ ] Test diagnostic CLI with intentional misconfigurations
- [ ] Verify removed scripts have migration notes or archive location

---

## Acceptance Criteria for v1.3.0 Release

### JOSS Readiness (Must-Have)
- ✅ Installation verification script succeeds on 3 platforms (Ubuntu, macOS, Windows)
- ✅ All public API endpoints have docstrings with examples (100% coverage)
- ✅ Test coverage report shows >80% for core pipelines
- ✅ External reviewer completes install + sample pipeline in <15 minutes
- ✅ CI status badge in README

### Documentation Quality (Must-Have)
- ✅ docs/ structure follows installation → usage → development → testing → deployment
- ✅ Troubleshooting guide covers top 10 common issues
- ✅ "Getting Started for Reviewers" guide exists
- ✅ 2+ external reviewers provide positive feedback
- ✅ All generated docs have "DO NOT EDIT" warnings

### Script Quality (Should-Have)
- ✅ scripts/ folder contains only documented, actively maintained tools
- ✅ Essential test scripts migrated to pytest or documented as intentionally separate
- ✅ Diagnostic CLI commands implemented: `videoannotator diagnose [system|gpu|storage|database]`
- ✅ All retained scripts have purpose and usage documentation

### Nice-to-Have (Can Defer to v1.3.1)
- ⚠️ Jupyter notebook examples (if time permits)
- ⚠️ Video tutorials for common workflows
- ⚠️ API client SDKs for Python/JavaScript (out of scope, deferred to v1.4.0)

---

## Open Questions

1. **Timeframe for external reviewer availability?**
   - Need 2+ reviewers for documentation validation
   - Need 1+ reviewer for full JOSS mock review
   - Suggest scheduling early (Week 3-4) to allow iteration time

2. **Script removal approval process?**
   - Who approves script removal decisions?
   - Should we notify all contributors before removing?
   - Archive location: archive/scripts/ or GitHub history only?

3. **Documentation platform preference?**
   - Keep markdown in docs/ folder (current)?
   - Migrate to Read the Docs / GitHub Pages?
   - Generate API docs with Sphinx/mkdocs?

4. **Diagnostic CLI scope boundaries?**
   - How deep should diagnostics go? (e.g., check CUDA version, model file integrity?)
   - Should it auto-fix issues or just report? (suggest: report only for v1.3.0)
   - Include in main CLI or separate tool?

---

## Notes

- JOSS requirements (FR-054 to FR-058) are **P1** because they are **release blockers** for publication
- Documentation cleanup (FR-059 to FR-064) is **P2** because it improves quality but doesn't block basic functionality
- Script consolidation (FR-065 to FR-070) is **P2** because it removes confusion but existing scripts still work
- All three areas (JOSS, docs, scripts) are "spring cleaning" that improves external contributor experience
- Timeline is tight (6-8 weeks total) so strict scope control is essential
- External validation is critical path - schedule early to allow iteration

---

## Related Documents

- [v1.3.0 Main Specification](./spec.md) - Full release specification
- [JOSS Readiness Assessment](./JOSS_READINESS_ASSESSMENT.md) - Detailed JOSS requirements analysis
- [JOSS Review Checklist](https://joss.readthedocs.io/en/latest/review_checklist.html) - Official JOSS criteria

---

## Changelog

- **2025-10-11**: Initial version - Added JOSS, documentation, and script requirements to v1.3.0 spec
