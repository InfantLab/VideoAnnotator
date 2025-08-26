# Examples CLI Update Checklist for v1.2.1

## 🎯 **Overview**
Post-v1.2.0 cleanup to modernize all examples and documentation to use the new `videoannotator` CLI interface instead of old direct Python script execution patterns.

## 📚 **Files Requiring Updates**

### **High Priority - Documentation**
- [ ] **examples/README.md** - Complete rewrite of CLI usage patterns
  - Replace: `python examples/basic_video_processing.py --video_path video.mp4`  
  - With: `uv run videoannotator job submit video.mp4 --pipelines scene,person,face`

- [ ] **docs/usage/GETTING_STARTED.md** - Verify CLI examples are current ✅ DONE
- [ ] **docs/usage/demo_commands.md** - Update all command examples (if exists)

### **Medium Priority - Example Scripts**
#### **Current Direct Pipeline Examples:**
1. **examples/basic_video_processing.py**
   - **Current**: Direct pipeline imports and argparse
   - **Update**: Add API integration alternative or convert to API example
   
2. **examples/batch_processing.py**
   - **Current**: Direct parallel processing with multiprocessing
   - **Update**: Show API-based batch job submission
   
3. **examples/test_individual_pipelines.py** 
   - **Current**: Direct pipeline testing with argparse
   - **Update**: Use `videoannotator pipelines` and API testing
   
4. **examples/custom_pipeline_config.py**
   - **Current**: Direct pipeline configuration 
   - **Update**: Use `videoannotator config --validate` patterns

5. **examples/diarization_example.py**
   - **Current**: Direct audio pipeline usage
   - **Update**: API-based audio processing example

#### **Research-Specific Examples:**
6. **examples/size_based_analysis_demo.py**
   - **Current**: Custom analysis workflow
   - **Update**: Integrate with API results retrieval

7. **examples/test_laion_voice_pipeline.py**
   - **Current**: Direct pipeline testing
   - **Update**: API-based LAION testing

## 🔄 **Recommended Approach**

### **Strategy: Dual Documentation**
Instead of replacing existing examples, provide both approaches:

#### **Keep Legacy Examples** (for direct pipeline usage)
- Rename with `_legacy` suffix
- Add header explaining when to use direct vs. API approach
- Maintain for users who need direct pipeline integration

#### **Add New Modern Examples**
- `examples/api_job_submission.py` - Complete API workflow
- `examples/api_batch_processing.py` - Multi-video API processing  
- `examples/cli_workflow_demo.py` - Modern CLI usage patterns
- `examples/api_results_analysis.py` - Working with API results

## 📝 **Specific Updates Needed**

### **examples/README.md** - Major Rewrite
```diff
## Quick Start

- ### Old Approach (Direct Pipeline)
+ ### Modern Approach (API Server)
+ ```bash
+ # Start server
+ uv run videoannotator server
+ 
+ # Submit job
+ uv run videoannotator job submit video.mp4 --pipelines scene,person,face
+ 
+ # Check status  
+ uv run videoannotator job status <job_id>
+ 
+ # Get results
+ uv run videoannotator job results <job_id>
+ ```
+ 
+ ### Legacy Approach (Direct Pipeline Usage)
```

### **Common Patterns to Replace:**
```diff
# Command Line Usage:
- python examples/basic_video_processing.py --video_path /path/to/video.mp4
+ uv run videoannotator job submit /path/to/video.mp4 --pipelines scene,person

# Configuration:
- python script.py --config configs/high_performance.yaml
+ uv run videoannotator config --validate configs/high_performance.yaml
+ uv run videoannotator job submit video.mp4 --config configs/high_performance.yaml

# Pipeline Testing:
- python examples/test_individual_pipelines.py --pipeline scene
+ uv run videoannotator pipelines --detailed
+ # Submit specific pipeline job through API

# Batch Processing:
- python examples/batch_processing.py --input_dir videos/ --max_workers 4
+ uv run videoannotator server &
+ # Submit multiple jobs through API or create batch script
```

## 🎯 **Success Criteria**

### **Documentation Quality**
- ✅ New users can follow examples without confusion about CLI vs. direct usage
- ✅ Clear distinction between legacy (direct) and modern (API) approaches
- ✅ All CLI examples use proper `uv run videoannotator` syntax

### **Backward Compatibility**  
- ✅ Existing direct pipeline usage still documented and working
- ✅ Researchers can choose appropriate approach for their needs
- ✅ Migration path clearly explained

### **Developer Experience**
- ✅ Examples run without modification on fresh installs
- ✅ API examples demonstrate complete workflows  
- ✅ Error handling and troubleshooting covered

## ⚡ **Quick Wins (1-2 hours each)**

1. **Update examples/README.md** - Replace CLI patterns, add modern workflow section
2. **Add examples/api_workflow_demo.py** - Simple complete API usage example
3. **Add examples/cli_quick_start.py** - CLI command demonstration script
4. **Update docstrings** - Modern CLI usage in script headers

## 🗓️ **Timeline Estimate**

- **examples/README.md rewrite**: 2-3 hours
- **4 new API example scripts**: 6-8 hours  
- **Legacy script header updates**: 2-3 hours
- **Testing and validation**: 2-3 hours

**Total**: 12-17 hours (1.5-2 developer days)

This work can be done incrementally and doesn't block v1.2.0 release.