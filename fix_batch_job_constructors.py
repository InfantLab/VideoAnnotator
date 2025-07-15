#!/usr/bin/env python3
"""
Advanced Test API Fixes - Phase 1.2: BatchJob Constructor

This script specifically targets the BatchJob constructor issues that the first script missed.
It handles more complex patterns and multi-line constructor calls.
"""

import sys
import re
from pathlib import Path
from typing import List, Tuple

def fix_batch_job_constructors_advanced(file_path: Path) -> Tuple[bool, List[str]]:
    """
    Fix complex BatchJob constructor patterns that include video_id parameter.
    """
    changes = []
    
    if not file_path.exists():
        return False, [f"File not found: {file_path}"]
    
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        
        # Add Path import if needed
        if 'from pathlib import Path' not in content and 'import Path' not in content and 'pathlib' not in content:
            # Find the right place to add import
            lines = content.split('\n')
            import_section_end = 0
            
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')) or line.strip() == '':
                    import_section_end = i
                else:
                    break
            
            # Insert Path import
            lines.insert(import_section_end, 'from pathlib import Path')
            content = '\n'.join(lines)
            changes.append("Added 'from pathlib import Path' import")
        
        # Pattern 1: Simple single-line video_id parameter
        # BatchJob(..., video_id='something', ...)
        pattern1 = r"(\s+)return BatchJob\(\s*([^)]*?)video_id\s*=\s*['\"]([^'\"]+)['\"]([^)]*?)\)"
        def replace1(match):
            indent, before_params, video_id, after_params = match.groups()
            # Remove video_id and add video_path
            before_clean = before_params.rstrip(', ')
            after_clean = after_params.lstrip(', ')
            
            # Build new parameter list
            new_params = []
            if before_clean:
                new_params.append(before_clean)
            new_params.append(f"video_path=Path('{video_id}.mp4')")
            if after_clean:
                new_params.append(after_clean)
            
            return f"{indent}return BatchJob({', '.join(new_params)})"
        
        content = re.sub(pattern1, replace1, content)
        
        # Pattern 2: Multi-line constructor with video_id
        # More complex pattern for multi-line BatchJob constructors
        lines = content.split('\n')
        in_batch_job_constructor = False
        constructor_lines = []
        start_line_idx = None
        
        for i, line in enumerate(lines):
            if 'BatchJob(' in line and not line.strip().endswith(')'):
                in_batch_job_constructor = True
                constructor_lines = [line]
                start_line_idx = i
            elif in_batch_job_constructor:
                constructor_lines.append(line)
                if ')' in line and line.count('(') <= line.count(')'):
                    # End of constructor
                    in_batch_job_constructor = False
                    
                    # Check if this constructor has video_id
                    full_constructor = '\n'.join(constructor_lines)
                    if 'video_id=' in full_constructor:
                        # Fix this constructor
                        fixed_constructor = fix_multiline_constructor(full_constructor)
                        if fixed_constructor != full_constructor:
                            # Replace the lines
                            for j, fixed_line in enumerate(fixed_constructor.split('\n')):
                                lines[start_line_idx + j] = fixed_line
                            changes.append(f"Fixed multi-line BatchJob constructor at line {start_line_idx + 1}")
                    
                    constructor_lines = []
                    start_line_idx = None
        
        content = '\n'.join(lines)
        
        # Pattern 3: Simple inline fixes for missed cases
        # video_id=kwargs.get('video_id', 'test_video')
        if "video_id=kwargs.get('video_id'" in content:
            content = content.replace(
                "video_id=kwargs.get('video_id', 'test_video')",
                "# video_id is a computed property from video_path"
            )
            changes.append("Removed video_id parameter assignment from kwargs")
        
        # Pattern 4: Direct video_id assignments in constructors
        content = re.sub(
            r"video_id\s*=\s*['\"]([^'\"]+)['\"]",
            lambda m: f"video_path=Path('{m.group(1)}.mp4')",
            content
        )
        
        # Write back if changed
        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            return True, changes
        else:
            return False, ["No BatchJob constructor issues found"]
            
    except Exception as e:
        return False, [f"Error processing file: {e}"]

def fix_multiline_constructor(constructor_text: str) -> str:
    """Fix a multi-line BatchJob constructor."""
    lines = constructor_text.split('\n')
    fixed_lines = []
    
    for line in lines:
        if 'video_id=' in line and 'video_path=' not in line:
            # Extract the video_id value
            match = re.search(r"video_id\s*=\s*['\"]([^'\"]+)['\"]", line)
            if match:
                video_id = match.group(1)
                # Replace video_id with video_path
                fixed_line = re.sub(
                    r"video_id\s*=\s*['\"][^'\"]+['\"]",
                    f"video_path=Path('{video_id}.mp4')",
                    line
                )
                fixed_lines.append(fixed_line)
            else:
                # Comment out the line if we can't parse it
                fixed_lines.append('            # ' + line.strip() + '  # FIXME: video_id parameter not supported')
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def main():
    """Apply advanced BatchJob constructor fixes."""
    print("🔧 VideoAnnotator Test Suite API Fixes - Phase 1.2: BatchJob Constructors")
    print("=" * 70)
    
    # Get workspace root
    workspace_root = Path(__file__).parent
    tests_dir = workspace_root / "tests"
    
    if not tests_dir.exists():
        print(f"❌ Tests directory not found: {tests_dir}")
        return False
    
    # Files with known BatchJob video_id constructor issues
    test_files = [
        tests_dir / "test_progress_tracker_real.py",
        tests_dir / "test_recovery_real.py",
        tests_dir / "test_integration_simple.py",
        tests_dir / "test_batch_orchestrator_real.py",
        tests_dir / "test_batch_integration.py",
    ]
    
    # Apply fixes
    fixed_count = 0
    total_changes = 0
    
    for test_file in test_files:
        if test_file.exists():
            print(f"\n🔧 Processing: {test_file.name}")
            fixed, changes = fix_batch_job_constructors_advanced(test_file)
            if fixed:
                print(f"  ✅ Applied {len(changes)} fixes:")
                for change in changes:
                    print(f"    - {change}")
                fixed_count += 1
                total_changes += len(changes)
            else:
                print(f"  ⚪ No issues found")
        else:
            print(f"⚠️  File not found: {test_file.name}")
    
    print(f"\n✅ Applied {total_changes} fixes to {fixed_count} files")
    print("\n📋 Test one of the fixed files:")
    print("python -m pytest tests/test_progress_tracker_real.py::TestProgressTrackerReal::test_get_status_empty_jobs -v")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
