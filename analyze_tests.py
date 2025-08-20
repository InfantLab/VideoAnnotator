#!/usr/bin/env python3
"""
Test Suite Analysis Script for VideoAnnotator
Analyzes the current test structure to help organize the test suite.
"""
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
import re

def analyze_test_collection():
    """Analyze what tests are collected and their distribution."""
    print("Analyzing Test Suite Structure...")
    
    # Get test collection output
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', 'tests/', '--co', '-q'],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    )
    
    if result.returncode != 0:
        print("❌ Error collecting tests:")
        print(result.stderr)
        return {}
    
    lines = result.stdout.splitlines()
    total_tests = len([line for line in lines if '::' in line])
    
    print(f"Total Tests Collected: {total_tests}")
    
    # Analyze by file
    file_counts = defaultdict(int)
    category_counts = defaultdict(int)
    
    for line in lines:
        if '::' in line:
            file_part = line.split('::')[0]
            file_counts[file_part] += 1
            
            # Categorize by file pattern
            if 'batch' in file_part:
                category_counts['Batch Processing'] += 1
            elif 'pipeline' in file_part or 'face' in file_part or 'person' in file_part or 'audio' in file_part or 'scene' in file_part:
                category_counts['Pipeline Tests'] += 1
            elif 'integration' in file_part:
                category_counts['Integration Tests'] += 1
            elif 'storage' in file_part:
                category_counts['Storage Tests'] += 1
            elif 'laion' in file_part:
                category_counts['LAION Tests'] += 1
            else:
                category_counts['Other/Misc'] += 1
    
    print("\nTests by File:")
    for file, count in sorted(file_counts.items()):
        print(f"  {file}: {count} tests")
    
    print("\nTests by Category:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count} tests")
    
    return file_counts, category_counts

def analyze_test_patterns():
    """Analyze test naming patterns and types."""
    print("\nAnalyzing Test Patterns...")
    
    test_files = list(Path('tests').glob('test_*.py'))
    
    patterns = {
        'Unit Tests': 0,
        'Integration Tests': 0, 
        'Modern Pipeline Tests': 0,
        'Legacy Tests': 0,
        'Real/Live Tests': 0,
        'Mock/Placeholder Tests': 0
    }
    
    for test_file in test_files:
        name = test_file.name
        if 'integration' in name:
            patterns['Integration Tests'] += 1
        elif 'modern' in name:
            patterns['Modern Pipeline Tests'] += 1
        elif 'real' in name:
            patterns['Real/Live Tests'] += 1
        elif any(x in name for x in ['batch', 'storage', 'recovery', 'types']):
            patterns['Unit Tests'] += 1
        else:
            patterns['Legacy Tests'] += 1
    
    print("Test File Patterns:")
    for pattern, count in patterns.items():
        print(f"  {pattern}: {count} files")
    
    return patterns

def propose_organization():
    """Propose a new test organization structure."""
    print("\nProposed Test Organization Structure:")
    
    structure = {
        "tests/unit/": [
            "Core data structures (BatchJob, types, etc.)",
            "Individual pipeline components",
            "Utility functions",
            "Configuration validation"
        ],
        "tests/integration/": [
            "Pipeline-to-pipeline interactions", 
            "Storage system integration",
            "Batch orchestrator workflows"
        ],
        "tests/pipelines/": [
            "Full pipeline processing tests",
            "Performance benchmarks",
            "Real video processing tests"
        ],
        "tests/experimental/": [
            "Placeholder tests for future features",
            "Research/prototype testing",
            "Incomplete implementations"
        ],
        "tests/fixtures/": [
            "Test data and mock objects",
            "Shared test utilities",
            "Test video samples"
        ]
    }
    
    for directory, contents in structure.items():
        print(f"\n{directory}")
        for item in contents:
            print(f"  - {item}")

if __name__ == "__main__":
    print("=" * 60)
    print("VideoAnnotator Test Suite Analysis")
    print("=" * 60)
    
    try:
        file_counts, category_counts = analyze_test_collection()
        patterns = analyze_test_patterns()
        propose_organization()
        
        print("\nAnalysis Complete!")
        print(f"Next steps: Reorganize {sum(file_counts.values())} tests into logical structure")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()