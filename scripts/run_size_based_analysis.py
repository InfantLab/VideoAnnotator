#!/usr/bin/env python3
"""
Size-Based Person Analysis Script

Command-line script to run size-based person analysis on existing person tracking results
or integrate with the person tracking pipeline.
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.size_based_person_analysis import (
    SizeBasedPersonAnalyzer,
    run_size_based_analysis,
    print_analysis_results
)


def load_person_annotations(file_path: str):
    """Load person annotations from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON formats
    if 'annotations' in data:
        annotations = data['annotations']
    elif isinstance(data, list):
        annotations = data
    else:
        raise ValueError(f"Unsupported JSON format in {file_path}")
    
    # Filter for person annotations only
    person_annotations = [
        ann for ann in annotations 
        if ann.get('category_id') == 1 and ann.get('person_id')
    ]
    
    print(f"Loaded {len(person_annotations)} person annotations from {file_path}")
    return person_annotations


def save_analysis_results(results: dict, output_path: str):
    """Save analysis results to JSON file."""
    output_data = {
        "analysis_type": "size_based_person_analysis",
        "results": results,
        "summary": {
            "total_persons": len(results),
            "infants": sum(1 for info in results.values() if info['label'] == 'infant'),
            "parents": sum(1 for info in results.values() if info['label'] == 'parent'),
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Analysis results saved to {output_path}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run size-based person analysis on tracking data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run analysis on person tracking results
  python scripts/run_size_based_analysis.py --input results/person_tracking.json
  
  # Use custom threshold and confidence
  python scripts/run_size_based_analysis.py --input results/person_tracking.json \\
    --threshold 0.5 --confidence 0.8
  
  # Save results to file
  python scripts/run_size_based_analysis.py --input results/person_tracking.json \\
    --output results/size_analysis.json
        """
    )
    
    parser.add_argument(
        '--input', '-i', 
        required=True,
        help='Input JSON file with person tracking annotations'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output JSON file for analysis results (optional)'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.4,
        help='Height threshold for adult/child classification (default: 0.4)'
    )
    
    parser.add_argument(
        '--confidence', '-c',
        type=float,
        default=0.7,
        help='Confidence score for automatic labels (default: 0.7)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress detailed output, only show summary'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {args.input} does not exist")
        sys.exit(1)
    
    try:
        # Load person annotations
        annotations = load_person_annotations(args.input)
        
        if not annotations:
            print("Warning: No person annotations found in input file")
            sys.exit(0)
        
        # Run size-based analysis
        print(f"Running size-based analysis with threshold={args.threshold}, confidence={args.confidence}")
        
        results = run_size_based_analysis(
            annotations,
            height_threshold=args.threshold,
            confidence=args.confidence
        )
        
        if not results:
            print("No persons could be analyzed (insufficient data or missing person_ids)")
            sys.exit(0)
        
        # Print results
        if not args.quiet:
            print_analysis_results(results)
        else:
            # Just print summary
            infant_count = sum(1 for info in results.values() if info['label'] == 'infant')
            parent_count = sum(1 for info in results.values() if info['label'] == 'parent')
            print(f"Analysis complete: {infant_count} infants, {parent_count} parents")
        
        # Save results if requested
        if args.output:
            save_analysis_results(results, args.output)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()