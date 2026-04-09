#!/usr/bin/env python3
"""
Fix NumPy 2.0 compatibility in Ditto by replacing deprecated np.atan2 with np.arctan2.

This script automatically patches all Python files in the Ditto repository.
"""
import os
import sys
from pathlib import Path

def main():
    # Find Ditto directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    ditto_dir = project_root / "external_models" / "ditto-talkinghead"
    
    if not ditto_dir.exists():
        print(f"❌ Ditto not found at {ditto_dir}")
        print(f"   Please install Ditto first using: bash sota_benchmarker/scripts/install_models/install_ditto.sh")
        sys.exit(1)
    
    print("=" * 60)
    print("Fixing NumPy 2.0 Compatibility in Ditto")
    print("=" * 60)
    print(f"Ditto directory: {ditto_dir}")
    print()
    
    # Search and replace
    files_fixed = []
    total_replacements = 0
    
    for py_file in ditto_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding='utf-8')
            
            # Count occurrences
            count = content.count("np.atan2")
            
            if count > 0:
                # Replace
                new_content = content.replace("np.atan2", "np.arctan2")
                py_file.write_text(new_content, encoding='utf-8')
                
                files_fixed.append(py_file)
                total_replacements += count
                
                rel_path = py_file.relative_to(ditto_dir)
                print(f"  ✓ Fixed: {rel_path} ({count} occurrence(s))")
        
        except Exception as e:
            print(f"  ⚠️  Warning: Could not process {py_file}: {e}")
    
    print()
    print("=" * 60)
    
    if files_fixed:
        print(f"✅ SUCCESS: Fixed {len(files_fixed)} file(s) with {total_replacements} replacement(s)")
        print()
        print("Files modified:")
        for f in files_fixed:
            print(f"  - {f.relative_to(ditto_dir)}")
    else:
        print("✓ No files needed fixing (already compatible)")
    
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Run benchmark: python sota_benchmarker/benchmark_runner.py --model ditto --precision FP16")
    print("  2. If you see other NumPy errors, run this script again")
    
    return 0 if files_fixed or True else 1

if __name__ == "__main__":
    sys.exit(main())
