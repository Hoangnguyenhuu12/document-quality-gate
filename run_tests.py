"""
Automated lightweight test runner for document-quality-gate.
Executes all test_* functions across the tests directory.
"""

import inspect
import importlib
import os
import sys
import traceback
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def run_all_tests():
    tests_dir = Path(__file__).resolve().parent / "tests"
    test_files = sorted(tests_dir.glob("test_*.py"))
    
    total_passed = 0
    total_failed = 0
    
    print("=" * 65)
    print("DOCUMENT QUALITY GATE - TEST EXECUTION SUITE")
    print("=" * 65)
    
    for file_path in test_files:
        module_name = f"tests.{file_path.stem}"
        try:
            mod = importlib.import_module(module_name)
        except Exception as e:
            print(f"[ERROR] Failed to import {module_name}: {e}")
            total_failed += 1
            continue
            
        test_funcs = [
            (name, func) for name, func in inspect.getmembers(mod, inspect.isfunction)
            if name.startswith("test_")
        ]
        
        print(f"\n[*] Suite: {file_path.name} ({len(test_funcs)} tests):")
        for name, func in test_funcs:
            sig = inspect.signature(func)
            try:
                if len(sig.parameters) == 0:
                    func()
                elif len(sig.parameters) == 1:
                    # Accepts a tempdir path
                    import tempfile
                    func(Path(tempfile.mkdtemp()))
                print(f"  [PASS] {name}")
                total_passed += 1
            except Exception as e:
                print(f"  [FAIL] {name} -> {e}")
                traceback.print_exc()
                total_failed += 1
                
    print("\n" + "=" * 65)
    print(f"RESULTS: {total_passed} PASSED | {total_failed} FAILED")
    print("=" * 65)
    
    if total_failed > 0:
        sys.exit(1)
    else:
        print("ALL QUALITY GATE SPECIFICATIONS VERIFIED!")

if __name__ == "__main__":
    run_all_tests()
