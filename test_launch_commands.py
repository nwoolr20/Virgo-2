#!/usr/bin/env python3
"""
Test script to verify all launch_virgo.py commands work correctly
"""

import subprocess
import sys
from pathlib import Path

def test_command(name, cmd, expected_in_output=None, timeout=30):
    """Test a command and check output"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent
        )
        
        output = result.stdout + result.stderr
        print(f"Exit code: {result.returncode}")
        print(f"Output (first 500 chars):\n{output[:500]}")
        
        if expected_in_output:
            if expected_in_output in output:
                print(f"✓ Found expected text: '{expected_in_output}'")
                return True
            else:
                print(f"✗ Missing expected text: '{expected_in_output}'")
                return False
        
        if result.returncode == 0:
            print(f"✓ Command succeeded")
            return True
        else:
            print(f"✗ Command failed with code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ Command timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    print("="*60)
    print("VIRGO LAUNCH SCRIPT COMMAND TESTS")
    print("="*60)
    
    results = {}
    
    # Test 1: Help command
    results['help'] = test_command(
        "Help Command",
        [sys.executable, "launch_virgo.py", "help"],
        expected_in_output="Virgo Neural Field Launch Script",
        timeout=10
    )
    
    # Test 2: Demo command (should run demo_nflm.py)
    results['demo'] = test_command(
        "Demo Command",
        [sys.executable, "launch_virgo.py", "demo"],
        expected_in_output="Neural Field Language Model Demo",
        timeout=120
    )
    
    # Test 3: Test command (runs pytest)
    results['test'] = test_command(
        "Test Command",
        [sys.executable, "launch_virgo.py", "test"],
        expected_in_output="test session starts",
        timeout=120
    )
    
    # Test 4: Evaluate command
    results['evaluate'] = test_command(
        "Evaluate Command",
        [sys.executable, "launch_virgo.py", "evaluate"],
        expected_in_output="Evaluation",
        timeout=60
    )
    
    # Test 5: Chat command (will fail without model, but should show proper error)
    results['chat_no_model'] = test_command(
        "Chat Command (no model)",
        [sys.executable, "launch_virgo.py", "chat"],
        expected_in_output="Model not found",
        timeout=10
    )
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for cmd, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {cmd}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
